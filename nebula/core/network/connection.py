import asyncio
import base64
import gc
import logging
import time
from geopy import distance
import sys
import json
import zlib, bz2, lzma, base64

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class Connection:
    DEFAULT_FEDERATED_ROUND = -1

    def __init__(
        self,
        cm: "CommunicationsManager",
        reader,
        writer,
        id,
        host,
        port,
        direct=True,
        active=True,
        compression="zlib",
        config=None,
    ):
        self.cm = cm
        self.reader = reader
        self.writer = writer
        self.id = str(id)
        self.host = host
        self.port = port
        self.addr = f"{host}:{port}"
        self.direct = direct
        self.active = active
        self.last_active = time.time()
        self.compression = compression
        self.config = config

        self.federated_round = Connection.DEFAULT_FEDERATED_ROUND
        self.latitude = None
        self.longitude = None
        self.loop = asyncio.get_event_loop()
        self.read_task = None

        self.EOT_CHAR = b"\x01\x02\x03\x04"
        self.COMPRESSION_CHAR = b"\x05\x06\x07\x08"
        self.DATA_TYPE_PREFIXES = {"pb": b"\x10\x11\x12\x13", "string": b"\x14\x15\x16\x17", "json": b"\x18\x19\x20\x21", "bytes": b"\x22\x23\x24\x25"}

        logging.info(f"Connection [established]: {self.addr} (id: {self.id}) (active: {self.active}) (direct: {self.direct})")

    def __str__(self):
        return f"Connection to {self.addr} (id: {self.id}) (active: {self.active}) (last active: {self.last_active}) (direct: {self.direct})"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        self.stop()

    def get_addr(self):
        return self.addr

    def get_federated_round(self):
        return self.federated_round

    def update_round(self, federated_round):
        self.federated_round = federated_round

    def update_geolocation(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
        self.config.participant["mobility_args"]["neighbors_distance"][self.addr] = self.compute_distance_myself()

    def get_geolocation(self):
        if self.latitude is None or self.longitude is None:
            raise ValueError("Geo-location not set for this neighbor")
        return self.latitude, self.longitude

    def get_neighbor_distance(self):
        if self.addr not in self.config.participant["mobility_args"]["neighbors_distance"]:
            return None
        return self.config.participant["mobility_args"]["neighbors_distance"][self.addr]

    def compute_distance(self, latitude, longitude):
        distance_m = distance.distance((self.latitude, self.longitude), (latitude, longitude)).m
        return distance_m

    def compute_distance_myself(self):
        distance_m = self.compute_distance(self.config.participant["mobility_args"]["latitude"], self.config.participant["mobility_args"]["longitude"])
        return distance_m

    def get_ready(self):
        return True if self.federated_round != Connection.DEFAULT_FEDERATED_ROUND else False

    def get_direct(self):
        return self.direct

    def set_direct(self, direct):
        # config.participant["network_args"]["neighbors"] only contains direct neighbors (frotend purposes)
        if direct:
            self.config.add_neighbor_from_config(self.addr)
        else:
            self.config.remove_neighbor_from_config(self.addr)
        self.last_active = time.time()
        self.direct = direct

    def set_active(self, active):
        self.active = active
        self.last_active = time.time()

    def is_active(self):
        return self.active

    def get_last_active(self):
        return self.last_active

    async def start(self):
        self.read_task = asyncio.create_task(self.handle_incoming_message(), name=f"Connection {self.addr} reader")

    async def stop(self):
        logging.info(f"❗️  Connection [stopped]: {self.addr} (id: {self.id})")
        if self.read_task is not None:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                logging.error(f"❗️  {self} cancelled...")
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()

    async def compress(self, data, compression):
        compressed = data
        try:
            if compression == "zlib":
                compressed = base64.b64encode(zlib.compress(data, 6) + b"zlib")
            elif compression == "bzip2":
                compressed = base64.b64encode(bz2.compress(data) + b"bzip2")
            elif compression == "lzma":
                compressed = base64.b64encode(lzma.compress(data) + b"lzma")
            else:
                logging.error(f"❗️  Unknown compression: {compression}")
                return None
        except Exception as e:
            logging.error(f"❗️  Compression error: {e}")
        logging.debug(f"Compression: {int(10000 * len(compressed) / len(data)) / 100}%")
        return compressed

    async def decompress(self, compressed):
        try:
            compressed = base64.b64decode(compressed)
        except Exception as e:
            logging.error(f"decompress: b64decode exception: {e}")
            return None
        try:
            if compressed[-4:] == b"zlib":
                compressed = zlib.decompress(compressed[:-4])
            elif compressed[-5:] == b"bzip2":
                compressed = bz2.decompress(compressed[:-5])
            elif compressed[-4:] == b"lzma":
                compressed = lzma.decompress(compressed[:-4])
        except Exception as e:
            logging.error(f"❗️  Compression error: {e}")
            return None
        return compressed

    async def send(self, data, pb=True, encoding_type="utf-8", compression="none"):
        try:
            logging.debug(f"Size of data (before compression -- if applied): {format(sys.getsizeof(data)/1024/1024, '.10f')} MB")
            if pb:
                data_prefix = self.DATA_TYPE_PREFIXES["pb"]
                encoded_data = data
            elif isinstance(data, str):
                data_prefix = self.DATA_TYPE_PREFIXES["string"]
                encoded_data = data.encode(encoding_type)
            elif isinstance(data, dict):
                data_prefix = self.DATA_TYPE_PREFIXES["json"]
                encoded_data = json.dumps(data).encode(encoding_type)
            elif isinstance(data, bytes):
                data_prefix = self.DATA_TYPE_PREFIXES["bytes"]
                encoded_data = data
            else:
                logging.error(f"❗️  Unknown data type to send: {type(data)}")
                return

            if compression != "none":
                encoded_data = await self.compress(encoded_data, compression)
                if encoded_data is None:
                    return
                data_to_send = data_prefix + encoded_data + self.COMPRESSION_CHAR + self.EOT_CHAR
            else:
                data_to_send = data_prefix + encoded_data + self.EOT_CHAR

            chunk_size = 100 * 1024 * 1024 # 100 MB
            total_size = len(data_to_send)

            for i in range(0, total_size, chunk_size):
                chunk = data_to_send[i : i + chunk_size]
                self.writer.write(chunk)
                await self.writer.drain()

            logging.debug(f"Size of data (after compression -- if applied): {format(sys.getsizeof(encoded_data)/1024/1024, '.10f')} MB")
        except Exception as e:
            logging.error(f"❗️  Error sending data to node: {e}")
            await self.stop()

    async def retrieve_message(self, message):
        try:
            data_type_prefix = message[0:4]
            message = message[4:]
            if message[-len(self.COMPRESSION_CHAR) :] == self.COMPRESSION_CHAR:
                message = await self.decompress(message[: -len(self.COMPRESSION_CHAR)])
                if message is None:
                    return None
            if data_type_prefix == self.DATA_TYPE_PREFIXES["pb"]:
                logging.debug(f"Received a successful message (protobuf)")
                asyncio.create_task(self.cm.handle_incoming_message(message, self.addr), name=f"Connection {self.addr} message handler")
            elif data_type_prefix == self.DATA_TYPE_PREFIXES["string"]:
                logging.debug(f"Received message (string): {message.decode('utf-8')}")
            elif data_type_prefix == self.DATA_TYPE_PREFIXES["json"]:
                logging.debug(f"Received message (json): {json.loads(message.decode('utf-8'))}")
            elif data_type_prefix == self.DATA_TYPE_PREFIXES["bytes"]:
                logging.debug(f"Received message (bytes): {message}")
                return message
            else:
                logging.error(f"❗️  Unknown data type prefix: {data_type_prefix}")
                return None
        except Exception as e:
            logging.error(f"❗️  Error retrieving message: {e}")
            return None
        finally:
            del message
            gc.collect()

    async def handle_incoming_message(self):
        try:
            buffer = bytearray()
            chunk_size = 100 * 1024 * 1024 # 100 MB
            max_buffer_size = 1024 * 1024 * 1024 # 1 GB
            while True:
                try:
                    chunk = await self.reader.read(chunk_size)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    logging.debug(f"Size of buffer: {format(sys.getsizeof(buffer)/1024/1024, '.10f')} MB")

                    if len(buffer) > max_buffer_size:
                        logging.warning(f"❗️  Buffer size exceeded maximum size: {max_buffer_size}")

                    while True:
                        eot_pos = buffer.find(self.EOT_CHAR)
                        if eot_pos < 0:
                            break
                        message = buffer[:eot_pos]
                        buffer = buffer[eot_pos + len(self.EOT_CHAR) :]
                        logging.debug(f"Size of message: {len(message)/1024/1024:.10f} MB")
                        asyncio.create_task(self.retrieve_message(bytes(message)))
                        logging.debug(f"Size of buffer (after message retrieval): {format(sys.getsizeof(buffer)/1024/1024, '.10f')} MB")

                except asyncio.IncompleteReadError:
                    logging.error(f"❗️  Incomplete read error")
                    break
                except Exception as e:
                    logging.error(f"❗️  Error handling connection: {e}")
                    break
        except asyncio.CancelledError:
            logging.error(f"❗️  {self} cancelled...")
        except Exception as e:
            logging.error(f"❗️  Error handling connection: {e}")
        finally:
            logging.error(f"❗️  {self} stopped...")
            await self.stop()
