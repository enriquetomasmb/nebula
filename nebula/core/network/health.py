import asyncio
import logging
import threading
import time
from nebula.addons.functions import print_msg_box
from typing import TYPE_CHECKING

from nebula.core.pb import nebula_pb2

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class Health(threading.Thread):
    def __init__(self, addr, config, cm: "CommunicationsManager"):
        threading.Thread.__init__(self, daemon=True, name="health_thread-" + config.participant["device_args"]["name"])
        print_msg_box(msg=f"Starting health thread...", indent=2, title="Health thread")
        self.addr = addr
        self.config = config
        self.cm = cm
        self.period = self.config.participant["health_args"]["health_interval"]
        self.alive_interval = self.config.participant["health_args"]["send_alive_interval"]
        self.timeout = self.config.participant["health_args"]["alive_timeout"]

    def run(self):
        loop = asyncio.new_event_loop()
        # loop.set_debug(True)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_health())
        loop.close()

    async def run_health(self):
        await asyncio.sleep(self.config.participant["health_args"]["grace_time_health"])
        # Set all connections to active at the beginning of the health thread
        for conn in self.cm.connections.values():
            conn.set_active(True)
        while True:
            current_time = time.time()
            if len(self.cm.connections) > 0:
                message = self.cm.mm.generate_control_message(nebula_pb2.ControlMessage.Action.ALIVE, log="Alive message")
                self.cm.get_connections_lock().acquire()
                current_connections = list(self.cm.connections.values())
                self.cm.get_connections_lock().release()
                for conn in current_connections:
                    if conn.get_direct():
                        if current_time - conn.get_last_active() > self.timeout:
                            logging.error(f"⬅️ 🕒  Heartbeat timeout for {conn.get_addr()}...")
                            await self.cm.disconnect(conn.get_addr(), mutual_disconnection=False)
                        else:
                            try:
                                logging.info(f"🕒  Sending alive message to {conn.get_addr()}...")
                                corutine = conn.send(data=message)
                                asyncio.run_coroutine_threadsafe(corutine, loop=conn.loop)
                            except Exception as e:
                                logging.error(f"❗️  Cannot send alive message to {conn.get_addr()}. Error: {str(e)}")
                                await self.cm.disconnect(conn.get_addr(), mutual_disconnection=False)
                    await asyncio.sleep(self.alive_interval)
            await asyncio.sleep(self.period)

    def alive(self, source):
        current_time = time.time()
        self.cm.get_connections_lock().acquire()
        try:
            conn = self.cm.connections[source]
            if conn.get_last_active() < current_time:
                logging.debug(f"🕒  Updating last active time for {source}")
                conn.set_active(True)
        finally:
            self.cm.get_connections_lock().release()
