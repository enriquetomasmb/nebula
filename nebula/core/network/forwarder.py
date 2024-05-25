import asyncio
import logging
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING
from nebula.addons.functions import print_msg_box
from nebula.core.utils.locker import Locker

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class Forwarder(threading.Thread):
    def __init__(self, config, cm: "CommunicationsManager"):
        threading.Thread.__init__(
            self,
            daemon=True,
            name="forwarder_thread-" + config.participant["device_args"]["name"],
        )
        print_msg_box(msg=f"Starting forwarder thread...", indent=2, title="Forwarder thread")
        self.config = config
        self.cm = cm
        self.pending_messages = Queue()
        self.pending_messages_lock = Locker("pending_messages_lock", verbose=False)

        self.interval = self.config.participant["forwarder_args"]["forwarder_interval"]
        self.number_forwarded_messages = self.config.participant["forwarder_args"]["number_forwarded_messages"]
        self.messages_interval = self.config.participant["forwarder_args"]["forward_messages_interval"]

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(self.run_forwarder())

        loop.close()

    async def run_forwarder(self):
        if self.config.participant["scenario_args"]["federation"] == "CFL":
            logging.info("🔁  Federation is CFL. Forwarder is disabled...")
            return
        while True:
            # logging.debug(f"🔁  Pending messages: {self.pending_messages.qsize()}")
            start_time = time.time()
            self.pending_messages_lock.acquire()
            await self.process_pending_messages(messages_left=self.number_forwarded_messages)
            self.pending_messages_lock.release()
            sleep_time = max(0, self.interval - (time.time() - start_time))
            await asyncio.sleep(sleep_time)

    async def process_pending_messages(self, messages_left):
        while messages_left > 0 and not self.pending_messages.empty():
            msg, neighbors = self.pending_messages.get()
            for neighbor in neighbors[:messages_left]:
                if neighbor not in self.cm.connections:
                    continue
                try:
                    logging.debug(f"🔁  Sending message (forwarding) --> to {neighbor}")
                    await self.cm.send_message(neighbor, msg)
                except Exception as e:
                    logging.error(f"🔁  Error forwarding message to {neighbor}. Error: {str(e)}")
                    pass
                await asyncio.sleep(self.messages_interval)
            messages_left -= len(neighbors)
            if len(neighbors) > messages_left:
                logging.debug(f"🔁  Putting message back in queue for forwarding to the remaining neighbors")
                self.pending_messages.put((msg, neighbors[messages_left:]))

    def forward(self, msg, addr_from):
        if self.config.participant["scenario_args"]["federation"] == "CFL":
            logging.info("🔁  Federation is CFL. Forwarder is disabled...")
            return
        try:
            self.pending_messages_lock.acquire()
            pending_nodes_to_send = [n for n in self.cm.get_addrs_current_connections(only_direct=True) if n != addr_from]
            logging.debug(f"🔁  Puting message in queue for forwarding to {pending_nodes_to_send}")
            self.pending_messages.put((msg, pending_nodes_to_send))
        except Exception as e:
            logging.error(f"🔁  Error forwarding message. Error: {str(e)}")
        finally:
            self.pending_messages_lock.release()
