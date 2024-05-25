import asyncio
import logging
import threading
from nebula.addons.functions import print_msg_box
from typing import TYPE_CHECKING

from nebula.core.pb import nebula_pb2


if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class Discoverer(threading.Thread):
    def __init__(self, addr, config, cm: "CommunicationsManager"):
        threading.Thread.__init__(self, daemon=True, name="discoverer_thread-" + config.participant["device_args"]["name"])
        print_msg_box(msg=f"Starting discoverer thread...", indent=2, title="Discoverer thread")
        self.addr = addr
        self.config = config
        self.cm = cm
        self.grace_time = self.config.participant["discoverer_args"]["grace_time_discovery"]
        self.period = self.config.participant["discoverer_args"]["discovery_frequency"]
        self.interval = self.config.participant["discoverer_args"]["discovery_interval"]

    def run(self):
        loop = asyncio.new_event_loop()
        # loop.set_debug(True)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_discover())
        loop.close()

    async def run_discover(self):
        if self.config.participant["scenario_args"]["federation"] == "CFL":
            logging.info("🔍  Federation is CFL. Discoverer is disabled...")
            return
        await asyncio.sleep(self.grace_time)
        while True:
            if len(self.cm.connections) > 0:
                latitude = self.config.participant["mobility_args"]["latitude"]
                longitude = self.config.participant["mobility_args"]["longitude"]
                message = self.cm.mm.generate_discovery_message(action=nebula_pb2.DiscoveryMessage.Action.DISCOVER, latitude=latitude, longitude=longitude)
                try:
                    logging.debug(f"🔍  Sending discovery message to neighbors...")
                    await self.cm.send_message_to_neighbors(message, self.cm.get_all_addrs_current_connections(), self.interval)
                except Exception as e:
                    logging.error(f"🔍  Cannot send discovery message to neighbors. Error: {str(e)}")
            await asyncio.sleep(self.period)
