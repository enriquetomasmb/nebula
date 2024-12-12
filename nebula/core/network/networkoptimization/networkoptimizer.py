import asyncio
import logging
import time
from nebula.core.network.connection import Connection
from nebula.core.network.networkoptimization.connectionoptimizer import ConnectionOptimizer
from nebula.core.network.networkoptimization.timergenerator import TimerGenerator
from nebula.core.network.communications import CommunicationsManager
from nebula.core.utils.locker import Locker


class NetworkOptimizer:
    def __init__(
        self, 
        communication_manager, 
        vanilla_max_timer, 
        adaptative_timeouts=False
    ):
        self._communications_manager = communication_manager
        self._connection_optimizer = ConnectionOptimizer()
        self._adaptative_timeouts = adaptative_timeouts
        self.max_time_to_wait = vanilla_max_timer
        self._timer_generator = None #TimerGenerator(self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False), self.max_time_to_wait, 80)
    
    @property
    def cm(self) -> CommunicationsManager:
        return self._communications_manager
    
    @property
    def co(self) -> ConnectionOptimizer:
        return self._connection_optimizer
    
    @property
    def tg(self) -> TimerGenerator:
        return self._timer_generator
    
    async def info_received_from_connection(self, connection : Connection):
        await self.co.update_connection_activity(connection)
        
    async def model_update_received_from_connection(self, source):
        arrived_time = time.time()
        await self.tg.receive_update(source, arrived_time)
        
    async def on_closing_connection(self, connection : Connection):
        await self.co.set_connection_inactivity(connection)
        
    async def connection_timeout_expired(self, connection):
        pass    
        
    async def start_connection_cleaner(self):
        self.co.start_daemon()
    
    async def process_direct_connection(self, source, closed=False):
        await self.tg.update_node(source, remove=closed)
        
    async def get_round_timeout(self):
        return await self.tg.get_timer()
    
    async def on_round_end(self):
        await self.tg.on_round_end()    