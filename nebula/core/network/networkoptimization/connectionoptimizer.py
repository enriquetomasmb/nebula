import asyncio
import logging
from nebula.core.network.connection import Connection
from nebula.core.network.networkoptimization.networkoptimizer import NetworkOptimizer
from nebula.core.utils.locker import Locker
import heapq
import time

PRIORITIES = {'HIGH': 30, 'MEDIUM': 20, 'LOW': 10}


class ConnectionOptimizer:
    def __init__(
        self,
        network_optimizer : NetworkOptimizer
    ):
        self._network_optimizer = network_optimizer
        self.connection_heap = []  # Heap: (expire_time, priority, connection)
        self.active_connections = {}  
        self.connection_heap_lock = Locker(name="connection_heap_lock", async_lock=True)  
        self._wake_up_event = asyncio.Event()
        self._running = True  

    @property
    def no(self):
        return self._network_optimizer

    async def update_connection_activity(self, connection: Connection):
        """
            Add new connection timeout to heap
        """
        priority = connection.get_prio()
        timeout = self._get_timeout_for_priority(priority)
        expire_time = time.time() + timeout
        async with self.connection_heap_lock:
            self.active_connections[connection] = (expire_time, priority, True)  # Activa
            heapq.heappush(self.connection_heap, (expire_time, priority, connection))
            self._wake_up_event.set()

    async def set_connection_inactivity(self, connection):
        """
            Set inactive state to a connection
        """
        async with self.connection_heap_lock:
            if connection in self.active_connections:
                # set conection as inactive
                self.active_connections[connection] = (*self.active_connections[connection][:2], False)  
            self._wake_up_event.set()  

    async def stop_daemon(self):
        self._running = False
    
    async def start_daemon(self):
        self._running = True
        while self._running:
            logging.info("Wake up | Connection optimizer daemon...")
            await self._check_timeouts()
            await self._wait_for_next_expiration()

    async def _check_timeouts(self):
        """
            Check to remove expired connections
        """
        current_time = time.time()
        async with self.connection_heap_lock:
            while self.connection_heap and self.connection_heap[0][0] <= current_time:
                expire_time, priority, connection = heapq.heappop(self.connection_heap)
                # Revisa si la conexión está activa
                if connection in self.active_connections:
                    _, _, is_active = self.active_connections[connection]
                    if is_active and self.active_connections[connection][0] == expire_time:
                        logging.info(f"Closing | Connection: {connection.get_addr()} (priority: {priority}) has expired...")
                        del self.active_connections[connection]

    async def _wait_for_next_expiration(self):
        """
            Sleep until new connection is stored or lower timeout passed
        """
        async with self.connection_heap_lock:
            if not self.connection_heap:
                self._wake_up_event.clear() 
                await self._wake_up_event.wait()
                return
            next_expiration = self.connection_heap[0][0]
            
        sleep_duration = max(0, next_expiration - time.time())

        try:
            await asyncio.wait_for(self._wake_up_event.wait(), timeout=sleep_duration)
        except asyncio.TimeoutError:
            pass
        finally:
            self._wake_up_event.clear()

    def _get_timeout_for_priority(self, priority):
        """
            Priority timeouts
        """
        try:
            return PRIORITIES[priority]
        except KeyError:
            logging.error(f"Not allowed: {priority}. PRIORITIES: {list(PRIORITIES.keys())}")
            raise ValueError()