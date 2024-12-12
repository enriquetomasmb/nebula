import asyncio
import numpy as np
from collections import deque
import time
from nebula.core.utils.locker import Locker
import logging

class TimerGenerator():
    def __init__(
        self,
        nodes,
        max_timer_value,
        acceptable_percent,
        round = None,
        initial_timer_value = None,  
        max_historic_size = 10,
        adaptative=False
        ):
        logging.info("🌐  Initializing Timer Generator..")
        self.waiting_time = initial_timer_value if initial_timer_value != None else max_timer_value
        self.max_timer_value = max_timer_value
        self.acceptable_percent = acceptable_percent
        self.max_historic_size = max_historic_size
        self.round_completed = round
        self.nodes_historic = {node_id: deque(maxlen=self.max_historic_size) for node_id in nodes} 
        self.adaptative = adaptative
        self.max_updates_number = len(self.nodes_historic)
        self.updates_receive_from_nodes = set()
        self.n_updates_receive = 0
        self.last_update_receive_time = 0
        self.start_moment = 0
        self.update_lock = Locker(name="update_lock")
        #self.all_updates_received = asyncio.Condition()
        
    def get_stop_condition(self):
        return True #self.all_updates_received     
        
    async def get_timer(self):
        self.round_completed = self.round_completed + 1
        sm = time.time()
        self.start_moment = round(sm, 2)
        return self.waiting_time

    async def update_node(self, node, remove=False):
        if remove:
            self.nodes_historic.pop(node, None)
            self.max_updates_number -= 1
        else:
            self.nodes_historic.update({node: deque(maxlen=self.max_historic_size)})
            self.max_updates_number += 1        

    async def receive_update(self, node_id, node_response_time):
        """
            In this function the response time is saved in the historic, structures are updated
            
        Args:
            node_id : node addr
            node_response_time : the time when the update was received
        """
        nrt = round(node_response_time,2)
        t_n = nrt - self.start_moment if nrt - self.start_moment >= 0 else 0
        async with self.update_lock:
            self.n_updates_receive +=1
            self.updates_receive_from_nodes.add(node_id)                    # this node has send update
            self.nodes_historic[node_id].append(t_n)                        # add time
            if self.n_updates_receive == self.max_updates_number:           # it means all updates are being receive
                self.last_update_receive_time = t_n 
                #async with self.all_updates_received:               
                #    self.all_updates_received.notify_all() 

    async def on_round_end(self):
        await self._adjust_timer()

    async def _adjust_timer(self):
        """
            The process of adjusting the timer is simple. if adaptative is not set up it will use the MAX_TIMER all the time.
            If not, the strategy will depend on the percent of updates receive the last round.
                -updates < 25%:
                    timer = MAX_TIMER, the results are so bad, then we need an aggresive strategy
                -updates < 75%
                    timer = last_timer + 40%, big increase, because the results are not good enough yet
                -updates < 100%
                    if updates < acceptable_percent
                        if the historic is enough it will use EMA * 1.25, else last_timer * 1.20
                    acceptable_percent
                        if the historic is enough it will use EMA * 1.05, else last_timer * 1.15
                -all updates received
                    we will reduce the timer the min value between 10% of the time wasted and last_updated_time * 1.2, trying
                    to adjust the timer to not waste time when bad variations occur
        """       
        self._complete_data() # fill not receive data for historic
        
        if self.adaptative:         
            if self.n_updates_receive == self.max_updates_number:
                update_times = []
                for node_id, times_deque in self.nodes_historic.items():
                    if node_id in self.updates_receive_from_nodes:
                        if times_deque:
                            update_times.append(times_deque[-1])                      
                max_update_time = np.max(update_times)
                new_waiting_time = self.waiting_time - (self.waiting_time - max_update_time)*0.1    # Reduced 10% the difference between last update receive and waiting_time
                new_waiting_time = np.max([(max_update_time*1.20),new_waiting_time])                # select max from worst_time*1.20 or 10% reduced
                self.waiting_time = self._change_timer_value(new_waiting_time)           
            else:
                percentile = (self.n_updates_receive / self.max_updates_number) * 100      
                if percentile <= 25:
                    self.waiting_time = self.max_timer_value
                elif percentile <= 75:
                    self.waiting_time = self._change_timer_value(self.waiting_time*1.4)           # timer + 40% from max
                else:                   
                    max_ema = 0
                    for node_id, times_deque in self.nodes_historic.items():
                        if len(times_deque) == 0:  # If the deque is empty, skip this node
                            continue
                        ema = self._exponential_moving_average(times_deque, alpha=0.1)
                        max_ema = max(max_ema, ema)                          
                    if percentile < self.acceptable_percent:
                        if not self.round_completed >= self.max_historic_size:
                            self.waiting_time = self._change_timer_value(self.waiting_time*1.2)    # timer + 20% from max
                        else:
                            self.waiting_time = self._change_timer_value(max_ema*1.25)             # if enough data for historic EMA, EMA*1.25                               
                    else:
                        if not self.round_completed >= self.max_historic_size:
                            self.waiting_time = self._change_timer_value(self.waiting_time*1.05)   # timer + 5% from max
                        else:
                            self.waiting_time = self._change_timer_value(max_ema*1.15)             # if enough data for historic EMA, EMA*1.15         
        # reset round variables                    
        self.n_updates_receive = 0
        self.last_update_receive_time = 0
        self.updates_receive_from_nodes.clear()

    def _exponential_moving_average(self, data, alpha=0.1):
        if not data:  # Handle the case where the data list is empty
            return 0
        ema = [data[0]]  
        data_left = list(data)[1:]
        for value in data_left:
            ema.append((1 - alpha) * ema[-1] + alpha * value)
        return ema[-1]
    
    def _change_timer_value(self, new_value):
        return new_value if new_value < self.max_timer_value else  self.max_timer_value
    
    def _complete_data(self):
        """
            fill empty times using worst acceptable time
        """
        for node_id, times_deque in self.nodes_historic.items():
            if not node_id in self.updates_receive_from_nodes:
                times_deque.append(self.max_timer_value)