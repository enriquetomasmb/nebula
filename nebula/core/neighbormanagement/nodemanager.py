import asyncio
import logging
import os
import asyncio
import threading

from nebula.core.utils.locker import Locker
from nebula.core.neighbormanagement.candidateselection.candidateselector import factory_CandidateSelector
from nebula.core.neighbormanagement.modelhandlers.modelhandler import factory_ModelHandler
from nebula.core.neighbormanagement.neighborpolicies.neighborpolicy import factory_NeighborPolicy
from nebula.core.neighbormanagement.timergenerator import TimerGenerator
from nebula.core.pb import nebula_pb2
from nebula.core.network.communications import CommunicationsManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nebula.core.engine import Engine

class NodeManager():
    
    def __init__(
        self,  
        topology,
        model_handler,
        engine : "Engine"
    ):
        logging.info("🌐  Initializing Node Manager")
        self._engine = engine
        self.config = engine.get_config()
        self._neighbor_policy = factory_NeighborPolicy(topology)  
        self._candidate_selector = factory_CandidateSelector(topology) 
        self._model_handler = factory_ModelHandler(model_handler)
        self.late_connection_process_lock = Locker(name="late_connection_process_lock")
        self.weight_modifier = {}
        self.weight_modifier_lock = Locker(name="weight_modifier_lock")
        self.new_node_weight_value = 3
        self.accept_candidates_lock = Locker(name="accept_candidates_lock")
        self.recieve_offer_timer = 5
        self._restructure_process_lock = Locker(name="restructure_process_lock")
        self.restructure = False
        self.max_time_to_wait = 6
        self._timer_generator = TimerGenerator(self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False), self.max_time_to_wait, 80)
        
        self.set_confings()

    @property
    def engine(self):
        return self._engine
    
    @property
    def neighbor_policy(self):
        return self._neighbor_policy

    @property
    def candidate_selector(self):
        return self._candidate_selector

    @property
    def model_handler(self):
        return self._model_handler
    
    @property
    def timer_generator(self):
        return self._timer_generator
    
    def get_restructure_process_lock(self):
        return self._restructure_process_lock
    
    def set_confings(self):
        """
            neighbor_policy config:
                - direct connections a.k.a neighbors
                - non-direct connections
                - self addr
                
            model_handler config:  
                - self total rounds
                - self current round
                - self epochs
                
            candidate_selector config:
                - self model loss
                - self weight distance
                - self weight hetereogeneity
        """
        self.neighbor_policy.set_config([self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False), self.engine.cm.get_addrs_current_connections(only_direct=False, myself=False), self.engine.addr])
        #self.model_handler.set_config([self.engine.get_round(), self.engine.config.participant["training_args"]["epochs"]])
        self.candidate_selector.set_config([self.engine.get_loss(), self.config.participant["molibity_args"]["weight_distance"], self.config.participant["molibity_args"]["weight_het"]])
    
    def get_timer(self):
        return self.timer_generator.get_timer(self.engine.get_round())
    
    def adjust_timer(self):
        self.timer_generator.adjust_timer()
        
    def get_stop_condition(self):
        return self.timer_generator.get_stop_condition()    
              
    def add_weight_modifier(self, addr):
        self.weight_modifier_lock().acquire()
        if not addr in self.weight_modifier:
            self.weight_modifier[addr] = self.new_node_weight_value
        self.weight_modifier_lock().release()
    
    def remove_weight_modifier(self, addr):
        self.weight_modifier_lock().acquire()
        if addr in self.weight_modifier:
            del self.weight_modifier[addr]
        self.weight_modifier_lock().release()
        
    def _update_weight_modifier(self, addr):
        self.weight_modifier_lock().acquire()
        if addr in self.weight_modifier:
            new_weight = self.weight_modifier[addr] - 1/self.engine.get_round()**2
            if new_weight > 1:
                self.weight_modifier[addr] = new_weight
            else:
                self.remove_weight_modifier(addr)
        self.weight_modifier_lock().release()
    
    def get_weight_modifier(self, addr):
        self.weight_modifier_lock().acquire()
        if addr in self.weight_modifier:
            wm = self.weight_modifier[addr]
            self._update_weight_modifier(addr, self.engine.get_round()) 
        else:
            wm = 1
        self.weight_modifier_lock().release()
        return wm
    
    def accept_connection(self,source):
        if self.accept_candidates_lock().locked():
            return False
        return self.neighbor_policy.accept_connection(source)
    
    def need_more_neighbors(self):
        return self.neighbor_policy.need_more_neighbors()
    
    def get_actions(self):
        return self.neighbor_policy.get_actions()
    
    def update_neighbors(self, node, remove=False):
        self.neighbor_policy.update_neighbors(node, remove)
        self.timer_generator.update_node(node, remove)
        if remove:
            self.remove_weight_modifier(node)
        if not remove:
            self.neighbor_policy.meet_node(node)
    
    def no_neighbors_left(self):
        return len(self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False))
    
    def meet_node(self, node):
        self.neighbor_policy.meet_node(node)
        
    def get_nodes_known(self, neighbors_too=False):
        return self.neighbor_policy.get_nodes_known(neighbors_too)
    
    def accept_model(self, source, decoded_model, rounds, round, epochs, n_neighbors, loss): 
        if not self.accept_candidates_lock().locked():
            self.model_handler.accept_model(decoded_model)
            self.model_handler.set_config(config=(rounds, round, epochs))    
            self.candidate_selector.add_candidate((source, n_neighbors, loss))

    def add_candidate(self,source, n_neighbors, loss):
        if not self.accept_candidates_lock().locked():
            self.candidate_selector.add_candidate((source, n_neighbors, loss))

    async def start_late_connection_process(self):
        """
            This function represents the process of discovering the federation and stablish the first
            connections with it. The first step is to send the DISCOVER_JOIN message to look for nodes,
            the ones that receive that message will send back a OFFER_MODEL message. It contains info to do
            a selection process among candidates to later on connect do the best ones. 
            The process will repeat until at least one candidate is found and the process will be locked
            to avoid concurrency.

        Returns:
            data neccesary to create trainer
        """
        logging.info("🌐  Initializing start late connection process from Node Manager")
        
        self.late_connection_process_lock.acquire()
        best_candidates = []
        self.candidate_selector.remove_candidates()
        
        # send discover
        msg = self.engine.cm.mm.generate_discover_message(nebula_pb2.DiscoverMessage.Action.DISCOVER_JOIN)
        await self.engine.cm.establish_connection_with_federation(msg)
           
        # wait offer
        await asyncio.sleep(self.recieve_offer_timer)
        
        # acquire lock to not accept late candidates
        self.accept_candidates_lock.acquire()
        
        if self.candidate_selector.any_candidate():
            
            # create message to send to new neightbors
            msg = self.engine.cm.mm.generate_connection_message(nebula_pb2.ConnectionMessage.Action.LATE_CONNECT)
            
            best_candidates =  self.candidate_selector.select_candidates()
            
            for addr, _, _ in best_candidates:
                await self.engine.cm.connect(addr, direct=True)
                await self.engine.cm.send_message(addr, msg)
                                    
            model, rounds, round, epochs = self.model_handler.get_model()
            self.accept_candidates_lock().release()
            self.late_connection_process_lock.release()
            return (model, rounds, round, epochs)         
                                                                                       
        # if no candidates, repeat process
        else:
            self.accept_candidates_lock.release()
            self.late_connection_process_lock.release()
            return self.start_late_connection_process()
    
    
    
    """
        Retopology in progress
     """
    
    async def find_new_connections(self):
        logging.info("🌐  Initializing restructure process from Node Manager")
        self._restructure_process_lock.acquire()
        # Update the config params of candidate_selector
        self.candidate_selector.set_config([self.engine.get_loss(), self.engine.weight_distance, self.engine.weight_het])
        self.thread = threading.Thread(target=self._find_connections_thread, args=(self))
        self.thread.start()
        self.restructure = True
        while self.restructure:
            await asyncio.sleep(1)
        self._restructure_process_lock.release()
        
    
    async def _find_connections_thread(self):
        posible_connections = self.get_nodes_known(neighbors_too=False)
        while self.restructure:
            # out of federation but got info about nodes inside
            if len(posible_connections) > 0:
                msg = self.engine.cm.mm.generate_discover_message(nebula_pb2.DiscoverMessage.Action.DISCOVER_NODE)
                for addr in posible_connections:
                    # send message to known nodes, wait for response and select
                    pass
            # im out of federation without info about any nodes inside of it
            else:
                await self.start_late_connection_process()  
            
            self.restructure = self.need_more_neighbors()      