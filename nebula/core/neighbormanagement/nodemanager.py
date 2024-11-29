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
from nebula.addons.functions import print_msg_box

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nebula.core.engine import Engine

class NodeManager():
    
    def __init__(
        self,  
        topology,
        model_handler,
        push_acceleration,
        engine : "Engine"
    ):
        self.topology = topology
        print_msg_box(msg=f"Starting NodeManager module...\nTopology: {self.topology}", indent=2, title="NodeManager module")
        logging.info("🌐  Initializing Node Manager")
        self._engine = engine
        self.config = engine.get_config()
        logging.info("Initializing Neighbor policy")
        self._neighbor_policy = factory_NeighborPolicy(self.topology)
        logging.info("Initializing Candidate Selector")  
        self._candidate_selector = factory_CandidateSelector(self.topology)
        logging.info("Initializing Model Handler")
        self._model_handler = factory_ModelHandler(model_handler)
        self.late_connection_process_lock = Locker(name="late_connection_process_lock")
        self.weight_modifier = {}
        self.weight_modifier_lock = Locker(name="weight_modifier_lock")
        self.new_node_weight_value = 3
        self.accept_candidates_lock = Locker(name="accept_candidates_lock")
        self.recieve_offer_timer = 5
        self._restructure_process_lock = Locker(name="restructure_process_lock")
        self.restructure = False
        self.discarded_offers_addr_lock = Locker(name="discarded_offers_addr_lock")
        self.discarded_offers_addr = []
        
        self.max_time_to_wait = 20
        logging.info("Initializing Timer generator")
        self._timer_generator = None #TimerGenerator(self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False), self.max_time_to_wait, 80)
        
        self._push_acceleration = push_acceleration
        self.rounds_pushed_lock = Locker(name="rounds_pushed_lock")
        self.rounds_pushed = 0
        
        self.synchronizing_rounds = False
        
        #self.set_confings()

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
    
    def get_push_acceleration(self):
        return self._push_acceleration
    
    def get_restructure_process_lock(self):
        return self._restructure_process_lock
    
    def set_synchronizing_rounds(self, status):
        self.synchronizing_rounds = status
        
    def get_syncrhonizing_rounds(self):
        return self.synchronizing_rounds    
    
    def set_rounds_pushed(self, rp):
        with self.rounds_pushed_lock:
            self.rounds_pushed = rp
    
    def still_waiting_for_candidates(self):
        return not self.accept_candidates_lock.locked()
    
    async def set_confings(self):
        """
            neighbor_policy config:
                - direct connections a.k.a neighbors
                - all nodes known
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
        logging.info(f"Building neighbor policy configuration..")
        self.neighbor_policy.set_config(
            [
                await self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False), 
                await self.engine.cm.get_addrs_current_connections(only_direct=False, only_undirected=False, myself=False), 
                self.engine.addr
            ]
        )
        logging.info(f"Building candidate selector configuration..")
        self.candidate_selector.set_config(
            [
                0, 
                0.5, 
                0.5
            ]
        )
        #self.engine.trainer.get_loss(), self.config.participant["molibity_args"]["weight_distance"], self.config.participant["molibity_args"]["weight_het"]
        #self.model_handler.set_config([self.engine.get_round(), self.engine.config.participant["training_args"]["epochs"]])
    
    def add_to_discarded_offers(self, addr_discarded):
        self.discarded_offers_addr_lock.acquire()
        self.discarded_offers_addr.append(addr_discarded)
        self.discarded_offers_addr_lock.release()
    
    def get_timer(self):
        return self.timer_generator.get_timer(self.engine.get_round())
    
    def adjust_timer(self):
        self.timer_generator.adjust_timer()
        
    def get_stop_condition(self):
        return self.timer_generator.get_stop_condition()
    
    async def receive_update_from_node(self, node_id, node_response_time):
        await self.timer_generator.receive_update(node_id, node_response_time)   
              
    def add_weight_modifier(self, addr):
        self.weight_modifier_lock.acquire()
        if not addr in self.weight_modifier:
            wv = self.new_node_weight_value 
            logging.info(f"📝 Registering | Weight modifier registered for source {addr} | round: {self.engine.get_round()} | value: {wv}")
            self.weight_modifier[addr] = wv
        self.weight_modifier_lock.release()
    
    def remove_weight_modifier(self, addr):
        self.weight_modifier_lock.acquire()
        if addr in self.weight_modifier:
            logging.info(f"📝 Removing | weight modifier registered for source {addr}")
            del self.weight_modifier[addr]
        self.weight_modifier_lock.release()
    
    def apply_weight_strategy(self, updates):
        logging.info(f"🔄 Applying weight Strategy...")
        # We must lower the weight_modifier value if a round jump has been occured
        # as many times as rounds have been jumped
        if self.rounds_pushed:
            round = self.engine.get_round()
            for i in range(0, self.rounds_pushed):
                self._update_weight_modifiers((round + i))
            self.rounds_pushed = 0
        for addr,update in updates.items():
            weight_modifier = self._get_weight_modifier(addr)
            if weight_modifier != 1:
                logging.info (f"📝 Appliying modified weight strategy | addr: {addr} | multiplier value: {weight_modifier}")
                model, weight = update
                updates.update({addr: (model, weight*weight_modifier)})
                
    def _update_weight_modifiers(self, round):
        self.weight_modifier_lock.acquire() 
        for addr,weight in self.weight_modifier.items():
            new_weight = weight - 1/(round**2)
            if new_weight > 1:
                self.weight_modifier[addr] = new_weight
            else:
                self.remove_weight_modifier(addr)
        self.weight_modifier_lock.release()
    
    def _get_weight_modifier(self, addr):
        self.weight_modifier_lock.acquire()
        if addr in self.weight_modifier:
            wm = self.weight_modifier[addr]      
        else:
            wm = 1
        self.weight_modifier_lock.release()
        return wm
    
    def accept_connection(self, source, joining=False):
        if not joining:
            if self.get_restructure_process_lock().locked():
                logging.info("NOT accepting connections | Currently upgrading network Robustness")
                return False
        else:
            return self.neighbor_policy.accept_connection(source)
    
    def need_more_neighbors(self):
        return self.neighbor_policy.need_more_neighbors()
    
    def get_actions(self):
        return self.neighbor_policy.get_actions()
    
    def update_neighbors(self, node, remove=False):
        self.neighbor_policy.update_neighbors(node, remove)
        #self.timer_generator.update_node(node, remove)
        if remove:
            self.remove_weight_modifier(node)
        if not remove:
            self.neighbor_policy.meet_node(node)
    
    def no_neighbors_left(self):
        return len(self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False))
    
    def meet_node(self, node):
        logging.info(f"Update nodes known | addr: {node}")
        self.neighbor_policy.meet_node(node)
        
    def get_nodes_known(self, neighbors_too=False):
        return self.neighbor_policy.get_nodes_known(neighbors_too)
    
    def accept_model_offer(self, source, decoded_model, rounds, round, epochs, n_neighbors, loss): 
        if not self.accept_candidates_lock.locked():
            model_accepted = self.model_handler.accept_model(decoded_model)
            self.model_handler.set_config(config=(rounds, round, epochs))
            if model_accepted:      
                self.candidate_selector.add_candidate((source, n_neighbors, loss))
                return True
        else:
            return False

    def get_trainning_info(self):
        return self.model_handler.get_model(None)

    def add_candidate(self,source, n_neighbors, loss):
        if not self.accept_candidates_lock.locked():
            self.candidate_selector.add_candidate((source, n_neighbors, loss))

    async def stop_not_selected_connections(self):
        try:
            if len(self.discarded_offers_addr) > 0:
                logging.info(f"Interrupting connections | discarded offers | nodes discarded: {self.discarded_offers_addr}")
                for addr in self.discarded_offers_addr:
                    await self.engine.cm.disconnect(addr, mutual_disconnection=True)
                    await asyncio.sleep(1) 
                self.discarded_offers_addr = []
        except asyncio.CancelledError as e:
            pass

    async def start_late_connection_process(self):
        """
            This function represents the process of discovering the federation and stablish the first
            connections with it. The first step is to send the DISCOVER_JOIN message to look for nodes,
            the ones that receive that message will send back a OFFER_MODEL message. It contains info to do
            a selection process among candidates to later on connect to the best ones. 
            The process will repeat until at least one candidate is found and the process will be locked
            to avoid concurrency.

        Returns:
            data neccesary to create trainer
        """
        logging.info("🌐  Initializing late connection process..")
        
        self.late_connection_process_lock.acquire()
        best_candidates = []
        self.candidate_selector.remove_candidates()
        
        # find federation and send discover
        await self.engine.cm.establish_connection_with_federation()
           
        # wait offer
        logging.info(f"Waiting: {self.recieve_offer_timer}s to receive offers from federation")
        await asyncio.sleep(self.recieve_offer_timer)
        
        # acquire lock to not accept late candidates
        self.accept_candidates_lock.acquire()
        
        if self.candidate_selector.any_candidate():
            logging.info("Candidates found to connect to...")    
            # create message to send to new neightbors
            msg = self.engine.cm.mm.generate_connection_message(nebula_pb2.ConnectionMessage.Action.LATE_CONNECT)       
            best_candidates = self.candidate_selector.select_candidates()
            logging.info(f"Candidates | {[addr for addr,_,_ in best_candidates]}")
            # candidates not choosen --> disconnect
            for addr, _, _ in best_candidates:
                await self.engine.cm.connect(addr, direct=True)
                await self.engine.cm.send_message(addr, msg)
                await asyncio.sleep(1) 
                                    
            self.accept_candidates_lock.release()
            self.late_connection_process_lock.release()       
                                                                                       
        # if no candidates, repeat process
        else:
            logging.info("No Candidates found | repeating process")
            self.accept_candidates_lock.release()
            self.late_connection_process_lock.release()
            await self.start_late_connection_process()
    
    
    
    """
        Retopology in progress
     """
    
    async def check_robustness(self):
        logging.info("Analizing node network robustness...")
        if len(self.engine.get_federation_nodes()) == 0:
            logging.info("No Neighbors left | reconnecting with Federation")
        elif self.neighbor_policy.need_more_neighbors():
            logging.info("Insufficient Robustness | searching for more connections")
        else:
            logging.info("Sufficient Robustness | no actions required")
            
    
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