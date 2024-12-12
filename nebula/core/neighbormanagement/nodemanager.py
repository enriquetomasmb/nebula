import asyncio
import logging
import os
import asyncio
import threading

from nebula.core.utils.locker import Locker
from nebula.core.neighbormanagement.candidateselection.candidateselector import factory_CandidateSelector
from nebula.core.neighbormanagement.modelhandlers.modelhandler import factory_ModelHandler
from nebula.core.neighbormanagement.neighborpolicies.neighborpolicy import factory_NeighborPolicy
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
        self.pending_confirmation_from_nodes = []
        self.pending_confirmation_from_nodes_lock = Locker(name="pending_confirmation_from_nodes_lock")
        self.weight_modifier = {}
        self.weight_modifier_lock = Locker(name="weight_modifier_lock")
        self.new_node_weight_multiplier = 3
        self.accept_candidates_lock = Locker(name="accept_candidates_lock")
        self.recieve_offer_timer = 5
        self._restructure_process_lock = Locker(name="restructure_process_lock")
        self.restructure = False
        self.discarded_offers_addr_lock = Locker(name="discarded_offers_addr_lock")
        self.discarded_offers_addr = [] 
        self._push_acceleration = push_acceleration
        self.rounds_pushed_lock = Locker(name="rounds_pushed_lock")
        self.rounds_pushed = 0
        
        self.synchronizing_rounds = False
        
        self._fast_reboot = True
        self._learning_rate=2e-3
        
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
    
    def get_learning_rate_increase(self):
        return self._learning_rate
    
    def fast_reboot_on(self):
        return self._fast_reboot
    
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
  

                                                        ##############################
                                                        #       WEIGHT STRATEGY      #
                                                        ##############################

    def add_weight_modifier(self, addr):
        self.weight_modifier_lock.acquire()
        if not addr in self.weight_modifier:
            wm = self.new_node_weight_multiplier 
            logging.info(f"📝 Registering | Weight modifier registered for source {addr} | round: {self.engine.get_round()} | value: {wm}")
            self.weight_modifier[addr] = (wm,1)
        self.weight_modifier_lock.release()
    
    def remove_weight_modifier(self, addr):
        self.weight_modifier_lock.acquire()
        if addr in self.weight_modifier:
            logging.info(f"📝 Removing | weight modifier registered for source {addr}")
            del self.weight_modifier[addr]
        self.weight_modifier_lock.release()
    
    async def apply_weight_strategy(self, updates: dict):
        logging.info(f"🔄  Applying weight Strategy...")
        # We must lower the weight_modifier value if a round jump has been occured
        # as many times as rounds have been jumped
        if self.rounds_pushed:
            logging.info(f"🔄  There are rounds being pushed...")
            for i in range(0, self.rounds_pushed):
                logging.info(f"🔄  Update | weights being updated cause of push...")
                self._update_weight_modifiers()
            self.rounds_pushed = 0  
        for addr,update in updates.items():
            weightmodifier, rounds = self._get_weight_modifier(addr)
            if weightmodifier != 1:
                logging.info (f"📝 Appliying modified weight strategy | addr: {addr} | multiplier value: {weightmodifier}")
                model, weight = update
                updates.update({addr: (model, weight*weightmodifier)})
        await self._update_weight_modifiers()
      
    async def _update_weight_modifiers(self):
        self.weight_modifier_lock.acquire()
        logging.info(f"🔄  Update | weights being updated")
        if self.weight_modifier:
            for addr, (weight,rounds) in self.weight_modifier.items():
                new_weight = weight - 1/(rounds**2)
                rounds = rounds + 1
                if new_weight > 1 and rounds <= 20:
                    self.weight_modifier[addr] = (new_weight, rounds)            
                else:
                    self.remove_weight_modifier(addr)
        else:
            self._learning_rate = 1e-3
            await self.engine.update_model_learning_rate()
        self.weight_modifier_lock.release()
    
    def _get_weight_modifier(self, addr):
        self.weight_modifier_lock.acquire()
        wm = self.weight_modifier.get(addr, (1,0))     
        self.weight_modifier_lock.release()
        return wm


                                                        ##############################
                                                        #        CONNECTIONS         #
                                                        ##############################

    
    def accept_connection(self, source, joining=False):
        if not joining:
            if self.get_restructure_process_lock().locked():
                logging.info("NOT accepting connections | Currently upgrading network Robustness")
                return False
            else:
                return self.neighbor_policy.accept_connection(source)
        else:
            return self.neighbor_policy.accept_connection(source)
        
    #TODO añadir un remove    
    def add_pending_connection_confirmation(self, addr):
        logging.info(f" Addition | pending connection confirmation from: {addr}")
        with self.pending_confirmation_from_nodes_lock:
            self.pending_confirmation_from_nodes.append(addr)
     
    def clear_pending_confirmations(self):
        with self.pending_confirmation_from_nodes_lock:
            self.pending_confirmation_from_nodes.clear()
     
    def waiting_confirmation_from(self, addr):
        with self.pending_confirmation_from_nodes_lock:
            return addr in self.pending_confirmation_from_nodes  
            
    async def confirmation_received(self, addr, confirmation=False):
        logging.info(f" Update | connection confirmation received from: {addr} | confirmation: {confirmation}")
        if confirmation:
            await self.engine.cm.connect(addr, direct=True)    
            self.update_neighbors(addr)
        else:
            with self.pending_confirmation_from_nodes_lock:
                self.pending_confirmation_from_nodes.remove(addr)  
        
    def add_to_discarded_offers(self, addr_discarded):
        self.discarded_offers_addr_lock.acquire()
        self.discarded_offers_addr.append(addr_discarded)
        self.discarded_offers_addr_lock.release()
    
    def need_more_neighbors(self):
        return self.neighbor_policy.need_more_neighbors()
    
    def get_actions(self):
        return self.neighbor_policy.get_actions()
    
    def update_neighbors(self, node, remove=False):
        logging.info(f"Update neighbor | node addr: {node} | remove: {remove}")
        self.neighbor_policy.update_neighbors(node, remove)
        #self.timer_generator.update_node(node, remove)
        if remove:
            self.remove_weight_modifier(node)
        if not remove:
            self.neighbor_policy.meet_node(node)
    
    async def neighbors_left(self):
        return len(await self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False)) > 0
    
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
            await asyncio.sleep(20)
            with self.discarded_offers_addr_lock:
                if len(self.discarded_offers_addr) > 0:
                    self.discarded_offers_addr = set(self.discarded_offers_addr) - await self.engine.cm.get_addrs_current_connections(only_direct=True, myself=False)
                    logging.info(f"Interrupting connections | discarded offers | nodes discarded: {self.discarded_offers_addr}")
                    for addr in self.discarded_offers_addr:
                        await self.engine.cm.disconnect(addr, mutual_disconnection=True)
                        await asyncio.sleep(1) 
                    self.discarded_offers_addr = []
        except asyncio.CancelledError as e:
            pass

    async def check_external_connection_service_status(self):   
        logging.info(f"🔄 Checking external connection service status...")
        n = await self.neighbors_left()
        ecs = await self.engine.cm.is_external_connection_service_running()
        ss = self.engine.get_sinchronized_status()
        action = None
        logging.info(f"Stats | neighbors: {n} | service running: {ecs} | synchronized status: {ss}")
        if not await self.neighbors_left() and await self.engine.cm.is_external_connection_service_running():
            logging.info(f"❗️  Isolated node | Shutdowning service required")
            action = lambda: self.engine.cm.stop_external_connection_service()
        elif await self.neighbors_left() and not await self.engine.cm.is_external_connection_service_running() and self.engine.get_sinchronized_status():
            logging.info(f"🔄 NOT isolated node | Service not running | Starting service...")
            action = lambda: self.engine.cm.init_external_connection_service()
        return action

    #TODO NOT infinite loop, define n_tries
    async def start_late_connection_process(self, connected=False, msg_type="discover_join", addrs_known=None):
        """
            This function represents the process of discovering the federation and stablish the first
            connections with it. The first step is to send the DISCOVER_JOIN/NODES message to look for nodes,
            the ones that receive that message will send back a OFFER_MODEL/METRIC message. It contains info to do
            a selection process among candidates to later on connect to the best ones. 
            The process will repeat until at least one candidate is found and the process will be locked
            to avoid concurrency.
        """
        logging.info("🌐  Initializing late connection process..")
        
        self.late_connection_process_lock.acquire()
        best_candidates = []
        self.candidate_selector.remove_candidates()
        self.clear_pending_confirmations()
        
        # find federation and send discover
        await self.engine.cm.stablish_connection_to_federation(msg_type, addrs_known)
           
        # wait offer
        logging.info(f"Waiting: {self.recieve_offer_timer}s to receive offers from federation")
        await asyncio.sleep(self.recieve_offer_timer)
        
        # acquire lock to not accept late candidates
        self.accept_candidates_lock.acquire()
        
        if self.candidate_selector.any_candidate():
            logging.info("Candidates found to connect to...")    
            # create message to send to candidates selected
            if not connected:
                msg = self.engine.cm.mm.generate_connection_message(nebula_pb2.ConnectionMessage.Action.LATE_CONNECT)
            else:
                msg = self.engine.cm.mm.generate_connection_message(nebula_pb2.ConnectionMessage.Action.RESTRUCTURE)
                        
            best_candidates = self.candidate_selector.select_candidates()
            logging.info(f"Candidates | {[addr for addr,_,_ in best_candidates]}")
            # candidates not choosen --> disconnect
            try:
                for addr, _, _ in best_candidates:
                    await self.engine.cm.send_message(addr, msg)
                    self.add_pending_connection_confirmation(addr)
                    #await self.engine.cm.connect(addr, direct=True)    
                    #self.update_neighbors(addr)
                    await asyncio.sleep(1) 
            except asyncio.CancelledError as e:
                self.update_neighbors(addr, remove=True)
                pass                        
            self.accept_candidates_lock.release()
            self.late_connection_process_lock.release()       
            self.candidate_selector.remove_candidates()                                                                           
        # if no candidates, repeat process
        else:
            logging.info("❗️  No Candidates found...")
            self.accept_candidates_lock.release()
            self.late_connection_process_lock.release()
            if not connected:
                logging.info("❗️  repeating process...")
                await self.start_late_connection_process(connected, msg_type, addrs_known)
    
    
    
                                                        ##############################
                                                        #         ROBUSTNESS         #
                                                        ##############################
    
    
    async def check_robustness(self):
        logging.info("🔄 Analizing node network robustness...")
        if not self._restructure_process_lock.locked():
            if not self.neighbors_left():
                logging.info("No Neighbors left | reconnecting with Federation")
                #await self.reconnect_to_federation()
            elif self.neighbor_policy.need_more_neighbors() and self.engine.get_sinchronized_status():
                logging.info("Insufficient Robustness | Upgrading robustness | Searching for more connections")
                asyncio.create_task(self.upgrade_connection_robustness())
            else:
                if not self.engine.get_sinchronized_status():
                    logging.info("Device not synchronized with federation")
                else:
                    logging.info("Sufficient Robustness | no actions required")
        else:
            logging.info("❗️ Reestructure/Reconnecting process already running...")
            
    async def reconnect_to_federation(self):
        # If we got some refs, try to reconnect to them
        self._restructure_process_lock.acquire()
        if self.neighbor_policy.get_nodes_known() > 0:
            logging.info("Reconnecting | Addrs availables")
            await self.start_late_connection_process(connected=False, msg_type="discover_nodes", addrs_known=self.neighbor_policy.get_nodes_known())
        # Otherwise stablish connection to federation sending discover nodes instead of join 
        else:
            logging.info("Reconnecting | NO Addrs availables")
            await self.start_late_connection_process(connected=False, msg_type="discover_nodes")
        self._restructure_process_lock.release()
        
    async def upgrade_connection_robustness(self):
        self._restructure_process_lock.acquire()
        addrs_to_connect = self.neighbor_policy.get_nodes_known(neighbors_too=False)
        # If we got some refs, try to connect to them
        if len(addrs_to_connect) > 0:
            logging.info(f"Reestructuring | Addrs availables | addr list: {addrs_to_connect}")
            await self.start_late_connection_process(connected=True, msg_type="discover_nodes", addrs_known=addrs_to_connect)
        else:
            logging.info("Reestructuring | NO Addrs availables")
            await self.start_late_connection_process(connected=True, msg_type="discover_nodes")
        self._restructure_process_lock.release()
        
                