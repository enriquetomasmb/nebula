
import os
import socket
import sys
import platform
import time
import threading
import logging
from nebula.core.utils.locker import Locker
from nebula.core.network.externalconnectionservice import ExternalConnectionService

class NebulaServer(threading.Thread):
        
    BCAST_IP = '239.255.255.250'
    UPNP_PORT = 1900
    IP = '0.0.0.0'
    M_SEARCH_REQ_MATCH = "M-SEARCH"
    
    def __init__(self, nebula_service: "NebulaConnectionService", addr):
        threading.Thread.__init__(self)
        self.interrupted = False
        self.ns = nebula_service
        self.addr = addr
        
    def run(self):
        self.listen()
        
    def stop(self):
        self.interrupted = True
        logging.info("Nebula upnp server stop")
        
    def listen(self):
        """
        Listen on broadcast addr with standard 1900 port
        It will reponse a standard ssdp message with blockchain ip and port info if receive a M_SEARCH message
        """
        try:
            macro = socket.SO_REUSEPORT
            os_name = platform.system()
            if os_name == "Windows":
                macro = socket.SO_REUSEADDR
                
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, macro, 1)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(self.BCAST_IP) + socket.inet_aton(self.IP))
            sock.bind((self.IP, self.UPNP_PORT))
            sock.settimeout(1)
            logging.info("Nebula upnp server is listening...")
            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                except socket.error:
                    if self.interrupted:
                        sock.close()
                        return
                else:
                    if self._is_nebula_message(data): 
                        self.respond(addr)
                    time.sleep(1)
                    self.stop()
        except Exception as e:
            logging.info('Error in Nebula npnp server listening: %s', e)

    def _is_nebula_message(self, msg):
        msg_str = msg.decode('utf-8')
        return "ST: urn:nebula-service" in msg_str
        
    def respond(self, addr):
        try:
            #local_ip = # FIND THE IP
            UPNP_RESPOND = """HTTP/1.1 200 OK
            CACHE-CONTROL: max-age=1800
            ST: urn:nebula-service
            EXT:
            LOCATION: {}
            """.format(
                self.addr
            ).replace("\n", "\r\n")
            outSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            outSock.sendto(UPNP_RESPOND.encode('ASCII'), addr)
            outSock.close()
        except Exception as e:
            logging.info('Error in Nebula upnp response message to client %s', e)

class NebulaClient(threading.Thread):
    # 30 seconds for search_interval
    SEARCH_INTERVAL = 5
    BCAST_IP = '239.255.255.250'
    BCAST_PORT = 1900
           
    def __init__(self, nebula_service: "NebulaConnectionService"):
        threading.Thread.__init__(self)
        self.interrupted = False
        self.ns = nebula_service
        
    def run(self):
        self.keep_search()
        
    def stop(self):
        self.interrupted = True
        logging.info(" Nebula upnp client stop")

    def keep_search(self):
        """
        run search function every SEARCH_INTERVAL
        """
        try:
            while True:
                self.search()
                for x in range(self.SEARCH_INTERVAL):
                    time.sleep(1)
                    if self.interrupted:
                        return
        except Exception as e:
            logging.info('Error in Nebula upnp client keep search %s', e)

    def search(self):
        """
        broadcast SSDP DISCOVER message to LAN network
        filter our protocal and add to network
        """
        try:
            SSDP_DISCOVER = ('M-SEARCH * HTTP/1.1\r\n' +
                            'HOST: 239.255.255.250:1900\r\n' +
                            'MAN: "ssdp:discover"\r\n' +
                            'MX: 1\r\n' +
                            'ST: urn:nebula-service\r\n' +
                            '\r\n')
                
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(SSDP_DISCOVER.encode('ASCII'), (self.BCAST_IP, self.BCAST_PORT))
            sock.settimeout(3)
            while True:
                data, addr = sock.recvfrom(1024)
                if self._is_nebula_message(data):
                    self.ns.response_recieved(data, addr)
        except:
            sock.close()
            
    def _is_nebula_message(self, msg):
        msg_str = msg.decode('utf-8')
        return "ST: urn:nebula-service" in msg_str

class NebulaConnectionService(ExternalConnectionService):
    
    def __init__(self, addr):
        self.addrs_found_lock = Locker(name="addrs_found_lock")
        self.nodes_found = []
        self.repeatsearch_interval = 3
        self.addr = addr
        self.server = None
        self.client = None
        
    def start(self):
        self.server = NebulaServer(self, self.addr)
        self.server.start()
        
    def stop(self):
        self.server.stop
    
    def find_federation(self): 
        """
            Initialization client thread to send broadcast discover to federation
        """
        logging.info(f"Node {self.addr} trying to find federation..")
        self.nodes_found = []
        self.client = NebulaClient(self)
        self.client.start()
        time.sleep(self.repeatsearch_interval)
        while not len(self.get_nodes()):
            time.sleep(self.repeatsearch_interval)
        self.client.stop()
              
    def response_recieved(self, data, addr):
        print("NebulaMulticastingService: Response recieved")
        msg_str = data.decode('utf-8')
        self._add_addr(msg_str)
        
    def _add_addr(self, msg_str):
        self.mutex.acquire()
        lineas = msg_str.splitlines()
        # Buscar la línea que contiene "LOCATION: "
        for linea in lineas:
            if linea.strip().startswith("LOCATION:"):
                addr = linea.split(": ")[1].strip()
                break
        self.nodes_found.append(addr)
        self.mutex.release()
        
    def get_nodes(self):
        self.mutex.acquire()
        cp = self.nodes_found.copy()
        self.mutex.release()
        return cp
            