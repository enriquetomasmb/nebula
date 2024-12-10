 █████╗ ██╗   ██╗████████╗██╗  ██╗ ██████╗ ██████╗ 
██╔══██╗██║   ██║╚══██╔══╝██║  ██║██╔═══██╗██╔══██╗
███████║██║   ██║   ██║   ███████║██║   ██║██████╔╝
██╔══██║██║   ██║   ██║   ██╔══██║██║   ██║██╔═██╗ 
██║  ██║╚██████╔╝   ██║   ██║  ██║╚██████╔╝██║  ██╗
╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝

Alejandro Avilés Serrano.


This module implements the functionality for managing federation nodes in terms of mobility. 
To achieve this, a NodeManager class will manage the processes of establishing a connection 
with the federation when a node wants to join it, receiving models from the federation, 
selecting the best candidates to connect with, and deciding, once inside the federation, 
when it is necessary or convenient to establish connections with more nodes, 
as well as applying strategies to increase the relevance of information coming from recently joined nodes in the federation.

To accomplish this, it relies on three main blocks that together develop the previously described activity, namely:
-> Candidate Selection Module
-> Received Models Handling Module
-> Neighbor Policy Module

The first two are self-explanatory. Surely, the third one may raise more questions. 
Neighbor policies refer to, for example, restrictions on neighbor aggregation due to the specific topology of the use case. 
Similarly, it will have information about the current neighbors of the node and the rest of the known nodes, 
which will allow it to make decisions about when it might be an appropriate time to initiate a process of establishing new connections.


The process of discovering potential candidates is based on an exchange of discover and offer messages, 
with certain characteristics differing depending on whether it is a federation joining or a network restructuring.
After this exchange of messages, the candidate selector will choose the most promising ones and a connection will be made to them. 
It is important to note that the receiving node can reject the connection.

1) Establish Communication
 __________                                                     ________________
| New node | -------> --------> *DISCOVER* --------> --------> | Federation node |
|__________|                                                   | _______________ |

 __________                                                     _________________
| New node | -------> --------> *OFFER* --------> -------->    | Federation node |
|__________|                                                   | _______________ |

2) Select Candidates and connect to them

 __________            ____________________                               ___________                   
| New node | -------> | Candidate Selector | ----> *LATE_CONNECT* ---->  | Candidate |
|__________|          | __________________ |                   	         | _________ |

 ___________                                   __________                   
| Candidate | ------->  *LATE_CONNECT* ---->  | New Node |
|___________|                            	  | _________|

Retopology works the same way but with diferent arguments on the messages.

The supported topologies are:
-> RING
-> FULLY CONNECTED
-> RANDOM
-> STAR (not yet)

If you want to make new implementations, use the interfaces provide.


                ########################
                ### WORK IN PROGRESS ###
                ########################

Currently working on retopology process.