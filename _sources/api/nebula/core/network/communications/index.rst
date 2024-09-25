nebula.core.network.communications
==================================

.. py:module:: nebula.core.network.communications


Classes
-------

.. autoapisummary::

   nebula.core.network.communications.CommunicationsManager


Module Contents
---------------

.. py:class:: CommunicationsManager(engine)

   .. py:attribute:: addr


   .. py:attribute:: host


   .. py:attribute:: port


   .. py:attribute:: config


   .. py:attribute:: id


   .. py:attribute:: register_endpoint


   .. py:attribute:: wait_endpoint


   .. py:attribute:: connections_lock


   .. py:attribute:: connections_manager_lock


   .. py:attribute:: connection_attempt_lock_incoming


   .. py:attribute:: connection_attempt_lock_outgoing


   .. py:attribute:: pending_connections


   .. py:attribute:: incoming_connections


   .. py:attribute:: outgoing_connections


   .. py:attribute:: ready_connections


   .. py:attribute:: received_messages_hashes


   .. py:attribute:: receive_messages_lock


   .. py:attribute:: connections_reconnect
      :value: []



   .. py:attribute:: max_connections
      :value: 1000



   .. py:attribute:: network_engine
      :value: None



   .. py:attribute:: stop_network_engine


   .. py:property:: engine


   .. py:property:: connections


   .. py:property:: mm


   .. py:property:: discoverer


   .. py:property:: health


   .. py:property:: forwarder


   .. py:property:: propagator


   .. py:property:: mobility


   .. py:method:: check_federation_ready()
      :async:



   .. py:method:: add_ready_connection(addr)
      :async:



   .. py:method:: handle_incoming_message(data, addr_from)
      :async:



   .. py:method:: handle_discovery_message(source, message)
      :async:



   .. py:method:: handle_control_message(source, message)
      :async:



   .. py:method:: handle_federation_message(source, message)
      :async:



   .. py:method:: handle_model_message(source, message)
      :async:



   .. py:method:: handle_connection_message(source, message)
      :async:



   .. py:method:: get_connections_lock()


   .. py:method:: get_config()


   .. py:method:: get_addr()


   .. py:method:: get_round()


   .. py:method:: start()
      :async:



   .. py:method:: deploy_network_engine()
      :async:



   .. py:method:: handle_connection_wrapper(reader, writer)
      :async:



   .. py:method:: handle_connection(reader, writer)
      :async:



   .. py:method:: stop()
      :async:



   .. py:method:: run_reconnections()
      :async:



   .. py:method:: verify_connections(neighbors)


   .. py:method:: network_wait()
      :async:



   .. py:method:: deploy_additional_services()
      :async:



   .. py:method:: include_received_message_hash(hash_message)
      :async:



   .. py:method:: send_message_to_neighbors(message, neighbors=None, interval=0)
      :async:



   .. py:method:: send_message(dest_addr, message)
      :async:



   .. py:method:: send_messages(messages, interval=0)
      :async:



   .. py:method:: send_model(dest_addr, round, serialized_model, weight=1)
      :async:



   .. py:method:: send_models(models, round)
      :async:



   .. py:method:: establish_connection(addr, direct=True, reconnect=False)
      :async:



   .. py:method:: connect(addr, direct=True)
      :async:



   .. py:method:: register()
      :async:



   .. py:method:: wait_for_controller()
      :async:



   .. py:method:: disconnect(dest_addr, mutual_disconnection=True)
      :async:



   .. py:method:: get_all_addrs_current_connections(only_direct=False, only_undirected=False)
      :async:



   .. py:method:: get_addrs_current_connections(only_direct=False, only_undirected=False, myself=False)
      :async:



   .. py:method:: get_connection_by_addr(addr)
      :async:



   .. py:method:: get_direct_connections()
      :async:



   .. py:method:: get_undirect_connections()
      :async:



   .. py:method:: get_nearest_connections(top = 1)
      :async:



   .. py:method:: get_ready_connections()


   .. py:method:: check_finished_experiment()


