nebula.core.network.connection
==============================

.. py:module:: nebula.core.network.connection


Classes
-------

.. autoapisummary::

   nebula.core.network.connection.MessageChunk
   nebula.core.network.connection.Connection


Module Contents
---------------

.. py:class:: MessageChunk

   .. py:attribute:: index
      :type:  int


   .. py:attribute:: data
      :type:  bytes


   .. py:attribute:: is_last
      :type:  bool


.. py:class:: Connection(cm, reader, writer, id, host, port, direct=True, active=True, compression='zlib', config=None)

   .. py:attribute:: DEFAULT_FEDERATED_ROUND


   .. py:attribute:: cm


   .. py:attribute:: reader


   .. py:attribute:: writer


   .. py:attribute:: id


   .. py:attribute:: host


   .. py:attribute:: port


   .. py:attribute:: addr


   .. py:attribute:: direct


   .. py:attribute:: active


   .. py:attribute:: last_active


   .. py:attribute:: compression


   .. py:attribute:: config


   .. py:attribute:: federated_round


   .. py:attribute:: latitude
      :value: None



   .. py:attribute:: longitude
      :value: None



   .. py:attribute:: loop


   .. py:attribute:: read_task
      :value: None



   .. py:attribute:: process_task
      :value: None



   .. py:attribute:: pending_messages_queue


   .. py:attribute:: message_buffers
      :type:  Dict[bytes, Dict[int, MessageChunk]]


   .. py:attribute:: EOT_CHAR
      :value: b'\x00\x00\x00\x04'



   .. py:attribute:: COMPRESSION_CHAR
      :value: b'\x00\x00\x00\x01'



   .. py:attribute:: DATA_TYPE_PREFIXES


   .. py:attribute:: HEADER_SIZE
      :value: 21



   .. py:attribute:: MAX_CHUNK_SIZE


   .. py:attribute:: BUFFER_SIZE
      :value: 65536



   .. py:method:: get_addr()


   .. py:method:: get_federated_round()


   .. py:method:: get_tunnel_status()


   .. py:method:: update_round(federated_round)


   .. py:method:: update_geolocation(latitude, longitude)


   .. py:method:: get_geolocation()


   .. py:method:: get_neighbor_distance()


   .. py:method:: compute_distance(latitude, longitude)


   .. py:method:: compute_distance_myself()


   .. py:method:: get_ready()


   .. py:method:: get_direct()


   .. py:method:: set_direct(direct)


   .. py:method:: set_active(active)


   .. py:method:: is_active()


   .. py:method:: get_last_active()


   .. py:method:: start()
      :async:



   .. py:method:: stop()
      :async:



   .. py:method:: reconnect(max_retries = 5, delay = 5)
      :async:



   .. py:method:: send(data, pb = True, encoding_type = 'utf-8', is_compressed = False)
      :async:



   .. py:method:: handle_incoming_message()
      :async:



   .. py:method:: process_message_queue()
      :async:



