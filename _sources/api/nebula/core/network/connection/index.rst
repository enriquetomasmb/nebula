:py:mod:`nebula.core.network.connection`
========================================

.. py:module:: nebula.core.network.connection


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.network.connection.Connection




.. py:class:: Connection(cm, reader, writer, id, host, port, direct=True, active=True, compression='zlib', config=None)


   .. py:attribute:: DEFAULT_FEDERATED_ROUND

      

   .. py:method:: get_addr()


   .. py:method:: get_federated_round()


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


   .. py:method:: compress(data, compression)
      :async:


   .. py:method:: decompress(compressed)
      :async:


   .. py:method:: send(data, pb=True, encoding_type='utf-8', compression='none')
      :async:


   .. py:method:: retrieve_message(message)
      :async:


   .. py:method:: handle_incoming_message()
      :async:



