nebula.core.network.forwarder
=============================

.. py:module:: nebula.core.network.forwarder


Classes
-------

.. autoapisummary::

   nebula.core.network.forwarder.Forwarder


Module Contents
---------------

.. py:class:: Forwarder(config, cm)

   .. py:attribute:: config


   .. py:attribute:: cm


   .. py:attribute:: pending_messages


   .. py:attribute:: pending_messages_lock


   .. py:attribute:: interval


   .. py:attribute:: number_forwarded_messages


   .. py:attribute:: messages_interval


   .. py:method:: start()
      :async:



   .. py:method:: run_forwarder()
      :async:



   .. py:method:: process_pending_messages(messages_left)
      :async:



   .. py:method:: forward(msg, addr_from)
      :async:



