nebula.core.network.messages
============================

.. py:module:: nebula.core.network.messages


Classes
-------

.. autoapisummary::

   nebula.core.network.messages.MessagesManager


Module Contents
---------------

.. py:class:: MessagesManager(addr, config, cm)

   .. py:attribute:: addr


   .. py:attribute:: config


   .. py:attribute:: cm


   .. py:method:: generate_discovery_message(action, latitude=0.0, longitude=0.0)


   .. py:method:: generate_control_message(action, log='Control message')


   .. py:method:: generate_federation_message(action, arguments=[], round=None)


   .. py:method:: generate_model_message(round, serialized_model, weight=1)


   .. py:method:: generate_connection_message(action)


   .. py:method:: generate_reputation_message(reputation)


