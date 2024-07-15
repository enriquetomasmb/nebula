nebula.config.config
====================

.. py:module:: nebula.config.config


Classes
-------

.. autoapisummary::

   nebula.config.config.Config


Module Contents
---------------

.. py:class:: Config(entity, topology_config_file=None, participant_config_file=None)

   .. py:attribute:: topology


   .. py:attribute:: participant


   .. py:attribute:: participants
      :value: []



   .. py:attribute:: participants_path
      :value: []



   .. py:method:: get_topology_config()


   .. py:method:: get_participant_config()


   .. py:method:: to_json()


   .. py:method:: set_participant_config(participant_config)


   .. py:method:: set_topology_config(topology_config_file)


   .. py:method:: add_participant_config(participant_config)


   .. py:method:: set_participants_config(participants_config)


   .. py:method:: add_participants_config(participants_config)


   .. py:method:: add_neighbor_from_config(addr)


   .. py:method:: update_neighbors_from_config(current_connections, dest_addr)


   .. py:method:: remove_neighbor_from_config(addr)


   .. py:method:: reload_config_file()


