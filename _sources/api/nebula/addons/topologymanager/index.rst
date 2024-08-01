nebula.addons.topologymanager
=============================

.. py:module:: nebula.addons.topologymanager


Classes
-------

.. autoapisummary::

   nebula.addons.topologymanager.TopologyManager


Module Contents
---------------

.. py:class:: TopologyManager(scenario_name=None, n_nodes=5, b_symmetric=True, undirected_neighbor_num=5, topology=None)

   .. py:attribute:: scenario_name


   .. py:attribute:: n_nodes


   .. py:attribute:: b_symmetric


   .. py:attribute:: undirected_neighbor_num


   .. py:attribute:: topology


   .. py:attribute:: nodes


   .. py:attribute:: b_fully_connected
      :value: False



   .. py:method:: draw_graph(plot=False, path=None)


   .. py:method:: generate_topology()


   .. py:method:: generate_server_topology()


   .. py:method:: generate_ring_topology(increase_convergence=False)


   .. py:method:: generate_custom_topology(topology)


   .. py:method:: get_matrix_adjacency_from_neighbors(neighbors)


   .. py:method:: get_topology()


   .. py:method:: get_nodes()


   .. py:method:: get_coordinates(random_geo=True)
      :staticmethod:



   .. py:method:: add_nodes(nodes)


   .. py:method:: update_nodes(config_participants)


   .. py:method:: get_node(node_idx)


   .. py:method:: get_neighbors_string(node_idx)


