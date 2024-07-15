:py:mod:`nebula.scenarios`
==========================

.. py:module:: nebula.scenarios


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.scenarios.Scenario
   nebula.scenarios.ScenarioManagement




.. py:class:: Scenario(scenario_title, scenario_description, simulation, federation, topology, nodes, nodes_graph, n_nodes, matrix, dataset, iid, partition_selection, partition_parameter, model, agg_algorithm, rounds, logginglevel, accelerator, network_subnet, network_gateway, epochs, attacks, poisoned_node_percent, poisoned_sample_percent, poisoned_noise_percent, with_reputation, is_dynamic_topology, is_dynamic_aggregation, target_aggregation, random_geo, latitude, longitude, mobility, mobility_type, radius_federation, scheme_mobility, round_frequency, mobile_participants_percent, additional_participants, schema_additional_participants)


   .. py:method:: attack_node_assign(nodes, federation, attack, poisoned_node_percent, poisoned_sample_percent, poisoned_noise_percent)

      Identify which nodes will be attacked


   .. py:method:: mobility_assign(nodes, mobile_participants_percent)

      Assign mobility to nodes


   .. py:method:: from_dict(data)
      :classmethod:



.. py:class:: ScenarioManagement(scenario, controller)


   .. py:method:: stop_blockchain()
      :staticmethod:


   .. py:method:: stop_participants()
      :staticmethod:


   .. py:method:: stop_nodes()
      :staticmethod:


   .. py:method:: load_configurations_and_start_nodes(additional_participants=None, schema_additional_participants=None)


   .. py:method:: create_topology(matrix=None)


   .. py:method:: start_blockchain()


   .. py:method:: start_nodes_docker()


   .. py:method:: remove_files_by_scenario(scenario_name)
      :classmethod:


   .. py:method:: scenario_finished(timeout_seconds)



