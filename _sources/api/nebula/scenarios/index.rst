nebula.scenarios
================

.. py:module:: nebula.scenarios


Classes
-------

.. autoapisummary::

   nebula.scenarios.Scenario
   nebula.scenarios.ScenarioManagement


Module Contents
---------------

.. py:class:: Scenario(scenario_title, scenario_description, simulation, federation, topology, nodes, nodes_graph, n_nodes, matrix, dataset, iid, partition_selection, partition_parameter, model, agg_algorithm, rounds, logginglevel, accelerator, network_subnet, network_gateway, epochs, attacks, poisoned_node_percent, poisoned_sample_percent, poisoned_noise_percent, with_reputation, is_dynamic_topology, is_dynamic_aggregation, target_aggregation, random_geo, latitude, longitude, mobility, mobility_type, radius_federation, scheme_mobility, round_frequency, mobile_participants_percent, additional_participants, schema_additional_participants)

   .. py:attribute:: scenario_title


   .. py:attribute:: scenario_description


   .. py:attribute:: simulation


   .. py:attribute:: federation


   .. py:attribute:: topology


   .. py:attribute:: nodes


   .. py:attribute:: nodes_graph


   .. py:attribute:: n_nodes


   .. py:attribute:: matrix


   .. py:attribute:: dataset


   .. py:attribute:: iid


   .. py:attribute:: partition_selection


   .. py:attribute:: partition_parameter


   .. py:attribute:: model


   .. py:attribute:: agg_algorithm


   .. py:attribute:: rounds


   .. py:attribute:: logginglevel


   .. py:attribute:: accelerator


   .. py:attribute:: network_subnet


   .. py:attribute:: network_gateway


   .. py:attribute:: epochs


   .. py:attribute:: attacks


   .. py:attribute:: poisoned_node_percent


   .. py:attribute:: poisoned_sample_percent


   .. py:attribute:: poisoned_noise_percent


   .. py:attribute:: with_reputation


   .. py:attribute:: is_dynamic_topology


   .. py:attribute:: is_dynamic_aggregation


   .. py:attribute:: target_aggregation


   .. py:attribute:: random_geo


   .. py:attribute:: latitude


   .. py:attribute:: longitude


   .. py:attribute:: mobility


   .. py:attribute:: mobility_type


   .. py:attribute:: radius_federation


   .. py:attribute:: scheme_mobility


   .. py:attribute:: round_frequency


   .. py:attribute:: mobile_participants_percent


   .. py:attribute:: additional_participants


   .. py:attribute:: schema_additional_participants


   .. py:method:: attack_node_assign(nodes, federation, attack, poisoned_node_percent, poisoned_sample_percent, poisoned_noise_percent)

      Identify which nodes will be attacked



   .. py:method:: mobility_assign(nodes, mobile_participants_percent)

      Assign mobility to nodes



   .. py:method:: from_dict(data)
      :classmethod:



.. py:class:: ScenarioManagement(scenario, controller)

   .. py:attribute:: scenario


   .. py:attribute:: start_date_scenario


   .. py:attribute:: scenario_name


   .. py:attribute:: root_path


   .. py:attribute:: config_dir


   .. py:attribute:: log_dir


   .. py:attribute:: cert_dir


   .. py:attribute:: advanced_analytics


   .. py:attribute:: config


   .. py:attribute:: controller


   .. py:attribute:: topologymanager
      :value: None



   .. py:attribute:: env_path
      :value: None



   .. py:attribute:: use_blockchain


   .. py:attribute:: scenario_file


   .. py:attribute:: settings


   .. py:attribute:: settings_file


   .. py:attribute:: nodes


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


