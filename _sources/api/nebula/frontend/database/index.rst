nebula.frontend.database
========================

.. py:module:: nebula.frontend.database


Attributes
----------

.. autoapisummary::

   nebula.frontend.database.user_db_file_location
   nebula.frontend.database.node_db_file_location
   nebula.frontend.database.scenario_db_file_location
   nebula.frontend.database.notes_db_file_location
   nebula.frontend.database.PRAGMA_SETTINGS


Functions
---------

.. autoapisummary::

   nebula.frontend.database.setup_database
   nebula.frontend.database.initialize_databases
   nebula.frontend.database.list_users
   nebula.frontend.database.get_user_info
   nebula.frontend.database.verify
   nebula.frontend.database.delete_user_from_db
   nebula.frontend.database.add_user
   nebula.frontend.database.update_user
   nebula.frontend.database.list_nodes
   nebula.frontend.database.list_nodes_by_scenario_name
   nebula.frontend.database.update_node_record
   nebula.frontend.database.remove_all_nodes
   nebula.frontend.database.remove_nodes_by_scenario_name
   nebula.frontend.database.get_run_hashes_scenario
   nebula.frontend.database.get_all_scenarios
   nebula.frontend.database.get_all_scenarios_and_check_completed
   nebula.frontend.database.scenario_update_record
   nebula.frontend.database.scenario_set_all_status_to_finished
   nebula.frontend.database.scenario_set_status_to_finished
   nebula.frontend.database.scenario_set_status_to_completed
   nebula.frontend.database.get_running_scenario
   nebula.frontend.database.get_completed_scenario
   nebula.frontend.database.get_scenario_by_name
   nebula.frontend.database.remove_scenario_by_name
   nebula.frontend.database.check_scenario_federation_completed
   nebula.frontend.database.check_scenario_with_role
   nebula.frontend.database.save_notes
   nebula.frontend.database.get_notes
   nebula.frontend.database.remove_note


Module Contents
---------------

.. py:data:: user_db_file_location
   :value: 'databases/users.db'


.. py:data:: node_db_file_location
   :value: 'databases/nodes.db'


.. py:data:: scenario_db_file_location
   :value: 'databases/scenarios.db'


.. py:data:: notes_db_file_location
   :value: 'databases/notes.db'


.. py:data:: PRAGMA_SETTINGS
   :value: ['PRAGMA journal_mode=WAL;', 'PRAGMA synchronous=NORMAL;', 'PRAGMA journal_size_limit=1048576;',...


.. py:function:: setup_database(db_file_location)
   :async:


.. py:function:: initialize_databases()
   :async:


.. py:function:: list_users(all_info=False)

.. py:function:: get_user_info(user)

.. py:function:: verify(user, password)

.. py:function:: delete_user_from_db(user)

.. py:function:: add_user(user, password, role)

.. py:function:: update_user(user, password, role)

.. py:function:: list_nodes(scenario_name=None, sort_by='idx')

.. py:function:: list_nodes_by_scenario_name(scenario_name)

.. py:function:: update_node_record(node_uid, idx, ip, port, role, neighbors, latitude, longitude, timestamp, federation, federation_round, scenario, run_hash)
   :async:


.. py:function:: remove_all_nodes()

.. py:function:: remove_nodes_by_scenario_name(scenario_name)

.. py:function:: get_run_hashes_scenario(scenario_name)

.. py:function:: get_all_scenarios(sort_by='start_time')

.. py:function:: get_all_scenarios_and_check_completed(sort_by='start_time')

.. py:function:: scenario_update_record(scenario_name, start_time, end_time, title, description, status, network_subnet, model, dataset, rounds, role)

.. py:function:: scenario_set_all_status_to_finished()

.. py:function:: scenario_set_status_to_finished(scenario_name)

.. py:function:: scenario_set_status_to_completed(scenario_name)

.. py:function:: get_running_scenario()

.. py:function:: get_completed_scenario()

.. py:function:: get_scenario_by_name(scenario_name)

.. py:function:: remove_scenario_by_name(scenario_name)

.. py:function:: check_scenario_federation_completed(scenario_name)

.. py:function:: check_scenario_with_role(role, scenario_name)

.. py:function:: save_notes(scenario, notes)

.. py:function:: get_notes(scenario)

.. py:function:: remove_note(scenario)

