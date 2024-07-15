:py:mod:`nebula.frontend.app`
=============================

.. py:module:: nebula.frontend.app


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.frontend.app.Settings
   nebula.frontend.app.ConnectionManager



Functions
~~~~~~~~~

.. autoapisummary::

   nebula.frontend.app.websocket_endpoint
   nebula.frontend.app.datetimeformat
   nebula.frontend.app.get_session
   nebula.frontend.app.set_default_user
   nebula.frontend.app.startup_event
   nebula.frontend.app.signal_handler
   nebula.frontend.app.custom_http_exception_handler
   nebula.frontend.app.index
   nebula.frontend.app.nebula_home
   nebula.frontend.app.nebula_dashboard_private
   nebula.frontend.app.nebula_admin
   nebula.frontend.app.save_note_for_scenario
   nebula.frontend.app.get_notes_for_scenario
   nebula.frontend.app.nebula_login
   nebula.frontend.app.nebula_logout
   nebula.frontend.app.nebula_delete_user
   nebula.frontend.app.nebula_add_user
   nebula.frontend.app.nebula_update_user
   nebula.frontend.app.nebula_dashboard
   nebula.frontend.app.nebula_dashboard_monitor
   nebula.frontend.app.update_topology
   nebula.frontend.app.nebula_update_node
   nebula.frontend.app.nebula_register_node
   nebula.frontend.app.nebula_wait_nodes
   nebula.frontend.app.nebula_monitor_log
   nebula.frontend.app.nebula_monitor_log_x
   nebula.frontend.app.nebula_monitor_log_debug
   nebula.frontend.app.nebula_monitor_log_error
   nebula.frontend.app.nebula_monitor_image
   nebula.frontend.app.stop_scenario
   nebula.frontend.app.stop_all_scenarios
   nebula.frontend.app.nebula_stop_scenario
   nebula.frontend.app.remove_scenario
   nebula.frontend.app.nebula_remove_scenario
   nebula.frontend.app.get_tracking_hash_scenario
   nebula.frontend.app.zipdir
   nebula.frontend.app.nebula_dashboard_download_logs_metrics
   nebula.frontend.app.nebula_dashboard_deployment
   nebula.frontend.app.attack_node_assign
   nebula.frontend.app.mobility_assign
   nebula.frontend.app.node_stopped
   nebula.frontend.app.run_scenario
   nebula.frontend.app.run_scenarios
   nebula.frontend.app.nebula_dashboard_deployment_run



Attributes
~~~~~~~~~~

.. autoapisummary::

   nebula.frontend.app.settings
   nebula.frontend.app.app
   nebula.frontend.app.manager
   nebula.frontend.app.templates
   nebula.frontend.app.nodes_registration
   nebula.frontend.app.scenarios_list_length
   nebula.frontend.app.scenarios_finished
   nebula.frontend.app.stop_all_scenarios_event
   nebula.frontend.app.finish_scenario_event
   nebula.frontend.app.nodes_finished
   nebula.frontend.app.parser


.. py:class:: Settings


   .. py:attribute:: debug
      :type: bool

      

   .. py:attribute:: advanced_analytics
      :type: bool

      

   .. py:attribute:: log_dir
      :type: str

      

   .. py:attribute:: config_dir
      :type: str

      

   .. py:attribute:: cert_dir
      :type: str

      

   .. py:attribute:: root_host_path
      :type: str

      

   .. py:attribute:: config_frontend_dir
      :type: str

      

   .. py:attribute:: statistics_port
      :type: int

      

   .. py:attribute:: secret_key
      :type: str

      

   .. py:attribute:: PERMANENT_SESSION_LIFETIME
      :type: datetime.timedelta

      

   .. py:attribute:: templates_dir
      :type: str
      :value: 'templates'

      


.. py:data:: settings

   

.. py:data:: app

   

.. py:class:: ConnectionManager


   .. py:method:: connect(websocket)
      :async:


   .. py:method:: disconnect(websocket)


   .. py:method:: send_personal_message(message, websocket)
      :async:


   .. py:method:: broadcast(message)
      :async:



.. py:data:: manager

   

.. py:function:: websocket_endpoint(websocket, client_id)
   :async:


.. py:data:: templates

   

.. py:function:: datetimeformat(value, format='%B %d, %Y %H:%M')


.. py:function:: get_session(request)


.. py:function:: set_default_user()


.. py:function:: startup_event()
   :async:


.. py:data:: nodes_registration

   

.. py:data:: scenarios_list_length
   :value: 0

   

.. py:data:: scenarios_finished
   :value: 0

   

.. py:function:: signal_handler(signal, frame)


.. py:function:: custom_http_exception_handler(request, exc)
   :async:


.. py:function:: index()
   :async:


.. py:function:: nebula_home(request)
   :async:


.. py:function:: nebula_dashboard_private(request, scenario_name, session = Depends(get_session))
   :async:


.. py:function:: nebula_admin(request, session = Depends(get_session))
   :async:


.. py:function:: save_note_for_scenario(scenario_name, request, session = Depends(get_session))
   :async:


.. py:function:: get_notes_for_scenario(scenario_name)
   :async:


.. py:function:: nebula_login(request, session = Depends(get_session), user = Form(...), password = Form(...))
   :async:


.. py:function:: nebula_logout(request, session = Depends(get_session))
   :async:


.. py:function:: nebula_delete_user(user, request, session = Depends(get_session))
   :async:


.. py:function:: nebula_add_user(request, session = Depends(get_session), user = Form(...), password = Form(...), role = Form(...))
   :async:


.. py:function:: nebula_update_user(request, session = Depends(get_session), user = Form(...), password = Form(...), role = Form(...))
   :async:


.. py:function:: nebula_dashboard(request, session = Depends(get_session))
   :async:


.. py:function:: nebula_dashboard_monitor(scenario_name, request, session = Depends(get_session))
   :async:


.. py:function:: update_topology(scenario_name, nodes_list, nodes_config)


.. py:function:: nebula_update_node(scenario_name, request, session = Depends(get_session))
   :async:


.. py:function:: nebula_register_node(scenario_name, request)
   :async:


.. py:function:: nebula_wait_nodes(scenario_name)
   :async:


.. py:function:: nebula_monitor_log(scenario_name, id)
   :async:


.. py:function:: nebula_monitor_log_x(scenario_name, id, number)
   :async:


.. py:function:: nebula_monitor_log_debug(scenario_name, id)
   :async:


.. py:function:: nebula_monitor_log_error(scenario_name, id)
   :async:


.. py:function:: nebula_monitor_image(scenario_name)
   :async:


.. py:function:: stop_scenario(scenario_name)


.. py:function:: stop_all_scenarios()


.. py:function:: nebula_stop_scenario(scenario_name, stop_all, request, session = Depends(get_session))
   :async:


.. py:function:: remove_scenario(scenario_name=None)


.. py:function:: nebula_remove_scenario(scenario_name, request, session = Depends(get_session))
   :async:


.. py:function:: get_tracking_hash_scenario(scenario_name)


.. py:function:: zipdir(path, ziph)


.. py:function:: nebula_dashboard_download_logs_metrics(scenario_name, request, session = Depends(get_session))
   :async:


.. py:function:: nebula_dashboard_deployment(request, session = Depends(get_session))
   :async:


.. py:function:: attack_node_assign(nodes, federation, attack, poisoned_node_percent, poisoned_sample_percent, poisoned_noise_percent)

   Identify which nodes will be attacked


.. py:function:: mobility_assign(nodes, mobile_participants_percent)

   Assign mobility to nodes


.. py:data:: stop_all_scenarios_event

   

.. py:data:: finish_scenario_event

   

.. py:data:: nodes_finished
   :value: []

   

.. py:function:: node_stopped(scenario_name, request)
   :async:


.. py:function:: run_scenario(scenario_data, role)


.. py:function:: run_scenarios(data, role)


.. py:function:: nebula_dashboard_deployment_run(request, background_tasks, session = Depends(get_session))
   :async:


.. py:data:: parser

   

