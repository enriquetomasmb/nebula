nebula.core.engine
==================

.. py:module:: nebula.core.engine


Classes
-------

.. autoapisummary::

   nebula.core.engine.Engine
   nebula.core.engine.MaliciousNode
   nebula.core.engine.AggregatorNode
   nebula.core.engine.ServerNode
   nebula.core.engine.TrainerNode
   nebula.core.engine.IdleNode


Functions
---------

.. autoapisummary::

   nebula.core.engine.handle_exception
   nebula.core.engine.signal_handler
   nebula.core.engine.print_banner


Module Contents
---------------

.. py:function:: handle_exception(exc_type, exc_value, exc_traceback)

.. py:function:: signal_handler(sig, frame)

.. py:function:: print_banner()

.. py:class:: Engine(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')

   .. py:attribute:: config


   .. py:attribute:: idx


   .. py:attribute:: experiment_name


   .. py:attribute:: ip


   .. py:attribute:: port


   .. py:attribute:: addr


   .. py:attribute:: role


   .. py:attribute:: name


   .. py:attribute:: docker_id


   .. py:attribute:: client


   .. py:attribute:: round
      :value: None



   .. py:attribute:: total_rounds
      :value: None



   .. py:attribute:: federation_nodes


   .. py:attribute:: initialized
      :value: False



   .. py:attribute:: log_dir


   .. py:attribute:: security


   .. py:attribute:: model_poisoning


   .. py:attribute:: poisoned_ratio


   .. py:attribute:: noise_type


   .. py:attribute:: msg


   .. py:attribute:: with_reputation


   .. py:attribute:: is_dynamic_topology


   .. py:attribute:: is_dynamic_aggregation


   .. py:attribute:: target_aggregation


   .. py:attribute:: learning_cycle_lock


   .. py:attribute:: federation_setup_lock


   .. py:attribute:: federation_ready_lock


   .. py:attribute:: round_lock


   .. py:property:: cm


   .. py:property:: reporter


   .. py:property:: event_manager


   .. py:property:: aggregator


   .. py:method:: get_aggregator_type()


   .. py:property:: trainer


   .. py:method:: get_addr()


   .. py:method:: get_config()


   .. py:method:: get_federation_nodes()


   .. py:method:: get_initialization_status()


   .. py:method:: set_initialization_status(status)


   .. py:method:: get_round()


   .. py:method:: get_federation_ready_lock()


   .. py:method:: get_federation_setup_lock()


   .. py:method:: get_round_lock()


   .. py:method:: create_trainer_module()
      :async:



   .. py:method:: start_communications()
      :async:



   .. py:method:: deploy_federation()
      :async:



   .. py:method:: reputation_calculation(aggregated_models_weights)


   .. py:method:: send_reputation(malicious_nodes)
      :async:



.. py:class:: MaliciousNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')

   Bases: :py:obj:`Engine`


   .. py:attribute:: attack


   .. py:attribute:: fit_time
      :value: 0.0



   .. py:attribute:: extra_time
      :value: 0.0



   .. py:attribute:: round_start_attack
      :value: 3



   .. py:attribute:: round_stop_attack
      :value: 6



   .. py:attribute:: aggregator_bening


.. py:class:: AggregatorNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')

   Bases: :py:obj:`Engine`


.. py:class:: ServerNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')

   Bases: :py:obj:`Engine`


.. py:class:: TrainerNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')

   Bases: :py:obj:`Engine`


.. py:class:: IdleNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')

   Bases: :py:obj:`Engine`


