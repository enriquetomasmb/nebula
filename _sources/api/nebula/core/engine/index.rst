:py:mod:`nebula.core.engine`
============================

.. py:module:: nebula.core.engine


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.engine.Engine
   nebula.core.engine.MaliciousNode
   nebula.core.engine.AggregatorNode
   nebula.core.engine.ServerNode
   nebula.core.engine.TrainerNode
   nebula.core.engine.IdleNode



Functions
~~~~~~~~~

.. autoapisummary::

   nebula.core.engine.handle_exception
   nebula.core.engine.signal_handler
   nebula.core.engine.print_banner



.. py:function:: handle_exception(exc_type, exc_value, exc_traceback)


.. py:function:: signal_handler(sig, frame)


.. py:function:: print_banner()


.. py:class:: Engine(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')


   .. py:property:: cm


   .. py:property:: reporter


   .. py:property:: event_manager


   .. py:property:: aggregator


   .. py:property:: trainer


   .. py:method:: get_aggregator_type()


   .. py:method:: get_addr()


   .. py:method:: get_config()


   .. py:method:: get_federation_nodes()


   .. py:method:: get_initialization_status()


   .. py:method:: set_initialization_status(status)


   .. py:method:: get_round()


   .. py:method:: get_federation_ready_lock()


   .. py:method:: get_round_lock()


   .. py:method:: create_trainer_service()


   .. py:method:: get_trainer_service()


   .. py:method:: start_communications()
      :async:


   .. py:method:: deploy_federation()
      :async:


   .. py:method:: reputation_calculation(aggregated_models_weights)


   .. py:method:: send_reputation(malicious_nodes)
      :async:



.. py:class:: MaliciousNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')


   Bases: :py:obj:`Engine`


.. py:class:: AggregatorNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')


   Bases: :py:obj:`Engine`


.. py:class:: ServerNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')


   Bases: :py:obj:`Engine`


.. py:class:: TrainerNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')


   Bases: :py:obj:`Engine`


.. py:class:: IdleNode(model, dataset, config=Config, trainer=Lightning, security=False, model_poisoning=False, poisoned_ratio=0, noise_type='gaussian')


   Bases: :py:obj:`Engine`


