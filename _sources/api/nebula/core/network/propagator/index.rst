nebula.core.network.propagator
==============================

.. py:module:: nebula.core.network.propagator


Classes
-------

.. autoapisummary::

   nebula.core.network.propagator.PropagationStrategy
   nebula.core.network.propagator.InitialModelPropagation
   nebula.core.network.propagator.StableModelPropagation
   nebula.core.network.propagator.Propagator


Module Contents
---------------

.. py:class:: PropagationStrategy

   Bases: :py:obj:`abc.ABC`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:method:: is_node_eligible(node)
      :abstractmethod:



   .. py:method:: prepare_model_payload(node)
      :abstractmethod:



.. py:class:: InitialModelPropagation(aggregator, trainer, engine)

   Bases: :py:obj:`PropagationStrategy`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: aggregator


   .. py:attribute:: trainer


   .. py:attribute:: engine


   .. py:method:: get_round()


   .. py:method:: is_node_eligible(node)


   .. py:method:: prepare_model_payload(node)


.. py:class:: StableModelPropagation(aggregator, trainer, engine)

   Bases: :py:obj:`PropagationStrategy`


   Helper class that provides a standard way to create an ABC using
   inheritance.


   .. py:attribute:: aggregator


   .. py:attribute:: trainer


   .. py:attribute:: engine


   .. py:attribute:: addr


   .. py:method:: get_round()


   .. py:method:: is_node_eligible(node)


   .. py:method:: prepare_model_payload(node)


.. py:class:: Propagator(cm)

   .. py:attribute:: engine
      :type:  nebula.core.engine.Engine


   .. py:attribute:: config
      :type:  nebula.config.config.Config


   .. py:attribute:: addr


   .. py:attribute:: cm
      :type:  nebula.core.network.communications.CommunicationsManager


   .. py:attribute:: aggregator
      :type:  nebula.core.aggregation.aggregator.Aggregator


   .. py:attribute:: trainer
      :type:  nebula.core.training.lightning.Lightning


   .. py:attribute:: status_history


   .. py:attribute:: interval


   .. py:attribute:: model_interval


   .. py:attribute:: early_stop


   .. py:attribute:: stable_rounds_count
      :value: 0



   .. py:attribute:: strategies


   .. py:method:: start()


   .. py:method:: get_round()


   .. py:method:: update_and_check_neighbors(strategy, eligible_neighbors)


   .. py:method:: reset_status_history()


   .. py:method:: propagate(strategy_id)
      :async:



