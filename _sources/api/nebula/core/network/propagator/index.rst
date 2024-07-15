:py:mod:`nebula.core.network.propagator`
========================================

.. py:module:: nebula.core.network.propagator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.network.propagator.PropagationStrategy
   nebula.core.network.propagator.InitialModelPropagation
   nebula.core.network.propagator.StableModelPropagation
   nebula.core.network.propagator.Propagator




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

   .. py:method:: get_round()


   .. py:method:: is_node_eligible(node)


   .. py:method:: prepare_model_payload(node)



.. py:class:: StableModelPropagation(aggregator, trainer, engine)


   Bases: :py:obj:`PropagationStrategy`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:method:: get_round()


   .. py:method:: is_node_eligible(node)


   .. py:method:: prepare_model_payload(node)



.. py:class:: Propagator(cm)


   .. py:method:: start()


   .. py:method:: get_round()


   .. py:method:: update_and_check_neighbors(strategy, eligible_neighbors)


   .. py:method:: reset_status_history()


   .. py:method:: propagate(strategy_id)
      :async:


   .. py:method:: propagate_continuously(strategy_id)
      :async:



