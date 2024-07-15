:py:mod:`nebula.core.aggregation.aggregator`
============================================

.. py:module:: nebula.core.aggregation.aggregator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.aggregation.aggregator.Aggregator



Functions
~~~~~~~~~

.. autoapisummary::

   nebula.core.aggregation.aggregator.create_aggregator
   nebula.core.aggregation.aggregator.create_target_aggregator
   nebula.core.aggregation.aggregator.create_malicious_aggregator



.. py:exception:: AggregatorException


   Bases: :py:obj:`Exception`

   Common base class for all non-exit exceptions.


.. py:function:: create_aggregator(config, engine)


.. py:function:: create_target_aggregator(config, engine)


.. py:class:: Aggregator(config=None, engine=None)


   Bases: :py:obj:`abc.ABC`

   Helper class that provides a standard way to create an ABC using
   inheritance.

   .. py:property:: cm


   .. py:method:: run_aggregation(models)
      :abstractmethod:


   .. py:method:: update_federation_nodes(federation_nodes)


   .. py:method:: set_waiting_global_update()


   .. py:method:: reset()


   .. py:method:: get_nodes_pending_models_to_aggregate()


   .. py:method:: include_model_in_buffer(model, weight, source=None, round=None, local=False)
      :async:


   .. py:method:: get_aggregation()


   .. py:method:: print_model_size(model)



.. py:function:: create_malicious_aggregator(aggregator, attack)


