nebula.core.aggregation.fedavgSVM
=================================

.. py:module:: nebula.core.aggregation.fedavgSVM


Classes
-------

.. autoapisummary::

   nebula.core.aggregation.fedavgSVM.FedAvgSVM


Module Contents
---------------

.. py:class:: FedAvgSVM(config=None, **kwargs)

   Bases: :py:obj:`nebula.core.aggregation.aggregator.Aggregator`


   Aggregator: Federated Averaging (FedAvg)
   Authors: McMahan et al.
   Year: 2016
   Note: This is a modified version of FedAvg for SVMs.


   .. py:method:: run_aggregation(models)


