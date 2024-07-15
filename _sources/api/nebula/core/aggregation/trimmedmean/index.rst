:py:mod:`nebula.core.aggregation.trimmedmean`
=============================================

.. py:module:: nebula.core.aggregation.trimmedmean


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.aggregation.trimmedmean.TrimmedMean




.. py:class:: TrimmedMean(config=None, beta=0, **kwargs)


   Bases: :py:obj:`nebula.core.aggregation.aggregator.Aggregator`

   Aggregator: TrimmedMean
   Authors: Dong Yin et al et al.
   Year: 2021
   Note: https://arxiv.org/pdf/1803.01498.pdf

   .. py:method:: get_trimmedmean(weights)


   .. py:method:: run_aggregation(models)



