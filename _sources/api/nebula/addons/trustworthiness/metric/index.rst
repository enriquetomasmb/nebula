nebula.addons.trustworthiness.metric
====================================

.. py:module:: nebula.addons.trustworthiness.metric


Attributes
----------

.. autoapisummary::

   nebula.addons.trustworthiness.metric.dirname
   nebula.addons.trustworthiness.metric.logger


Classes
-------

.. autoapisummary::

   nebula.addons.trustworthiness.metric.TrustMetricManager


Module Contents
---------------

.. py:data:: dirname

.. py:data:: logger

.. py:class:: TrustMetricManager

   Manager class to help store the output directory and handle calls from the FL framework.


   .. py:attribute:: factsheet_file_nm
      :value: 'factsheet.json'



   .. py:attribute:: eval_metrics_file_nm
      :value: 'eval_metrics.json'



   .. py:attribute:: nebula_trust_results_nm
      :value: 'nebula_trust_results.json'



   .. py:method:: evaluate(scenario, weights, use_weights=False)

      Evaluates the trustworthiness score.

      :param scenario: The scenario in whith the trustworthiness will be calculated.
      :type scenario: object
      :param weights: The desired weghts of the pillars.
      :type weights: dict
      :param use_weights: True to turn on the weights in the metric config file, default to False.
      :type use_weights: bool



