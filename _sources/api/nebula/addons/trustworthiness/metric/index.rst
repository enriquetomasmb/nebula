:py:mod:`nebula.addons.trustworthiness.metric`
==============================================

.. py:module:: nebula.addons.trustworthiness.metric


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.addons.trustworthiness.metric.TrustMetricManager




Attributes
~~~~~~~~~~

.. autoapisummary::

   nebula.addons.trustworthiness.metric.dirname
   nebula.addons.trustworthiness.metric.logger


.. py:data:: dirname

   

.. py:data:: logger

   

.. py:class:: TrustMetricManager


   Manager class to help store the output directory and handle calls from the FL framework.

   .. py:method:: evaluate(scenario, weights, use_weights=False)

      Evaluates the trustworthiness score.

      :param scenario: The scenario in whith the trustworthiness will be calculated.
      :type scenario: object
      :param weights: The desired weghts of the pillars.
      :type weights: dict
      :param use_weights: True to turn on the weights in the metric config file, default to False.
      :type use_weights: bool



