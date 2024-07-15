nebula.addons.trustworthiness.pillar
====================================

.. py:module:: nebula.addons.trustworthiness.pillar


Attributes
----------

.. autoapisummary::

   nebula.addons.trustworthiness.pillar.logger


Classes
-------

.. autoapisummary::

   nebula.addons.trustworthiness.pillar.TrustPillar


Module Contents
---------------

.. py:data:: logger

.. py:class:: TrustPillar(name, metrics, input_docs, use_weights=False)

   Class to represent a trust pillar.

   :param name: Name of the pillar.
   :type name: string
   :param metrics: Metric definitions for the pillar.
   :type metrics: dict
   :param input_docs: Input documents.
   :type input_docs: dict
   :param use_weights: True to turn on the weights in the metric config file.
   :type use_weights: bool


   .. py:method:: evaluate()

      Evaluate the trust score for the pillar.

      :returns: Score of [0, 1].
      :rtype: float



   .. py:method:: get_notion_score(name, metrics)

      Evaluate the trust score for the notion.

      :param name: Name of the notion.
      :type name: string
      :param metrics: Metrics definitions of the notion.
      :type metrics: list

      :returns: Score of [0, 1].
      :rtype: float



   .. py:method:: get_metric_score(result, name, metric)

      Evaluate the trust score for the metric.

      :param result: The result object
      :type result: object
      :param name: Name of the metric.
      :type name: string
      :param metrics: The metric definition.
      :type metrics: dict

      :returns: Score of [0, 1].
      :rtype: float



