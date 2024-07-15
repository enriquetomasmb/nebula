nebula.addons.trustworthiness.factsheet
=======================================

.. py:module:: nebula.addons.trustworthiness.factsheet


Attributes
----------

.. autoapisummary::

   nebula.addons.trustworthiness.factsheet.dirname
   nebula.addons.trustworthiness.factsheet.logger


Classes
-------

.. autoapisummary::

   nebula.addons.trustworthiness.factsheet.Factsheet


Module Contents
---------------

.. py:data:: dirname

.. py:data:: logger

.. py:class:: Factsheet

   .. py:method:: populate_factsheet_pre_train(data, scenario_name)

      Populates the factsheet with values before the training.

      :param data: Contains the data from the scenario.
      :type data: dict
      :param scenario_name: The name of the scenario.
      :type scenario_name: string



   .. py:method:: populate_factsheet_post_train(scenario)

      Populates the factsheet with values after the training.

      :param scenario: The scenario object.
      :type scenario: object



