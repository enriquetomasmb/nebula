nebula.core.aggregation.blockchainReputation
============================================

.. py:module:: nebula.core.aggregation.blockchainReputation


Classes
-------

.. autoapisummary::

   nebula.core.aggregation.blockchainReputation.BlockchainReputation
   nebula.core.aggregation.blockchainReputation.BlockchainHandler


Functions
---------

.. autoapisummary::

   nebula.core.aggregation.blockchainReputation.cossim_euclidean
   nebula.core.aggregation.blockchainReputation.print_table
   nebula.core.aggregation.blockchainReputation.print_with_frame


Module Contents
---------------

.. py:function:: cossim_euclidean(model1, model2, similarity)

.. py:class:: BlockchainReputation(similarity_metric = 'CossimEuclid', config=None, **kwargs)

   Bases: :py:obj:`nebula.core.aggregation.aggregator.Aggregator`


   # BAT-SandrinHunkeler (BlockchainReputation)
   Weighted FedAvg by using relative reputation of each model's trainer
   Returns: aggregated model


   .. py:attribute:: ALGORITHM_MAP


   .. py:method:: run_aggregation(model_buffer)


.. py:function:: print_table(title, values, headers)

   Prints a title, all values ordered in a table, with the headers as column titles.
   :param title: Title of the table
   :param values: Rows of table
   :param headers: Column headers of table

   Returns: None, prints output



.. py:function:: print_with_frame(message)

   Prints a large frame with a title inside
   :param message: Title to put into the frame

   Returns: None



.. py:class:: BlockchainHandler(home_address)

   Handles interaction with Oracle and Non-Validator Node of Blockchain Network


   .. py:property:: oracle_url
      :type: str

      :classmethod:



   .. py:property:: rest_header
      :type: Mapping[str, str]

      :classmethod:



   .. py:method:: verify_balance()

      Calls blockchain directly for requesting current balance
      Returns: None




   .. py:method:: report_gas_oracle()

      Reports accumulated gas costs of all transactions made to the blockchain
      Returns: List of all accumulated gas costs per registered node




   .. py:method:: report_reputation_oracle(records)

      Reports reputations used for aggregation
      Returns: None




   .. py:method:: push_opinions(opinion_dict)

      Pushes all locally computed opinions of models to aggregate to the reputation system
      :param opinion_dict: Dict of all names:opinions for writing to the reputation system

      Returns: Json of transaction receipt




   .. py:method:: get_reputations(ip_addresses)

      Requests globally aggregated opinions values from reputation system for computing aggregation weights
      :param ip_addresses: Names of nodes of which the reputation values should be generated

      Returns: Dictionary of name:reputation from the reputation system




   .. py:method:: verify_registration()

      Verifies the successful registration of the node itself,
      executes registration again if reputation system returns false
      Returns: None




   .. py:method:: report_time_oracle(start)

      Reports time used for aggregation
      Returns: None




