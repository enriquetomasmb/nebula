nebula.addons.blockchain.oracle.app
===================================

.. py:module:: nebula.addons.blockchain.oracle.app


Attributes
----------

.. autoapisummary::

   nebula.addons.blockchain.oracle.app.app
   nebula.addons.blockchain.oracle.app.oracle


Classes
-------

.. autoapisummary::

   nebula.addons.blockchain.oracle.app.Oracle


Functions
---------

.. autoapisummary::

   nebula.addons.blockchain.oracle.app.error_handler
   nebula.addons.blockchain.oracle.app.home
   nebula.addons.blockchain.oracle.app.rest_transfer_funds
   nebula.addons.blockchain.oracle.app.rest_report_gas
   nebula.addons.blockchain.oracle.app.rest_get_balance
   nebula.addons.blockchain.oracle.app.rest_status
   nebula.addons.blockchain.oracle.app.rest_contract
   nebula.addons.blockchain.oracle.app.rest_get_gas_report
   nebula.addons.blockchain.oracle.app.rest_get_gas_series
   nebula.addons.blockchain.oracle.app.rest_report_time
   nebula.addons.blockchain.oracle.app.rest_get_time_report
   nebula.addons.blockchain.oracle.app.rest_report_reputation
   nebula.addons.blockchain.oracle.app.rest_get_reputation_timeseries


Module Contents
---------------

.. py:data:: app

.. py:function:: error_handler(func)

   Adds default status and header to all REST responses used for Oracle


.. py:class:: Oracle

   .. py:attribute:: acc


   .. py:attribute:: contract_obj


   .. py:property:: contract_abi


   .. py:property:: contract_address


   .. py:method:: wait_for_blockchain()

      Executes REST post request for a selected RPC method to check if blockchain
      is up and running
      Returns: None




   .. py:method:: transfer_funds(address)

      Creates transaction to blockchain network for assigning funds to Cores
      :param address: public wallet address of Core to assign funds to

      Returns: Transaction receipt




   .. py:method:: deploy_chaincode()

      Creates transaction to deploy chain code on the blockchain network by
      sending transaction to non-validator node
      Returns: address of chain code on the network




   .. py:method:: get_balance(addr)

      Creates transaction to blockchain network to request balance for parameter address
      :param addr: public wallet address of account

      Returns: current balance in ether (ETH)




   .. py:method:: report_gas(amount, aggregation_round)

      Experiment method for collecting and reporting gas usage statistics
      :param aggregation_round: Aggregation round of sender
      :param amount: Amount of gas spent in WEI

      Returns: None




   .. py:method:: get_gas_report()

      Experiment method for requesting the summed up records of reported gas usage
      Returns: JSON with name:value (WEI/USD) for every reported node




   .. py:property:: gas_store
      Experiment method for requesting the detailed records of the gas reports
      Returns: list of records of type: list[(node, timestamp, gas)]


   .. py:method:: report_time(time_s, aggregation_round)

      Experiment method for collecting and reporting time statistics
      :param aggregation_round: Aggregation round of node
      :param method: Name of node which reports time
      :param time_s: Amount of time spend on method

      Returns: None




   .. py:method:: report_reputation(records, aggregation_round, sender)

      Experiment method for collecting and reporting reputations statistics
      :param aggregation_round: Current aggregation round of sender
      :param records: list of (name:reputation) records
      :param sender: node reporting its local view

      Returns: None




   .. py:property:: time_store
      :type: list

      Experiment method for requesting all records of nodes which reported timings
      Returns: JSON with method:(sum_time, n_calls) for every reported node


   .. py:property:: reputation_store
      :type: list

      Experiment method for requesting all records of reputations
      Returns: list with (name, reputation, timestamp)


   .. py:property:: ready
      :type: bool

      Returns true if the Oracle is ready itself and the chain code was deployed successfully
      Returns: True if ready False otherwise


.. py:function:: home()

.. py:function:: rest_transfer_funds()

.. py:function:: rest_report_gas()

.. py:function:: rest_get_balance()

.. py:function:: rest_status()

.. py:function:: rest_contract()

.. py:function:: rest_get_gas_report()

.. py:function:: rest_get_gas_series()

.. py:function:: rest_report_time()

.. py:function:: rest_get_time_report()

.. py:function:: rest_report_reputation()

.. py:function:: rest_get_reputation_timeseries()

.. py:data:: oracle

