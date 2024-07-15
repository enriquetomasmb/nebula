nebula.core.training.lightning
==============================

.. py:module:: nebula.core.training.lightning


Classes
-------

.. autoapisummary::

   nebula.core.training.lightning.Lightning


Module Contents
---------------

.. py:class:: Lightning(model, data, config=None, logger=None)

   .. py:attribute:: DEFAULT_MODEL_WEIGHT
      :value: 1



   .. py:attribute:: BYPASS_MODEL_WEIGHT
      :value: 0



   .. py:property:: logger


   .. py:method:: get_round()


   .. py:method:: set_model(model)


   .. py:method:: set_data(data)


   .. py:method:: create_trainer()


   .. py:method:: validate_neighbour_model(neighbour_model_param)


   .. py:method:: get_hash_model()

      :returns: SHA256 hash of model parameters
      :rtype: str



   .. py:method:: set_epochs(epochs)


   .. py:method:: serialize_model(model)


   .. py:method:: deserialize_model(data)


   .. py:method:: set_model_parameters(params, initialize=False)


   .. py:method:: get_model_parameters(bytes=False)


   .. py:method:: train()


   .. py:method:: test()


   .. py:method:: get_model_weight()


   .. py:method:: on_round_start()


   .. py:method:: on_round_end()


   .. py:method:: on_learning_cycle_end()


