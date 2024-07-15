nebula.core.training.siamese
============================

.. py:module:: nebula.core.training.siamese


Classes
-------

.. autoapisummary::

   nebula.core.training.siamese.Siamese


Module Contents
---------------

.. py:class:: Siamese(model, data, config=None, logger=None)

   .. py:property:: logger


   .. py:method:: get_round()


   .. py:method:: set_model(model)


   .. py:method:: set_data(data)


   .. py:method:: create_trainer()


   .. py:method:: get_global_model_parameters()


   .. py:method:: set_parameter_second_aggregation(params)


   .. py:method:: get_model_parameters(bytes=False)


   .. py:method:: get_hash_model()

      :returns: SHA256 hash of model parameters
      :rtype: str



   .. py:method:: set_epochs(epochs)


   .. py:method:: serialize_model(model)


   .. py:method:: deserialize_model(data)


   .. py:method:: set_model_parameters(params, initialize=False)


   .. py:method:: train()


   .. py:method:: test()


   .. py:method:: get_model_weight()


   .. py:method:: finalize_round()


