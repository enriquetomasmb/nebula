nebula.core.training.lightning
==============================

.. py:module:: nebula.core.training.lightning


Attributes
----------

.. autoapisummary::

   nebula.core.training.lightning.logging_training


Classes
-------

.. autoapisummary::

   nebula.core.training.lightning.NebulaProgressBar
   nebula.core.training.lightning.Lightning


Module Contents
---------------

.. py:data:: logging_training

.. py:class:: NebulaProgressBar

   Bases: :py:obj:`lightning.pytorch.callbacks.ProgressBar`


   Nebula progress bar for training.
   Logs the percentage of completion of the training process using logging.


   .. py:attribute:: enable
      :value: True



   .. py:method:: disable()

      Disable the progress bar logging.



   .. py:method:: on_train_epoch_start(trainer, pl_module)

      Called when the training epoch starts.



   .. py:method:: on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

      Called at the end of each training batch.



   .. py:method:: on_train_epoch_end(trainer, pl_module)

      Called at the end of the training epoch.



   .. py:method:: on_validation_epoch_start(trainer, pl_module)


   .. py:method:: on_validation_epoch_end(trainer, pl_module)


   .. py:method:: on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)


   .. py:method:: on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

      Called at the end of each test batch.



   .. py:method:: on_test_epoch_start(trainer, pl_module)


   .. py:method:: on_test_epoch_end(trainer, pl_module)


.. py:class:: Lightning(model, data, config=None, logger=None)

   .. py:attribute:: DEFAULT_MODEL_WEIGHT
      :value: 1



   .. py:attribute:: BYPASS_MODEL_WEIGHT
      :value: 0



   .. py:attribute:: model


   .. py:attribute:: data


   .. py:attribute:: config


   .. py:attribute:: epochs
      :value: 1



   .. py:attribute:: round
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
      :async:



   .. py:method:: test()
      :async:



   .. py:method:: get_model_weight()


   .. py:method:: on_round_start()


   .. py:method:: on_round_end()


   .. py:method:: on_learning_cycle_end()


