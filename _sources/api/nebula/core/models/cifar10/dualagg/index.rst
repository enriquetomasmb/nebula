:py:mod:`nebula.core.models.cifar10.dualagg`
============================================

.. py:module:: nebula.core.models.cifar10.dualagg


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.models.cifar10.dualagg.ContrastiveLoss
   nebula.core.models.cifar10.dualagg.DualAggModel




.. py:class:: ContrastiveLoss(mu=0.5)


   Bases: :py:obj:`torch.nn.Module`

   Contrastive loss function.

   .. py:method:: forward(local_out, global_out, historical_out, labels)

      Calculates the contrastive loss between the local output, global output, and historical output.

      :param local_out: The local output tensor of shape (batch_size, embedding_size).
      :type local_out: torch.Tensor
      :param global_out: The global output tensor of shape (batch_size, embedding_size).
      :type global_out: torch.Tensor
      :param historical_out: The historical output tensor of shape (batch_size, embedding_size).
      :type historical_out: torch.Tensor
      :param labels: The ground truth labels tensor of shape (batch_size,).
      :type labels: torch.Tensor

      :returns: The contrastive loss value.
      :rtype: torch.Tensor

      :raises ValueError: If the input tensors have different shapes.

      .. rubric:: Notes

      - The contrastive loss is calculated as the difference between the mean cosine similarity of the local output
          with the historical output and the mean cosine similarity of the local output with the global output,
          multiplied by a scaling factor mu.
      - The cosine similarity values represent the similarity between the corresponding vectors in the input tensors.
      Higher values indicate greater similarity, while lower values indicate less similarity.



.. py:class:: DualAggModel(input_channels=3, num_classes=10, learning_rate=0.001, mu=0.5, metrics=None, confusion_matrix=None, seed=None)


   Bases: :py:obj:`lightning.LightningModule`

   .. py:method:: process_metrics(phase, y_pred, y, loss=None, mode='local')

      Calculate and log metrics for the given phase.
      :param phase: One of 'Train', 'Validation', or 'Test'
      :type phase: str
      :param y_pred: Model predictions
      :type y_pred: torch.Tensor
      :param y: Ground truth labels
      :type y: torch.Tensor
      :param loss: Loss value
      :type loss: torch.Tensor, optional


   .. py:method:: log_metrics_by_epoch(phase, print_cm=False, plot_cm=False, mode='local')

      Log all metrics at the end of an epoch for the given phase.
      :param phase: One of 'Train', 'Validation', or 'Test'
      :type phase: str
      :param : param phase:
      :param : param plot_cm:


   .. py:method:: forward(x, mode='local')

      Forward pass of the model.


   .. py:method:: configure_optimizers()


   .. py:method:: step(batch, phase)


   .. py:method:: training_step(batch, batch_id)

      Training step for the model.
      :param batch:
      :param batch_id:

      Returns:


   .. py:method:: on_train_epoch_end()


   .. py:method:: validation_step(batch, batch_idx)

      Validation step for the model.
      :param batch:
      :param batch_idx:

      Returns:


   .. py:method:: on_validation_epoch_end()


   .. py:method:: test_step(batch, batch_idx)

      Test step for the model.
      :param batch:
      :param batch_idx:

      Returns:


   .. py:method:: on_test_epoch_end()


   .. py:method:: save_historical_model()

      Save the current local model as the historical model.


   .. py:method:: global_load_state_dict(state_dict)

      Load the given state dictionary into the global model.
      :param state_dict: The state dictionary to load into the global model.
      :type state_dict: dict


   .. py:method:: historical_load_state_dict(state_dict)

      Load the given state dictionary into the historical model.
      :param state_dict: The state dictionary to load into the historical model.
      :type state_dict: dict


   .. py:method:: adapt_state_dict_for_model(state_dict, model_prefix)

      Adapt the keys in the provided state_dict to match the structure expected by the model.


   .. py:method:: get_global_model_parameters()

      Get the parameters of the global model.


   .. py:method:: print_summary()

      Print a summary of local, historical and global models to check if they are the same.



