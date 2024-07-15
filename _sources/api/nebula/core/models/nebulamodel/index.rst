:py:mod:`nebula.core.models.nebulamodel`
========================================

.. py:module:: nebula.core.models.nebulamodel


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.models.nebulamodel.NebulaModel




.. py:class:: NebulaModel(input_channels=1, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)


   Bases: :py:obj:`lightning.LightningModule`, :py:obj:`abc.ABC`

   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.

   .. py:method:: process_metrics(phase, y_pred, y, loss=None)

      Calculate and log metrics for the given phase.
      The metrics are calculated in each batch.
      :param phase: One of 'Train', 'Validation', or 'Test'
      :type phase: str
      :param y_pred: Model predictions
      :type y_pred: torch.Tensor
      :param y: Ground truth labels
      :type y: torch.Tensor
      :param loss: Loss value
      :type loss: torch.Tensor, optional


   .. py:method:: log_metrics_end(phase)

      Log metrics for the given phase.
      :param phase: One of 'Train', 'Validation', 'Test (Local)', or 'Test (Global)'
      :type phase: str
      :param print_cm: Print confusion matrix
      :type print_cm: bool
      :param plot_cm: Plot confusion matrix
      :type plot_cm: bool


   .. py:method:: generate_confusion_matrix(phase, print_cm=False, plot_cm=False)

      Generate and plot the confusion matrix for the given phase.
      :param phase: One of 'Train', 'Validation', 'Test (Local)', or 'Test (Global)'
      :type phase: str
      :param : param phase:
      :param : param print:
      :param : param plot:


   .. py:method:: forward(x)
      :abstractmethod:

      Forward pass of the model.


   .. py:method:: configure_optimizers()
      :abstractmethod:

      Optimizer configuration.


   .. py:method:: step(batch, batch_idx, phase)

      Training/validation/test step.


   .. py:method:: training_step(batch, batch_idx)

      Training step for the model.
      :param batch:
      :param batch_id:

      Returns:


   .. py:method:: on_train_end()


   .. py:method:: on_train_epoch_end()


   .. py:method:: validation_step(batch, batch_idx)

      Validation step for the model.
      :param batch:
      :param batch_idx:

      Returns:


   .. py:method:: on_validation_end()


   .. py:method:: on_validation_epoch_end()


   .. py:method:: test_step(batch, batch_idx, dataloader_idx=None)

      Test step for the model.
      :param batch:
      :param batch_idx:

      Returns:


   .. py:method:: on_test_end()


   .. py:method:: on_test_epoch_end()



