nebula.core.models.cifar10.resnet
=================================

.. py:module:: nebula.core.models.cifar10.resnet


Attributes
----------

.. autoapisummary::

   nebula.core.models.cifar10.resnet.IMAGE_SIZE
   nebula.core.models.cifar10.resnet.BATCH_SIZE
   nebula.core.models.cifar10.resnet.classifiers


Classes
-------

.. autoapisummary::

   nebula.core.models.cifar10.resnet.CIFAR10ModelResNet


Functions
---------

.. autoapisummary::

   nebula.core.models.cifar10.resnet.conv_block


Module Contents
---------------

.. py:data:: IMAGE_SIZE
   :value: 32


.. py:data:: BATCH_SIZE

.. py:data:: classifiers

.. py:function:: conv_block(input_channels, num_classes, pool=False)

.. py:class:: CIFAR10ModelResNet(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None, implementation='scratch', classifier='resnet9')

   Bases: :py:obj:`lightning.LightningModule`


   .. py:method:: process_metrics(phase, y_pred, y, loss=None)


   .. py:method:: log_metrics_by_epoch(phase, print_cm=False, plot_cm=False)


   .. py:attribute:: train_metrics


   .. py:attribute:: val_metrics


   .. py:attribute:: test_metrics


   .. py:attribute:: implementation


   .. py:attribute:: classifier


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: criterion


   .. py:attribute:: model


   .. py:attribute:: epoch_global_number


   .. py:method:: forward(x)


   .. py:method:: configure_optimizers()


   .. py:method:: step(batch, batch_idx, phase)


   .. py:method:: training_step(batch, batch_id)


   .. py:method:: on_train_epoch_end()


   .. py:method:: validation_step(batch, batch_idx)


   .. py:method:: on_validation_epoch_end()


   .. py:method:: test_step(batch, batch_idx)


   .. py:method:: on_test_epoch_end()


