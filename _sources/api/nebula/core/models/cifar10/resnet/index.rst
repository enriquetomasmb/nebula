:py:mod:`nebula.core.models.cifar10.resnet`
===========================================

.. py:module:: nebula.core.models.cifar10.resnet


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.models.cifar10.resnet.CIFAR10ModelResNet



Functions
~~~~~~~~~

.. autoapisummary::

   nebula.core.models.cifar10.resnet.conv_block



Attributes
~~~~~~~~~~

.. autoapisummary::

   nebula.core.models.cifar10.resnet.IMAGE_SIZE
   nebula.core.models.cifar10.resnet.BATCH_SIZE
   nebula.core.models.cifar10.resnet.classifiers


.. py:data:: IMAGE_SIZE
   :value: 32

   

.. py:data:: BATCH_SIZE

   

.. py:data:: classifiers

   

.. py:function:: conv_block(input_channels, num_classes, pool=False)


.. py:class:: CIFAR10ModelResNet(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None, implementation='scratch', classifier='resnet9')


   Bases: :py:obj:`lightning.LightningModule`

   .. py:method:: process_metrics(phase, y_pred, y, loss=None)


   .. py:method:: log_metrics_by_epoch(phase, print_cm=False, plot_cm=False)


   .. py:method:: forward(x)


   .. py:method:: configure_optimizers()


   .. py:method:: step(batch, phase)


   .. py:method:: training_step(batch, batch_id)


   .. py:method:: on_train_epoch_end()


   .. py:method:: validation_step(batch, batch_idx)


   .. py:method:: on_validation_epoch_end()


   .. py:method:: test_step(batch, batch_idx)


   .. py:method:: on_test_epoch_end()



