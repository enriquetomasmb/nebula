:py:mod:`nebula.core.models.cifar10.simplemobilenet`
====================================================

.. py:module:: nebula.core.models.cifar10.simplemobilenet


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.models.cifar10.simplemobilenet.SimpleMobileNetV1




.. py:class:: SimpleMobileNetV1(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)


   Bases: :py:obj:`lightning.LightningModule`

   .. py:method:: process_metrics(phase, y_pred, y, loss=None)


   .. py:method:: log_metrics_by_epoch(phase, print_cm=False, plot_cm=False)


   .. py:method:: forward(x)


   .. py:method:: configure_optimizers()



