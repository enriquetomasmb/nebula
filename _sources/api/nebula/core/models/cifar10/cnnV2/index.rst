:py:mod:`nebula.core.models.cifar10.cnnV2`
==========================================

.. py:module:: nebula.core.models.cifar10.cnnV2


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.models.cifar10.cnnV2.CIFAR10ModelCNN_V2




.. py:class:: CIFAR10ModelCNN_V2(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)


   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`

   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.

   .. py:method:: forward(x)

      Forward pass of the model.


   .. py:method:: configure_optimizers()

      Optimizer configuration.



