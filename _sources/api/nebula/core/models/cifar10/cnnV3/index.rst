nebula.core.models.cifar10.cnnV3
================================

.. py:module:: nebula.core.models.cifar10.cnnV3


Classes
-------

.. autoapisummary::

   nebula.core.models.cifar10.cnnV3.CIFAR10ModelCNN_V3


Module Contents
---------------

.. py:class:: CIFAR10ModelCNN_V3(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: criterion


   .. py:attribute:: layer1


   .. py:attribute:: layer2


   .. py:attribute:: layer3


   .. py:attribute:: fc_layer


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



