nebula.core.models.cifar10.cnn
==============================

.. py:module:: nebula.core.models.cifar10.cnn


Classes
-------

.. autoapisummary::

   nebula.core.models.cifar10.cnn.CIFAR10ModelCNN


Module Contents
---------------

.. py:class:: CIFAR10ModelCNN(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: criterion


   .. py:attribute:: conv1


   .. py:attribute:: conv2


   .. py:attribute:: conv3


   .. py:attribute:: pool


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



