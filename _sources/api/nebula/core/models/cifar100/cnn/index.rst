nebula.core.models.cifar100.cnn
===============================

.. py:module:: nebula.core.models.cifar100.cnn


Classes
-------

.. autoapisummary::

   nebula.core.models.cifar100.cnn.CNN


Module Contents
---------------

.. py:class:: CNN(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: criterion


   .. py:attribute:: conv1


   .. py:attribute:: conv2


   .. py:attribute:: res1


   .. py:attribute:: conv3


   .. py:attribute:: conv4


   .. py:attribute:: res2


   .. py:attribute:: classifier


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



