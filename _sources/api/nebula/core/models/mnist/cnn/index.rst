:py:mod:`nebula.core.models.mnist.cnn`
======================================

.. py:module:: nebula.core.models.mnist.cnn


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.models.mnist.cnn.MNISTModelCNN




.. py:class:: MNISTModelCNN(input_channels=1, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)


   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`

   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.

   .. py:method:: forward(x)

      Forward pass of the model.


   .. py:method:: configure_optimizers()

      Optimizer configuration.



