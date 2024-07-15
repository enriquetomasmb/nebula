nebula.core.models.mnist.mlp
============================

.. py:module:: nebula.core.models.mnist.mlp


Classes
-------

.. autoapisummary::

   nebula.core.models.mnist.mlp.MNISTModelMLP


Module Contents
---------------

.. py:class:: MNISTModelMLP(input_channels=1, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



