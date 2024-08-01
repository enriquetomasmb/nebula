nebula.core.models.emnist.mlp
=============================

.. py:module:: nebula.core.models.emnist.mlp


Classes
-------

.. autoapisummary::

   nebula.core.models.emnist.mlp.EMNISTModelMLP


Module Contents
---------------

.. py:class:: EMNISTModelMLP(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: criterion


   .. py:attribute:: l1


   .. py:attribute:: l2


   .. py:attribute:: l3


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



