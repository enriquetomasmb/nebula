nebula.core.models.wadi.mlp
===========================

.. py:module:: nebula.core.models.wadi.mlp


Classes
-------

.. autoapisummary::

   nebula.core.models.wadi.mlp.WADIModelMLP


Module Contents
---------------

.. py:class:: WADIModelMLP(input_channels=1, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

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


   .. py:attribute:: l4


   .. py:attribute:: l5


   .. py:attribute:: l6


   .. py:attribute:: l7


   .. py:attribute:: l8


   .. py:attribute:: l9


   .. py:attribute:: epoch_global_number


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



