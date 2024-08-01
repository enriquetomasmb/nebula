nebula.core.models.syscall.autoencoder
======================================

.. py:module:: nebula.core.models.syscall.autoencoder


Classes
-------

.. autoapisummary::

   nebula.core.models.syscall.autoencoder.SyscallModelAutoencoder


Module Contents
---------------

.. py:class:: SyscallModelAutoencoder(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: criterion


   .. py:attribute:: fc1


   .. py:attribute:: fc2


   .. py:attribute:: fc3


   .. py:attribute:: fc4


   .. py:attribute:: fc5


   .. py:attribute:: fc6


   .. py:attribute:: epoch_global_number


   .. py:method:: encode(x)


   .. py:method:: decode(x)


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



