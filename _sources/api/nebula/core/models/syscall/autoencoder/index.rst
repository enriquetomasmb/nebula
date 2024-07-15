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


   .. py:method:: encode(x)


   .. py:method:: decode(x)


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



