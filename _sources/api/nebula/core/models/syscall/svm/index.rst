nebula.core.models.syscall.svm
==============================

.. py:module:: nebula.core.models.syscall.svm


Classes
-------

.. autoapisummary::

   nebula.core.models.syscall.svm.SyscallModelSGDOneClassSVM


Module Contents
---------------

.. py:class:: SyscallModelSGDOneClassSVM(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



   .. py:method:: hinge_loss(y)


   .. py:method:: step(batch, phase)

      Training/validation/test step.



