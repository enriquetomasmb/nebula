nebula.core.models.militarysar.cnn
==================================

.. py:module:: nebula.core.models.militarysar.cnn


Classes
-------

.. autoapisummary::

   nebula.core.models.militarysar.cnn.MilitarySARModelCNN


Module Contents
---------------

.. py:class:: MilitarySARModelCNN(input_channels=2, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: momentum
      :value: 0.9



   .. py:attribute:: weight_decay
      :value: 0.004



   .. py:attribute:: dropout_rate
      :value: 0.5



   .. py:attribute:: criterion


   .. py:attribute:: model


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



