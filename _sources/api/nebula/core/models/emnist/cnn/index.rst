nebula.core.models.emnist.cnn
=============================

.. py:module:: nebula.core.models.emnist.cnn


Classes
-------

.. autoapisummary::

   nebula.core.models.emnist.cnn.EMNISTModelCNN


Module Contents
---------------

.. py:class:: EMNISTModelCNN(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: criterion


   .. py:attribute:: conv1


   .. py:attribute:: relu


   .. py:attribute:: pool1


   .. py:attribute:: conv2


   .. py:attribute:: pool2


   .. py:attribute:: l1


   .. py:attribute:: l2


   .. py:attribute:: epoch_global_number


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



