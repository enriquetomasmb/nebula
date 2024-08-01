nebula.core.models.sentiment140.cnn
===================================

.. py:module:: nebula.core.models.sentiment140.cnn


Classes
-------

.. autoapisummary::

   nebula.core.models.sentiment140.cnn.SentimentModelCNN


Module Contents
---------------

.. py:class:: SentimentModelCNN(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: criterion


   .. py:attribute:: filter_sizes
      :value: [2, 3, 4]



   .. py:attribute:: n_filters


   .. py:attribute:: convs


   .. py:attribute:: fc


   .. py:attribute:: dropout


   .. py:attribute:: epoch_global_number


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



