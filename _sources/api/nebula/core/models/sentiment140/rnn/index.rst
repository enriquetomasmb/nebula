nebula.core.models.sentiment140.rnn
===================================

.. py:module:: nebula.core.models.sentiment140.rnn


Classes
-------

.. autoapisummary::

   nebula.core.models.sentiment140.rnn.SentimentModelRNN


Module Contents
---------------

.. py:class:: SentimentModelRNN(input_channels=3, num_classes=10, learning_rate=0.001, metrics=None, confusion_matrix=None, seed=None)

   Bases: :py:obj:`nebula.core.models.nebulamodel.NebulaModel`


   Abstract class for the NEBULA model.

   This class is an abstract class that defines the interface for the NEBULA model.


   .. py:attribute:: config


   .. py:attribute:: example_input_array


   .. py:attribute:: learning_rate


   .. py:attribute:: embedding_dim
      :value: 300



   .. py:attribute:: hidden_dim
      :value: 256



   .. py:attribute:: n_layers
      :value: 1



   .. py:attribute:: bidirectional
      :value: True



   .. py:attribute:: output_dim


   .. py:attribute:: encoder


   .. py:attribute:: fc


   .. py:attribute:: dropout


   .. py:attribute:: criterion


   .. py:attribute:: l1


   .. py:attribute:: l2


   .. py:attribute:: l3


   .. py:attribute:: epoch_global_number


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



