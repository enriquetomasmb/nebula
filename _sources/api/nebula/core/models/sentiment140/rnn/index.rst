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


   .. py:method:: forward(x)

      Forward pass of the model.



   .. py:method:: configure_optimizers()

      Optimizer configuration.



