nebula.addons.attacks.attacks
=============================

.. py:module:: nebula.addons.attacks.attacks


Classes
-------

.. autoapisummary::

   nebula.addons.attacks.attacks.Attack
   nebula.addons.attacks.attacks.GLLNeuronInversionAttack
   nebula.addons.attacks.attacks.NoiseInjectionAttack
   nebula.addons.attacks.attacks.SwappingWeightsAttack
   nebula.addons.attacks.attacks.DelayerAttack


Functions
---------

.. autoapisummary::

   nebula.addons.attacks.attacks.create_attack


Module Contents
---------------

.. py:function:: create_attack(attack_name)

   Function to create an attack object from its name.


.. py:class:: Attack

   .. py:method:: attack(received_weights)
      :abstractmethod:


      Function to perform the attack on the received weights. It should return the
      attacked weights.



.. py:class:: GLLNeuronInversionAttack(strength=5.0, perc=1.0)

   Bases: :py:obj:`Attack`


   Function to perform neuron inversion attack on the received weights.


   .. py:attribute:: strength


   .. py:attribute:: perc


   .. py:method:: attack(received_weights)

      Function to perform the attack on the received weights. It should return the
      attacked weights.



.. py:class:: NoiseInjectionAttack(strength=10000, perc=1.0)

   Bases: :py:obj:`Attack`


   Function to perform noise injection attack on the received weights.


   .. py:attribute:: strength


   .. py:attribute:: perc


   .. py:method:: attack(received_weights)

      Function to perform the attack on the received weights. It should return the
      attacked weights.



.. py:class:: SwappingWeightsAttack(layer_idx=0)

   Bases: :py:obj:`Attack`


   Function to perform swapping weights attack on the received weights. Note that this
   attack performance is not consistent due to its stochasticity.

   Warning: depending on the layer the code may not work (due to reshaping in between),
   or it may be slow (scales quadratically with the layer size).
   Do not apply to last layer, as it would make the attack detectable (high loss
   on malicious node).


   .. py:attribute:: layer_idx


   .. py:method:: attack(received_weights)

      Function to perform the attack on the received weights. It should return the
      attacked weights.



.. py:class:: DelayerAttack

   Bases: :py:obj:`Attack`


   Function to perform delayer attack on the received weights. It delays the
   weights for an indefinite number of rounds.


   .. py:attribute:: weights
      :value: None



   .. py:method:: attack(received_weights)

      Function to perform the attack on the received weights. It should return the
      attacked weights.



