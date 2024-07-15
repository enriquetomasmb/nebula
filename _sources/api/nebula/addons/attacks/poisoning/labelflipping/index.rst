:py:mod:`nebula.addons.attacks.poisoning.labelflipping`
=======================================================

.. py:module:: nebula.addons.attacks.poisoning.labelflipping


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   nebula.addons.attacks.poisoning.labelflipping.labelFlipping



.. py:function:: labelFlipping(dataset, indices, poisoned_persent=0, targeted=False, target_label=4, target_changed_label=7)

   select flipping_persent of labels, and change them to random values.
   :param dataset: the dataset of training data, torch.util.data.dataset like.
   :param indices: Indices of subsets, list like.
   :param flipping_persent: The ratio of labels want to change, float like.


