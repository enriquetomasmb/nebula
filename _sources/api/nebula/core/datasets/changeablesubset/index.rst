nebula.core.datasets.changeablesubset
=====================================

.. py:module:: nebula.core.datasets.changeablesubset


Classes
-------

.. autoapisummary::

   nebula.core.datasets.changeablesubset.ChangeableSubset


Module Contents
---------------

.. py:class:: ChangeableSubset(dataset, indices, label_flipping=False, data_poisoning=False, poisoned_persent=0, poisoned_ratio=0, targeted=False, target_label=0, target_changed_label=0, noise_type='salt')

   Bases: :py:obj:`torch.utils.data.Subset`


   .. py:attribute:: new_dataset


   .. py:attribute:: dataset


   .. py:attribute:: indices


   .. py:attribute:: label_flipping


   .. py:attribute:: data_poisoning


   .. py:attribute:: poisoned_persent


   .. py:attribute:: poisoned_ratio


   .. py:attribute:: targeted


   .. py:attribute:: target_label


   .. py:attribute:: target_changed_label


   .. py:attribute:: noise_type


