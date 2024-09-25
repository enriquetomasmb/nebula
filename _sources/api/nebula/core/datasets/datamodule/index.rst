nebula.core.datasets.datamodule
===============================

.. py:module:: nebula.core.datasets.datamodule


Attributes
----------

.. autoapisummary::

   nebula.core.datasets.datamodule.logging_training


Classes
-------

.. autoapisummary::

   nebula.core.datasets.datamodule.DataModule


Module Contents
---------------

.. py:data:: logging_training

.. py:class:: DataModule(train_set, train_set_indices, test_set, test_set_indices, local_test_set_indices, partition_id=0, partitions_number=1, batch_size=32, num_workers=0, val_percent=0.1, label_flipping=False, data_poisoning=False, poisoned_persent=0, poisoned_ratio=0, targeted=False, target_label=0, target_changed_label=0, noise_type='salt')

   Bases: :py:obj:`lightning.LightningDataModule`


   .. py:attribute:: train_set


   .. py:attribute:: train_set_indices


   .. py:attribute:: test_set


   .. py:attribute:: test_set_indices


   .. py:attribute:: local_test_set_indices


   .. py:attribute:: partition_id


   .. py:attribute:: partitions_number


   .. py:attribute:: batch_size


   .. py:attribute:: num_workers


   .. py:attribute:: val_percent


   .. py:attribute:: label_flipping


   .. py:attribute:: data_poisoning


   .. py:attribute:: poisoned_percent


   .. py:attribute:: poisoned_ratio


   .. py:attribute:: targeted


   .. py:attribute:: target_label


   .. py:attribute:: target_changed_label


   .. py:attribute:: noise_type


   .. py:attribute:: tr_subset


   .. py:attribute:: train_size


   .. py:attribute:: val_size


   .. py:attribute:: global_te_subset


   .. py:attribute:: local_te_subset


   .. py:attribute:: train_loader


   .. py:attribute:: val_loader


   .. py:attribute:: test_loader


   .. py:attribute:: global_test_loader


   .. py:attribute:: random_sampler


   .. py:attribute:: bootstrap_loader


   .. py:method:: train_dataloader()


   .. py:method:: val_dataloader()


   .. py:method:: test_dataloader()


   .. py:method:: bootstrap_dataloader()


