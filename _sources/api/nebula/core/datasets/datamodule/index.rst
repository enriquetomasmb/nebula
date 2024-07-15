:py:mod:`nebula.core.datasets.datamodule`
=========================================

.. py:module:: nebula.core.datasets.datamodule


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.datasets.datamodule.DataModule




.. py:class:: DataModule(train_set, train_set_indices, test_set, test_set_indices, local_test_set_indices, partition_id=0, partitions_number=1, batch_size=32, num_workers=0, val_percent=0.1, label_flipping=False, data_poisoning=False, poisoned_persent=0, poisoned_ratio=0, targeted=False, target_label=0, target_changed_label=0, noise_type='salt')


   Bases: :py:obj:`lightning.LightningDataModule`

   .. py:method:: train_dataloader()


   .. py:method:: val_dataloader()


   .. py:method:: test_dataloader()


   .. py:method:: bootstrap_dataloader()



