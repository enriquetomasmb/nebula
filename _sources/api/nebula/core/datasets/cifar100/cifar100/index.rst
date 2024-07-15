:py:mod:`nebula.core.datasets.cifar100.cifar100`
================================================

.. py:module:: nebula.core.datasets.cifar100.cifar100


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   nebula.core.datasets.cifar100.cifar100.CIFAR100Dataset




.. py:class:: CIFAR100Dataset(num_classes=100, partition_id=0, partitions_number=1, batch_size=32, num_workers=4, iid=True, partition='dirichlet', partition_parameter=0.5, seed=42, config=None)


   Bases: :py:obj:`nebula.core.datasets.nebuladataset.NebulaDataset`

   Abstract class for a partitioned dataset.

   Classes inheriting from this class need to implement specific methods
   for loading and partitioning the dataset.

   .. py:method:: initialize_dataset()

      Initialize the dataset. This should load or create the dataset.


   .. py:method:: load_cifar100_dataset(train=True)


   .. py:method:: generate_non_iid_map(dataset, partition='dirichlet', partition_parameter=0.5)

      Create a non-iid map of the dataset.


   .. py:method:: generate_iid_map(dataset, partition='balancediid', partition_parameter=2)

      Create an iid map of the dataset.



