nebula.core.datasets.sentiment140.sentiment140
==============================================

.. py:module:: nebula.core.datasets.sentiment140.sentiment140


Classes
-------

.. autoapisummary::

   nebula.core.datasets.sentiment140.sentiment140.SENTIMENT140
   nebula.core.datasets.sentiment140.sentiment140.Sent140Dataset


Module Contents
---------------

.. py:class:: SENTIMENT140(train=True)

   Bases: :py:obj:`torchvision.datasets.MNIST`


   .. py:method:: dataset_download()


.. py:class:: Sent140Dataset(num_classes=10, partition_id=0, partitions_number=1, batch_size=32, num_workers=4, iid=True, partition='dirichlet', partition_parameter=0.5, seed=42, config=None)

   Bases: :py:obj:`nebula.core.datasets.nebuladataset.NebulaDataset`


   Abstract class for a partitioned dataset.

   Classes inheriting from this class need to implement specific methods
   for loading and partitioning the dataset.


   .. py:method:: initialize_dataset()

      Initialize the dataset. This should load or create the dataset.



   .. py:method:: load_sent14_dataset(train=True)


   .. py:method:: generate_non_iid_map(dataset, partition='dirichlet', partition_parameter=0.5)

      Create a non-iid map of the dataset.



   .. py:method:: generate_iid_map(dataset, partition='balancediid', partition_parameter=2)

      Create an iid map of the dataset.



