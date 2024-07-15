nebula.core.datasets.militarysar.militarysar
============================================

.. py:module:: nebula.core.datasets.militarysar.militarysar


Classes
-------

.. autoapisummary::

   nebula.core.datasets.militarysar.militarysar.RandomCrop
   nebula.core.datasets.militarysar.militarysar.CenterCrop
   nebula.core.datasets.militarysar.militarysar.MilitarySAR
   nebula.core.datasets.militarysar.militarysar.MilitarySARDataset


Module Contents
---------------

.. py:class:: RandomCrop(size)

   Bases: :py:obj:`object`


.. py:class:: CenterCrop(size)

   Bases: :py:obj:`object`


.. py:class:: MilitarySAR(name='soc', is_train=False, transform=None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   .. py:method:: get_targets()


.. py:class:: MilitarySARDataset(num_classes=10, partition_id=0, partitions_number=1, batch_size=32, num_workers=4, iid=True, partition='dirichlet', partition_parameter=0.5, seed=42, config=None)

   Bases: :py:obj:`nebula.core.datasets.nebuladataset.NebulaDataset`


   Abstract class for a partitioned dataset.

   Classes inheriting from this class need to implement specific methods
   for loading and partitioning the dataset.


   .. py:method:: initialize_dataset()

      Initialize the dataset. This should load or create the dataset.



   .. py:method:: load_militarysar_dataset(train=True)


   .. py:method:: generate_non_iid_map(dataset, partition='dirichlet', partition_parameter=0.5)

      Create a non-iid map of the dataset.



   .. py:method:: generate_iid_map(dataset, partition='balancediid', partition_parameter=2)

      Create an iid map of the dataset.



