nebula.core.datasets.nebuladataset
==================================

.. py:module:: nebula.core.datasets.nebuladataset


Classes
-------

.. autoapisummary::

   nebula.core.datasets.nebuladataset.NebulaDataset


Module Contents
---------------

.. py:class:: NebulaDataset(num_classes=10, partition_id=0, partitions_number=1, batch_size=32, num_workers=4, iid=True, partition='dirichlet', partition_parameter=0.5, seed=42, config=None)

   Bases: :py:obj:`torch.utils.data.Dataset`, :py:obj:`abc.ABC`


   Abstract class for a partitioned dataset.

   Classes inheriting from this class need to implement specific methods
   for loading and partitioning the dataset.


   .. py:attribute:: num_classes


   .. py:attribute:: partition_id


   .. py:attribute:: partitions_number


   .. py:attribute:: batch_size


   .. py:attribute:: num_workers


   .. py:attribute:: iid


   .. py:attribute:: partition


   .. py:attribute:: partition_parameter


   .. py:attribute:: seed


   .. py:attribute:: config


   .. py:attribute:: train_set
      :value: None



   .. py:attribute:: train_indices_map
      :value: None



   .. py:attribute:: test_set
      :value: None



   .. py:attribute:: test_indices_map
      :value: None



   .. py:method:: initialize_dataset()
      :abstractmethod:


      Initialize the dataset. This should load or create the dataset.



   .. py:method:: generate_non_iid_map(dataset, partition='dirichlet')
      :abstractmethod:


      Create a non-iid map of the dataset.



   .. py:method:: generate_iid_map(dataset)
      :abstractmethod:


      Create an iid map of the dataset.



   .. py:method:: plot_data_distribution(dataset, partitions_map)

      Plot the data distribution of the dataset.

      Plot the data distribution of the dataset according to the partitions map provided.

      :param dataset: The dataset to plot (torch.utils.data.Dataset).
      :param partitions_map: The map of the dataset partitions.



   .. py:method:: visualize_tsne(dataset)


   .. py:method:: dirichlet_partition(dataset, alpha=0.5)

      Partition the dataset into multiple subsets using a Dirichlet distribution.

      This function divides a dataset into a specified number of subsets (federated clients),
      where each subset has a different class distribution. The class distribution in each
      subset is determined by a Dirichlet distribution, making the partition suitable for
      simulating non-IID (non-Independently and Identically Distributed) data scenarios in
      federated learning.

      :param dataset: The dataset to partition. It should have
                      'data' and 'targets' attributes.
      :type dataset: torch.utils.data.Dataset
      :param alpha: The concentration parameter of the Dirichlet distribution. A lower
                    alpha value leads to more imbalanced partitions.
      :type alpha: float

      :returns:

                A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                    and values are lists of indices corresponding to the samples in each subset.
      :rtype: dict

      The function ensures that each class is represented in each subset but with varying
      proportions. The partitioning process involves iterating over each class, shuffling
      the indices of that class, and then splitting them according to the Dirichlet
      distribution. The function also prints the class distribution in each subset for reference.

      Example usage:
          federated_data = dirichlet_partition(my_dataset, alpha=0.5)
          # This creates federated data subsets with varying class distributions based on
          # a Dirichlet distribution with alpha = 0.5.



   .. py:method:: homo_partition(dataset)

      Homogeneously partition the dataset into multiple subsets.

      This function divides a dataset into a specified number of subsets, where each subset
      is intended to have a roughly equal number of samples. This method aims to ensure a
      homogeneous distribution of data across all subsets. It's particularly useful in
      scenarios where a uniform distribution of data is desired among all federated learning
      clients.

      :param dataset: The dataset to partition. It should have
                      'data' and 'targets' attributes.
      :type dataset: torch.utils.data.Dataset

      :returns:

                A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                    and values are lists of indices corresponding to the samples in each subset.
      :rtype: dict

      The function randomly shuffles the entire dataset and then splits it into the number
      of subsets specified by `partitions_number`. It ensures that each subset has a similar number
      of samples. The function also prints the class distribution in each subset for reference.

      Example usage:
          federated_data = homo_partition(my_dataset)
          # This creates federated data subsets with homogeneous distribution.



   .. py:method:: balanced_iid_partition(dataset)

      Partition the dataset into balanced and IID (Independent and Identically Distributed)
      subsets for each client.

      This function divides a dataset into a specified number of subsets (federated clients),
      where each subset has an equal class distribution. This makes the partition suitable for
      simulating IID data scenarios in federated learning.

      :param dataset: The dataset to partition. It should be a list of tuples where each
                      tuple represents a data sample and its corresponding label.
      :type dataset: list

      :returns:

                A dictionary where keys are client IDs (ranging from 0 to partitions_number-1) and
                        values are lists of indices corresponding to the samples assigned to each client.
      :rtype: dict

      The function ensures that each class is represented equally in each subset. The
      partitioning process involves iterating over each class, shuffling the indices of that class,
      and then splitting them equally among the clients. The function does not print the class
      distribution in each subset.

      Example usage:
          federated_data = balanced_iid_partition(my_dataset)
          # This creates federated data subsets with equal class distributions.



   .. py:method:: unbalanced_iid_partition(dataset, imbalance_factor=2)

      Partition the dataset into multiple IID (Independent and Identically Distributed)
      subsets with different size.

      This function divides a dataset into a specified number of IID subsets (federated
      clients), where each subset has a different number of samples. The number of samples
      in each subset is determined by an imbalance factor, making the partition suitable
      for simulating imbalanced data scenarios in federated learning.

      :param dataset: The dataset to partition. It should be a list of tuples where
                      each tuple represents a data sample and its corresponding label.
      :type dataset: list
      :param imbalance_factor: The factor to determine the degree of imbalance
                               among the subsets. A lower imbalance factor leads to more
                               imbalanced partitions.
      :type imbalance_factor: float

      :returns:

                A dictionary where keys are client IDs (ranging from 0 to partitions_number-1) and
                        values are lists of indices corresponding to the samples assigned to each client.
      :rtype: dict

      The function ensures that each class is represented in each subset but with varying
      proportions. The partitioning process involves iterating over each class, shuffling
      the indices of that class, and then splitting them according to the calculated subset
      sizes. The function does not print the class distribution in each subset.

      Example usage:
          federated_data = unbalanced_iid_partition(my_dataset, imbalance_factor=2)
          # This creates federated data subsets with varying number of samples based on
          # an imbalance factor of 2.



   .. py:method:: percentage_partition(dataset, percentage=20)

      Partition a dataset into multiple subsets with a specified level of non-IID-ness.

      This function divides a dataset into a specified number of subsets (federated
      clients), where each subset has a different class distribution. The class
      distribution in each subset is determined by a specified percentage, making the
      partition suitable for simulating non-IID (non-Independently and Identically
      Distributed) data scenarios in federated learning.

      :param dataset: The dataset to partition. It should have
                      'data' and 'targets' attributes.
      :type dataset: torch.utils.data.Dataset
      :param percentage: A value between 0 and 100 that specifies the desired
                         level of non-IID-ness for the labels of the federated data.
                         This percentage controls the imbalance in the class distribution
                         across different subsets.
      :type percentage: int

      :returns:

                A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                    and values are lists of indices corresponding to the samples in each subset.
      :rtype: dict

      The function ensures that the number of classes in each subset varies based on the selected
      percentage. The partitioning process involves iterating over each class, shuffling the
      indices of that class, and then splitting them according to the calculated subset sizes.
      The function also prints the class distribution in each subset for reference.

      Example usage:
          federated_data = percentage_partition(my_dataset, percentage=20)
          # This creates federated data subsets with varying class distributions based on
          # a percentage of 20.



   .. py:method:: plot_all_data_distribution(dataset, partitions_map)

      Plot all of the data distribution of the dataset according to the partitions map provided.

      :param dataset: The dataset to plot (torch.utils.data.Dataset).
      :param partitions_map: The map of the dataset partitions.



