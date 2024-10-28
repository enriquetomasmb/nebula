import logging
from abc import ABC, abstractmethod
import time
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, ConcatDataset
from nebula.core.utils.deterministic import enable_deterministic
from torch.utils.data import Subset

class NebulaDataset(Dataset, ABC):
    """
    Abstract class for a partitioned dataset.

    Classes inheriting from this class need to implement specific methods
    for loading and partitioning the dataset.
    """

    def __init__(
        self,
        num_classes=10,
        partition_id=0,
        partitions_number=1,
        batch_size=32,
        num_workers=4,
        iid=True,
        partition="dirichlet",
        partition_parameter=0.5,
        seed=42,
        config=None,
    ):
        super().__init__()

        if partition_id < 0 or partition_id >= partitions_number:
            raise ValueError(f"partition_id {partition_id} is out of range for partitions_number {partitions_number}")

        self.num_classes = num_classes
        self.partition_id = partition_id
        self.partitions_number = partitions_number
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.iid = iid
        self.partition = partition
        self.partition_parameter = partition_parameter
        self.seed = seed
        self.config = config

        self.train_set = None
        self.train_indices_map = None
        self.test_set = None
        self.test_indices_map = None


        # MIA setting
        self.indexing_map = None  # this is used to decompose the MIA result from micro view for the in eval group.
        self.shadow_train = None
        self.shadow_test = None
        self.in_eval = None
        self.out_eval = None

        enable_deterministic(config)

        if self.partition_id == 0:
            self.initialize_dataset()
        else:
            max_tries = 10
            for i in range(max_tries):
                try:
                    self.initialize_dataset()
                    break
                except Exception as e:
                    print(f"Error loading dataset: {e}. Retrying {i+1}/{max_tries} in 5 seconds...")
                    time.sleep(5)

    def initialize_eval_dataset(self, in_idxs, out_idxs):
        """
            Initializes the evaluation datasets.

            Args:
                in_idxs (list): List of indices for the in-sample evaluation dataset.
                out_idxs (list): List of indices for the out-of-sample evaluation dataset.

            This method assigns the provided indices to the class attributes for in-sample and out-sample evaluation datasets.
        """
        self.in_eval = in_idxs
        self.out_eval = out_idxs
        print(self.out_eval)

    def initialize_shadow_dataset(self, out_idxs, shadow_size, shadow_number):
        """
            Initializes the datasets for training and testing shadow models using a combined dataset approach.

            Args:
                out_idxs (list): List of indices for the out-of-sample training dataset.
                shadow_size (int): Size of each shadow dataset.
                shadow_number (int): Number of shadow datasets to create.

            Returns:
                ConcatDataset: A combined dataset of unused training data and test data.

            Raises:
                ValueError: If the combined size of the remaining training dataset and the test set is smaller than twice the shadow dataset size.

            This method combines the unused training data and the test data into a single dataset, shuffles the combined dataset,
            and selects random indices for the shadow training and testing datasets.
        """
        test_indices = np.arange(len(self.test_set))
        if len(out_idxs) + len(test_indices) < 2 * shadow_size:
            raise ValueError(
                "The remaining unused training dataset size and the test dataset size is smaller than shadow dataset size!")

        unused_train_subset = Subset(self.train_set, out_idxs)
        test_subset = Subset(self.test_set, test_indices)

        combined_dataset = ConcatDataset([unused_train_subset, test_subset])
        total_size = len(combined_dataset)

        np.random.seed(self.seed)
        combined_indices = np.random.permutation(len(combined_dataset))
        logging.info(combined_indices)
        logging.info(len(combined_indices))
        shadow_train_indices = combined_indices[:total_size // 2]
        shadow_test_indices = combined_indices[total_size // 2:]

        shadow_train_indices_ls = []
        shadow_test_indices_ls = []

        for i in range(shadow_number):
            shadow_train_index = np.random.choice(shadow_train_indices, size=shadow_size, replace=True)
            shadow_test_index = np.random.choice(shadow_test_indices, size=shadow_size, replace=True)

            shadow_train_indices_ls.append(shadow_train_index)
            shadow_test_indices_ls.append(shadow_test_index)

        self.shadow_train = shadow_train_indices_ls
        self.shadow_test = shadow_test_indices_ls

        return combined_dataset

    @abstractmethod
    def initialize_dataset(self):
        """
        Initialize the dataset. This should load or create the dataset.
        """
        pass

    @abstractmethod
    def generate_non_iid_map(self, dataset, partition="dirichlet"):
        """
        Create a non-iid map of the dataset.
        """
        pass

    @abstractmethod
    def generate_iid_map(self, dataset):
        """
        Create an iid map of the dataset.
        """
        pass

    def plot_data_distribution(self, dataset, partitions_map):
        """
        Plot the data distribution of the dataset.

        Plot the data distribution of the dataset according to the partitions map provided.

        Args:
            dataset: The dataset to plot (torch.utils.data.Dataset).
            partitions_map: The map of the dataset partitions.
        """
        # Plot the data distribution of the dataset, one graph per partition
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")

        for i in range(self.partitions_number):
            indices = partitions_map[i]
            class_counts = [0] * self.num_classes
            for idx in indices:
                label = dataset.targets[idx]
                class_counts[label] += 1
            print(f"Participant {i+1} class distribution: {class_counts}")
            plt.figure()
            plt.bar(range(self.num_classes), class_counts)
            plt.xlabel("Class")
            plt.ylabel("Number of samples")
            plt.xticks(range(self.num_classes))
            if self.iid:
                plt.title(f"Participant {i+1} class distribution (IID)")
            else:
                plt.title(f"Participant {i+1} class distribution (Non-IID - {self.partition}) - {self.partition_parameter}")
            plt.tight_layout()
            path_to_save = f"{self.config.participant['tracking_args']['log_dir']}/{self.config.participant['scenario_args']['name']}/participant_{i+1}_class_distribution_{'iid' if self.iid else 'non_iid'}{'_' + self.partition if not self.iid else ''}.png"
            plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
            plt.close()

        plt.figure()
        max_point_size = 500
        min_point_size = 0

        for i in range(self.partitions_number):
            class_counts = [0] * self.num_classes
            indices = partitions_map[i]
            for idx in indices:
                label = dataset.targets[idx]
                class_counts[label] += 1

            # Normalize the point sizes for this partition
            max_samples_partition = max(class_counts)
            sizes = [(size / max_samples_partition) * (max_point_size - min_point_size) + min_point_size for size in class_counts]
            plt.scatter([i] * self.num_classes, range(self.num_classes), s=sizes, alpha=0.5)

        plt.xlabel("Participant")
        plt.ylabel("Class")
        plt.xticks(range(self.partitions_number))
        plt.yticks(range(self.num_classes))
        if self.iid:
            plt.title(f"Participant {i+1} class distribution (IID)")
        else:
            plt.title(f"Participant {i+1} class distribution (Non-IID - {self.partition}) - {self.partition_parameter}")
        plt.tight_layout()

        # Saves the distribution display with circles of different size
        path_to_save = f"{self.config.participant['tracking_args']['log_dir']}/{self.config.participant['scenario_args']['name']}/class_distribution_{'iid' if self.iid else 'non_iid'}{'_' + self.partition if not self.iid else ''}.png"
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()

        if hasattr(self, "tsne") and self.tsne:
            self.visualize_tsne(dataset)

    def visualize_tsne(self, dataset):
        import matplotlib.pyplot as plt
        import seaborn as sns

        X = []  # List for storing the characteristics of the samples
        y = []  # Ready to store the labels of the samples
        for idx in range(len(dataset)):  # Assuming that 'dataset' is a list or array of your samples
            sample, label = dataset[idx]
            X.append(sample.flatten())
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(X)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y, palette=sns.color_palette("hsv", self.num_classes), legend="full", alpha=0.7)

        plt.title("t-SNE visualization of the dataset")
        plt.xlabel("t-SNE axis 1")
        plt.ylabel("t-SNE axis 2")
        plt.legend(title="Class")
        plt.tight_layout()

        path_to_save_tsne = f"{self.config.participant['tracking_args']['log_dir']}/{self.config.participant['scenario_args']['name']}/tsne_visualization.png"
        plt.savefig(path_to_save_tsne, dpi=300, bbox_inches="tight")
        plt.close()

    def dirichlet_partition(self, dataset, alpha=0.5):
        """
        Partition the dataset into multiple subsets using a Dirichlet distribution.

        This function divides a dataset into a specified number of subsets (federated clients),
        where each subset has a different class distribution. The class distribution in each
        subset is determined by a Dirichlet distribution, making the partition suitable for
        simulating non-IID (non-Independently and Identically Distributed) data scenarios in
        federated learning.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have
                                                'data' and 'targets' attributes.
            alpha (float): The concentration parameter of the Dirichlet distribution. A lower
                        alpha value leads to more imbalanced partitions.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                and values are lists of indices corresponding to the samples in each subset.

        The function ensures that each class is represented in each subset but with varying
        proportions. The partitioning process involves iterating over each class, shuffling
        the indices of that class, and then splitting them according to the Dirichlet
        distribution. The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = dirichlet_partition(my_dataset, alpha=0.5)
            # This creates federated data subsets with varying class distributions based on
            # a Dirichlet distribution with alpha = 0.5.
        """
        np.random.seed(self.seed)
        if isinstance(dataset.targets, np.ndarray):
            y_train = dataset.targets
        elif hasattr(dataset.targets, "numpy"):
            y_train = dataset.targets.numpy()
        else:
            y_train = np.asarray(dataset.targets)

        min_size = 0
        K = np.unique(y_train)
        N = y_train.shape[0]
        n_nets = self.partitions_number
        net_dataidx_map = {}
        #  ?  ? ? ? ? ? ? ? ?? ???? ? ? ?? ? ? ? ? ?
        node_size = int(self.config.participant["mia_args"]["data_size"])
        if node_size:
            restricted_size = n_nets * node_size
            idxs = np.random.permutation(N)[:restricted_size]
            out_idxs = np.random.permutation(N)[restricted_size:]
            X_train, y_train = dataset.data, np.array(dataset.targets)
            X_train, y_train = X_train[idxs], y_train[idxs]
            N = restricted_size
        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in K:
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)

                if len(idx_k) > 0:
                    proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                    proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # partitioned_datasets = []
        for i in range(self.partitions_number):
            #    subset = torch.utils.data.Subset(dataset, net_dataidx_map[i])
            #    partitioned_datasets.append(subset)

            # Print class distribution in the current partition
            class_counts = [0] * self.num_classes
            for idx in net_dataidx_map[i]:
                label = dataset.targets[idx]
                class_counts[label] += 1
            # print(f"Partition {i+1} class distribution: {class_counts}")

        if self.config.participant["mia_args"]["attack_type"] != "No Attack":
            self.initialize_eval_dataset(idxs, out_idxs)
            if self.config.participant["mia_args"]["attack_type"] == "Shadow Model Based MIA" \
                    or self.config.participant["mia_args"]["metric_detail"] in {"Prediction Class Confidence",
                                                                                "Prediction Class Entropy",
                                                                                "Prediction Modified Entropy"}:
                self.initialize_shadow_dataset(out_idxs, node_size * n_nets,
                                               self.config.participant["mia_args"]["shadow_model_number"])

        self.indexing_map = net_dataidx_map

        return net_dataidx_map

    def homo_partition(self, dataset):
        """
        Homogeneously partition the dataset into multiple subsets.

        This function divides a dataset into a specified number of subsets, where each subset
        is intended to have a roughly equal number of samples. This method aims to ensure a
        homogeneous distribution of data across all subsets. It's particularly useful in
        scenarios where a uniform distribution of data is desired among all federated learning
        clients.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have
                                                'data' and 'targets' attributes.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                and values are lists of indices corresponding to the samples in each subset.

        The function randomly shuffles the entire dataset and then splits it into the number
        of subsets specified by `partitions_number`. It ensures that each subset has a similar number
        of samples. The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = homo_partition(my_dataset)
            # This creates federated data subsets with homogeneous distribution.
        """
        np.random.seed(self.seed)
        n_nets = self.partitions_number

        n_train = len(dataset.targets)
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        node_size = int(self.config.participant["mia_args"]["data_size"])
        if node_size:
            restricted_size = n_nets * node_size
            in_dxs = idxs[:restricted_size]
            if n_train >= 2 * restricted_size:
                out_idxs = idxs[restricted_size:2 * restricted_size]
            else:
                print("""
                Warning: The out evaluation dataset is not enough to match the in evaluation dataset size.
                You may want to reconsider the evaluation of the precision of MIA here.
                """)
                out_idxs = idxs[restricted_size:]


        # partitioned_datasets = []
        for i in range(self.partitions_number):
            # subset = torch.utils.data.Subset(dataset, net_dataidx_map[i])
            # partitioned_datasets.append(subset)

            # Print class distribution in the current partition
            class_counts = [0] * self.num_classes
            for idx in net_dataidx_map[i]:
                label = dataset.targets[idx]
                class_counts[label] += 1
            print(f"Partition {i+1} class distribution: {class_counts}")

        if self.config.participant["mia_args"]["attack_type"] != "No Attack":
            self.initialize_eval_dataset(in_dxs, out_idxs)
            if self.config.participant["mia_args"]["attack_type"] == "Shadow Model Based MIA" \
                    or self.config.participant["mia_args"]["metric_detail"] in {"Prediction Class Confidence",
                                                                                "Prediction Class Entropy",
                                                                                "Prediction Modified Entropy"}:
                self.initialize_shadow_dataset(out_idxs, node_size * n_nets,
                                               self.config.participant["mia_args"]["shadow_model_number"])

        self.indexing_map = net_dataidx_map

        return net_dataidx_map

    def balanced_iid_partition(self, dataset):
        """
        Partition the dataset into balanced and IID (Independent and Identically Distributed)
        subsets for each client.

        This function divides a dataset into a specified number of subsets (federated clients),
        where each subset has an equal class distribution. This makes the partition suitable for
        simulating IID data scenarios in federated learning.

        Args:
            dataset (list): The dataset to partition. It should be a list of tuples where each
                            tuple represents a data sample and its corresponding label.

        Returns:
            dict: A dictionary where keys are client IDs (ranging from 0 to partitions_number-1) and
                    values are lists of indices corresponding to the samples assigned to each client.

        The function ensures that each class is represented equally in each subset. The
        partitioning process involves iterating over each class, shuffling the indices of that class,
        and then splitting them equally among the clients. The function does not print the class
        distribution in each subset.

        Example usage:
            federated_data = balanced_iid_partition(my_dataset)
            # This creates federated data subsets with equal class distributions.
        """
        np.random.seed(self.seed)
        num_clients = self.partitions_number
        clients_data = {i: [] for i in range(num_clients)}

        # Get the labels from the dataset
        if isinstance(dataset.targets, np.ndarray):
            labels = dataset.targets
        elif hasattr(dataset.targets, "numpy"):  # Check if it's a tensor with .numpy() method
            labels = dataset.targets.numpy()
        else:  # If it's a list
            labels = np.asarray(dataset.targets)

        label_counts = np.bincount(labels)
        min_label = label_counts.argmin()
        min_count = label_counts[min_label]

        for label in range(self.num_classes):
            # Get the indices of the same label samples
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)

            # Split the data based on their labels
            samples_per_client = min_count // num_clients

            for i in range(num_clients):
                start_idx = i * samples_per_client
                end_idx = (i + 1) * samples_per_client
                clients_data[i].extend(label_indices[start_idx:end_idx])

        return clients_data

    def unbalanced_iid_partition(self, dataset, imbalance_factor=2):
        """
        Partition the dataset into multiple IID (Independent and Identically Distributed)
        subsets with different size.

        This function divides a dataset into a specified number of IID subsets (federated
        clients), where each subset has a different number of samples. The number of samples
        in each subset is determined by an imbalance factor, making the partition suitable
        for simulating imbalanced data scenarios in federated learning.

        Args:
            dataset (list): The dataset to partition. It should be a list of tuples where
                            each tuple represents a data sample and its corresponding label.
            imbalance_factor (float): The factor to determine the degree of imbalance
                                    among the subsets. A lower imbalance factor leads to more
                                    imbalanced partitions.

        Returns:
            dict: A dictionary where keys are client IDs (ranging from 0 to partitions_number-1) and
                    values are lists of indices corresponding to the samples assigned to each client.

        The function ensures that each class is represented in each subset but with varying
        proportions. The partitioning process involves iterating over each class, shuffling
        the indices of that class, and then splitting them according to the calculated subset
        sizes. The function does not print the class distribution in each subset.

        Example usage:
            federated_data = unbalanced_iid_partition(my_dataset, imbalance_factor=2)
            # This creates federated data subsets with varying number of samples based on
            # an imbalance factor of 2.
        """
        np.random.seed(self.seed)
        num_clients = self.partitions_number
        clients_data = {i: [] for i in range(num_clients)}

        # Get the labels from the dataset
        labels = np.array([dataset.targets[idx] for idx in range(len(dataset))])
        label_counts = np.bincount(labels)

        min_label = label_counts.argmin()
        min_count = label_counts[min_label]

        # Set the initial_subset_size
        initial_subset_size = min_count // num_clients

        # Calculate the number of samples for each subset based on the imbalance factor
        subset_sizes = [initial_subset_size]
        for i in range(1, num_clients):
            subset_sizes.append(int(subset_sizes[i - 1] * ((imbalance_factor - 1) / imbalance_factor)))

        for label in range(self.num_classes):
            # Get the indices of the same label samples
            label_indices = np.where(labels == label)[0]
            np.random.shuffle(label_indices)

            # Split the data based on their labels
            start = 0
            for i in range(num_clients):
                end = start + subset_sizes[i]
                clients_data[i].extend(label_indices[start:end])
                start = end

        return clients_data

    def percentage_partition(self, dataset, percentage=20):
        """
        Partition a dataset into multiple subsets with a specified level of non-IID-ness.

        This function divides a dataset into a specified number of subsets (federated
        clients), where each subset has a different class distribution. The class
        distribution in each subset is determined by a specified percentage, making the
        partition suitable for simulating non-IID (non-Independently and Identically
        Distributed) data scenarios in federated learning.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have
                                                'data' and 'targets' attributes.
            percentage (int): A value between 0 and 100 that specifies the desired
                                level of non-IID-ness for the labels of the federated data.
                                This percentage controls the imbalance in the class distribution
                                across different subsets.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                and values are lists of indices corresponding to the samples in each subset.

        The function ensures that the number of classes in each subset varies based on the selected
        percentage. The partitioning process involves iterating over each class, shuffling the
        indices of that class, and then splitting them according to the calculated subset sizes.
        The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = percentage_partition(my_dataset, percentage=20)
            # This creates federated data subsets with varying class distributions based on
            # a percentage of 20.
        """
        np.random.seed(self.seed)
        if isinstance(dataset.targets, np.ndarray):
            y_train = dataset.targets
        elif hasattr(dataset.targets, "numpy"):  # Check if it's a tensor with .numpy() method
            y_train = dataset.targets.numpy()
        else:  # If it's a list
            y_train = np.asarray(dataset.targets)

        num_classes = self.num_classes
        num_subsets = self.partitions_number
        class_indices = {i: np.where(y_train == i)[0] for i in range(num_classes)}

        # Get the labels from the dataset
        labels = np.array([dataset.targets[idx] for idx in range(len(dataset))])
        label_counts = np.bincount(labels)

        min_label = label_counts.argmin()
        min_count = label_counts[min_label]

        classes_per_subset = int(num_classes * percentage / 100)
        if classes_per_subset < 1:
            raise ValueError("The percentage is too low to assign at least one class to each subset.")

        subset_indices = [[] for _ in range(num_subsets)]
        class_list = list(range(num_classes))
        np.random.shuffle(class_list)

        for i in range(num_subsets):
            for j in range(classes_per_subset):
                # Use modulo operation to cycle through the class_list
                class_idx = class_list[(i * classes_per_subset + j) % num_classes]
                indices = class_indices[class_idx]
                np.random.shuffle(indices)
                # Select approximately 50% of the indices
                subset_indices[i].extend(indices[: min_count // 2])

            class_counts = np.bincount(np.array([dataset.targets[idx] for idx in subset_indices[i]]))
            print(f"Partition {i+1} class distribution: {class_counts.tolist()}")

        partitioned_datasets = {i: subset_indices[i] for i in range(num_subsets)}

        return partitioned_datasets

    def plot_all_data_distribution(self, dataset, partitions_map):
        """

        Plot all of the data distribution of the dataset according to the partitions map provided.

        Args:
            dataset: The dataset to plot (torch.utils.data.Dataset).
            partitions_map: The map of the dataset partitions.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")

        num_clients = len(partitions_map)
        num_classes = self.num_classes

        plt.figure(figsize=(12, 8))

        label_distribution = [[] for _ in range(num_classes)]
        for c_id, idc in partitions_map.items():
            for idx in idc:
                label_distribution[dataset.targets[idx]].append(c_id)

        plt.hist(label_distribution, stacked=True, bins=np.arange(-0.5, num_clients + 1.5, 1), label=dataset.classes, rwidth=0.5)
        plt.xticks(np.arange(num_clients), ["Participant %d" % (c_id + 1) for c_id in range(num_clients)])
        plt.title("Distribution of splited datasets")
        plt.xlabel("Participant")
        plt.ylabel("Number of samples")
        plt.xticks(range(num_clients), [f" {i}" for i in range(num_clients)])
        plt.legend(loc="upper right")
        plt.tight_layout()

        path_to_save = f"{self.config.participant['tracking_args']['log_dir']}/{self.config.participant['scenario_args']['name']}/all_data_distribution_{'iid' if self.iid else 'non_iid'}{'_' + self.partition if not self.iid else ''}.png"
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()
