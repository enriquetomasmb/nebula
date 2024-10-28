import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.cluster import SpectralClustering
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from PIL import Image
from torch.autograd.functional import jacobian

from nebula.addons.attacks.mia.base_MIA import MembershipInferenceAttack
from nebula.core.datasets.cifar10.cifar10 import CIFAR10Dataset


class MetricBasedAttack(MembershipInferenceAttack):
    """
       Subclass for conducting Metric-Based Membership Inference Attacks.
    """
    def __init__(self, model, global_dataset, in_eval, out_eval, indexing_map, train_result, method_name):
        super().__init__(model, global_dataset, in_eval, out_eval, indexing_map)
        """
            Initializes the MetricBasedAttack class with the given parameters.
    
            Args:
                model (torch.nn.Module): The target model to be attacked.
                global_dataset (Dataset): The global dataset used (e.g., MNIST, FMNIST, CIFAR10).
                in_eval (list): DataLoader for in-sample evaluation.
                out_eval (DataLoader): DataLoader for out-sample evaluation.
                indexing_map (dict): Mapping of indices to decompose in-samples for each node.
                train_result (float): Training result used as a threshold for loss-based attack.
                method_name (str): The specific metric-based attack method to use.
        """

        self.train_result = train_result
        self.method_name = method_name

    def execute_all_attack(self):
        """
            Executes all the membership inference attacks defined in this class.
        """
        # the method to perform MIAs defined in this class
        for attr_name in dir(self):
            # Check if the attribute is a method and starts with "MIA"
            if attr_name.startswith("MIA") and callable(getattr(self, attr_name)):
                method = getattr(self, attr_name)
                method()

    def execute_specific_attack(self):
        """
            Executes a specific membership inference attack based on the method name.

            Returns:
                tuple: A tuple containing precision, recall, and F1 score of the attack.

            Raises:
                NotImplementedError: If the specified attack method is not implemented.
        """
        if self.method_name == "Prediction Correctness":
            return self.MIA_correctness_attack()
        elif self.method_name == "Prediction Loss":
            return self.MIA_loss_attack()
        elif self.method_name == "Prediction Maximal Confidence":
            return self.MIA_maximal_confidence_attack()
        elif self.method_name == "Prediction Entropy":
            return self.MIA_entropy_attack()
        elif self.method_name == "Prediction Sensitivity (Jacobian Matrix)":
            return self.MIA_sensitivity_attack()
        else:
            raise NotImplementedError("This kind of attack is still not implemented.")

    def MIA_correctness_attack(self):
        """
            Conducts a membership inference attack based on prediction correctness.
            The classifying rule is based whether the prediction of one data sample equals to
            its original label.

            Returns:
                tuple: A tuple containing precision, recall, and F1 score of the attack.
        """
        def correctness_check(dataset):
            predictions, labels = dataset
            _, predicted_labels = torch.max(predictions, dim=1)
            correct_predictions = predicted_labels == labels

            return correct_predictions

        in_predictions = correctness_check(self.in_eval_pre)
        out_predictions = correctness_check(self.out_eval_pre)

        true_positives = in_predictions.sum().item()
        false_positives = out_predictions.sum().item()

        print(true_positives)
        print(false_positives)

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives)

        # If you want to get a micro view of in evaluation group:
        # nodes_tp_dict = self.evaluate_tp_for_each_node(in_predictions)

        return precision, recall, f1

    def MIA_loss_attack(self):
        """
            Conducts a membership inference attack based on prediction loss.
            The classifying rule is based whether the loss of prediction of one data sample
            is larger than the training average loss of all training dataset.
        """
        loss_threshold = self.train_result

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.in_eval:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                losses = F.cross_entropy(logits, labels, reduction='none')
                in_predictions = losses < loss_threshold

            for inputs, labels in self.out_eval:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(inputs)
                losses = F.cross_entropy(logits, labels, reduction='none')
                out_predictions = losses < loss_threshold

        true_positives = in_predictions.sum().item()
        false_positives = out_predictions.sum().item()

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives)

        # If you want to get a micro view of in evaluation group:
        # nodes_tp_dict = self.evaluate_tp_for_each_node(in_predictions)

        return precision, recall, f1

    def _generate_random_images(self, batch_size):
        """
            Generates random images for threshold selection.

            Args:
                batch_size (int): The number of random images to generate.

            Returns:
                torch.Tensor: A tensor containing the generated random images.
        """
        images = []
        data_shape = self.global_dataset.train_set[0][0].shape

        if data_shape == (3, 32, 32):  # CIFAR-10 case
            height, width, channels = 32, 32, 3
            mean, std = [0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]

            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

            '''The following code is to simulate the case that we apply different data augmentation technique.'''
            '''if isinstance(self.global_dataset, CIFAR10DatasetNoAugmentation):
                transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
            elif isinstance(self.global_dataset, CIFAR10DatasetExtendedAugmentation):
                transform = T.Compose([
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
            else:'''
        else:  # gray scale images (FMNIST and MNIST)
            height, width, channels = 28, 28, 1
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])

        # Generate random images
        for _ in range(batch_size):
            data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            img = Image.fromarray(data.squeeze() if channels == 1 else data)
            images.append(img)

        # Apply transformations
        transformed_images = [transform(img) for img in images]

        return torch.stack(transformed_images)

    def _threshold_choosing(self, m_name):
        """
            Chooses the threshold for confidence or entropy-based attacks.

            Args:
                m_name (str): The metric name, either "confidence" or "entropy".

            Returns:
                list: A list of threshold percentiles.
        """
        random_images = self._generate_random_images(batch_size=len(self.out_eval_pre[0]))
        random_dataloader = DataLoader(TensorDataset(random_images), batch_size=128, shuffle=False, num_workers=0)

        threshold = []

        self.model.eval()
        with torch.no_grad():
            for batch in random_dataloader:
                inputs = batch[0].to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)

                if m_name == "confidence":
                    confidences, _ = torch.max(probs, dim=1)
                    threshold.append(confidences)
                else:
                    entropies = self._compute_entropy(probs)
                    threshold.append(entropies)

        threshold_tensor = torch.cat(threshold)

        sequence = list(range(10, 100, 10)) + [95]
        threshold_percentiles = [np.percentile(threshold_tensor.cpu().detach().numpy(), i) for i in sequence]

        return threshold_percentiles  # it contains 10 percentiles as the backup thresholds

    def MIA_maximal_confidence_attack(self):
        """
           Conducts a membership inference attack based on maximal confidence.

           This method evaluates the model's confidence in its predictions on in-sample and out-sample data,
           and selects the best threshold to maximize the F1 score.

           Returns:
               tuple: A tuple containing the best precision, recall, and F1 score of the attack.
        """
        threshold = self._threshold_choosing("confidence")

        def maximal_confidence_check(dataset):
            predictions, labels = dataset

            confidences, _ = torch.max(predictions, dim=1)

            return confidences

        best_f1 = 0
        final_precison = 0
        final_recall = 0

        in_confidences = maximal_confidence_check(self.in_eval_pre)
        out_confidences = maximal_confidence_check(self.out_eval_pre)

        # dict_p = {"precision": [], "recall": [], "f1": []}
        for i, thre in enumerate(threshold):
            in_predictions = in_confidences >= thre
            true_positives = in_predictions.sum().item()

            out_predictions = out_confidences >= thre
            false_positives = out_predictions.sum().item()

            precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives)
            # dict_p["precision"].append(precision)
            # dict_p["recall"].append(recall)
            # dict_p["f1"].append(f1)

            # Update the best threshold based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                final_precison = precision
                final_recall = recall

        return final_precison, final_recall, best_f1

    def _compute_entropy(self, probs):
        log_probs = torch.log(probs + 1e-6)  # Correctly use log on probabilities
        entropy = -(probs * log_probs).sum(dim=1)
        return entropy

    def MIA_entropy_attack(self):
        """
            Conducts a membership inference attack based on prediction entropy.

            This method evaluates the model's prediction entropy on in-sample and out-sample data,
            and selects the best threshold to maximize the F1 score.

            Returns:
                tuple: A tuple containing the best precision, recall, and F1 score of the attack.
        """
        threshold = self._threshold_choosing("entropy")

        def entropy_check(dataset):
            predictions, labels = dataset

            entropies = self._compute_entropy(predictions)

            return entropies

        best_f1 = 0
        final_precison = 0
        final_recall = 0

        in_entropies = entropy_check(self.in_eval_pre)
        out_entropies = entropy_check(self.out_eval_pre)

        # dict_p = {"precision": [], "recall": [], "f1": []}
        for i, thre in enumerate(threshold):
            in_predictions = in_entropies <= thre
            true_positives = in_predictions.sum().item()

            out_predictions = out_entropies <= thre
            false_positives = out_predictions.sum().item()

            precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives)
            # dict_p["precision"].append(precision)
            # dict_p["recall"].append(recall)
            # dict_p["f1"].append(f1)

            # Update the best threshold based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                final_precison = precision
                final_recall = recall

        return final_precison, final_recall, best_f1

    def _compute_jacobian_and_norm_white_box(self, inputs):
        """
            Computes the Jacobian matrix and its L2 norm for white-box sensitivity attack.
            It directly uses the jacobian calculation function from Pytorch to do this work on
            the assumption that the attacker knows the detailed information of model structure.

            Args:
                inputs (torch.Tensor): The input tensor for which the Jacobian is to be computed.

            Returns:
                float: The L2 norm of the Jacobian matrix.
        """
        inputs = inputs.to(self.device)
        inputs.requires_grad_(True)

        jacobian_matrix = jacobian(lambda x: self.model(x), inputs)

        jacobian_reshaped = jacobian_matrix.squeeze().reshape(inputs.size(1), -1)  # Reshape to 2D
        l2_norm = torch.norm(jacobian_reshaped, p=2)
        return l2_norm.item()

    def _compute_jacobian_and_norm_black_box(self, inputs, epsilon=1e-5):
        """
            Computes the Jacobian matrix and its L2 norm for black-box sensitivity attack using finite differences.
            This process approximates the jacobian matrix computation on the assumption that the attacker only
            has the access to the input and output of the model.

            Args:
                inputs (torch.Tensor): The input tensor for which the Jacobian is to be computed.
                epsilon (float): A small value for finite difference approximation.

            Returns:
                torch.Tensor: The L2 norm of the Jacobian matrix.
        """
        self.model.eval()
        inputs = inputs.clone().detach().requires_grad_(True).to(
            self.device)  # Ensure the inputs require gradients and move to device

        outputs = self.model(inputs)
        num_outputs = outputs.size(1)
        num_inputs = inputs.size(1)

        jacobian = torch.zeros(num_outputs, num_inputs).to(self.device)

        for i in range(num_inputs):
            inputs_pos = inputs.clone().detach()
            inputs_neg = inputs.clone().detach()

            inputs_pos[:, i] += epsilon
            inputs_neg[:, i] -= epsilon

            outputs_pos = self.model(inputs_pos)
            outputs_neg = self.model(inputs_neg)

            jacobian[:, i] = (outputs_pos - outputs_neg) / (2 * epsilon)

        l2_norm = torch.norm(jacobian, p=2)
        return l2_norm

    def MIA_sensitivity_attack(self):
        """
            Conducts a membership inference attack based on prediction sensitivity.

            This method evaluates the L2 norm of the Jacobian matrix for in-sample and out-sample data,
            and uses clustering to determine membership.

            Returns:
                tuple: A tuple containing precision, recall, and F1 score of the attack.
        """
        norms = []

        # Compute norms for in_eval_group
        for inputs, _ in self.in_eval:
            l2_norm = self._compute_jacobian_and_norm_black_box(inputs)
            norms.append(l2_norm.cpu().item())

        # Compute norms for out_eval_group
        for inputs, _ in self.out_eval:
            l2_norm = self._compute_jacobian_and_norm_black_box(inputs)
            norms.append(l2_norm.cpu().item())

        norm_array = np.array(norms)

        attack_cluster = SpectralClustering(n_clusters=6, n_jobs=-1, affinity='nearest_neighbors', n_neighbors=19)
        y_attack_pred = attack_cluster.fit_predict(norm_array.reshape(-1, 1))
        split = 1

        cluster_1 = np.where(y_attack_pred >= split)[0]
        cluster_0 = np.where(y_attack_pred < split)[0]

        y_attack_pred[cluster_1] = 1
        y_attack_pred[cluster_0] = 0
        cluster_1_mean_norm = norm_array[cluster_1].mean()
        cluster_0_mean_norm = norm_array[cluster_0].mean()
        if cluster_1_mean_norm > cluster_0_mean_norm:
            y_attack_pred = np.abs(y_attack_pred - 1)

        size = len(self.in_eval_pre[0])

        true_positives = np.sum(y_attack_pred[:size] == 1)
        false_positives = np.sum(y_attack_pred[size:] == 1)

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives)

        return precision, recall, f1
