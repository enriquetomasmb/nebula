import logging

import torch


class MembershipInferenceAttack:
    """
        Base Class for conducting Membership Inference Attacks on a given model and dataset.

        Attributes:
            model (torch.nn.Module): The model to be attacked.
            global_dataset (Dataset): The global dataset used (e.g., MNIST, FMNIST, CIFAR10).
            in_eval (DataLoader): DataLoader for in-sample evaluation.
            out_eval (DataLoader): DataLoader for out-sample evaluation.
            device (torch.device): The device used for computation (CPU or GPU).
            in_eval_pre (tuple): Prediction scores and labels for in-sample evaluation.
            out_eval_pre (tuple): Prediction scores and labels for out-sample evaluation.
            index_mapping (dict): Mapping of indices to decompose in-samples for each node.

        Methods:
            _compute_predictions(model, dataloader):
                Computes the predictions and labels for a given model and dataloader.
            execute_attack():
                Placeholder method to be overridden by specific attack implementations.
            evaluate_metrics(true_p, false_p):
                Evaluates and returns precision, recall, false positive rate, and F1 score.
            evaluate_tp_for_each_node(in_predictions):
                Records and returns the number of true positives for each node's in-sample evaluation group.
    """

    def __init__(self, model, global_dataset, in_eval, out_eval, indexing_map):
        """
           Initializes the MembershipInferenceAttack class with the given model, dataset, and evaluation data.

           Args:
               model (torch.nn.Module): The model to be attacked.
               global_dataset (Dataset): The global dataset used (e.g., MNIST, FMNIST, CIFAR10).
               in_eval (DataLoader): DataLoader for in-sample evaluation.
               out_eval (DataLoader): DataLoader for out-sample evaluation.
               indexing_map (dict): Mapping of indices to decompose in-samples for each node.
        """
        import logging
        logging.info("[MIA] Initializing Membership Inference Attack")
        self.model = model
        self.global_dataset = global_dataset
        self.in_eval = in_eval
        self.out_eval = out_eval

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.in_eval_pre = self._compute_predictions(self.model, self.in_eval)
        self.out_eval_pre = self._compute_predictions(self.model, self.out_eval)

        self.index_mapping = indexing_map

    def _compute_predictions(self, model, dataloader):
        """
            Computes the predictions and labels for a given model and dataloader.

            Args:
                model (torch.nn.Module): The model to be used for predictions.
                dataloader (DataLoader): The dataloader providing input data and labels.

            Returns:
                tuple: A tuple containing tensors of predictions and labels.
        """
        model.eval()
        predictions = []
        labels = []
        logging.info(model)
        logging.info(dataloader)
        logging.info(len(dataloader))
        for inputs, label in dataloader:
            logging.info("INPUTS")
            logging.info(inputs)
            logging.info("LABEL")
            logging.info(label)

        with torch.no_grad():
            for inputs, label in dataloader:
                inputs = inputs.to(self.device)
                label = label.to(self.device)

                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)

                predictions.append(probs)
                labels.append(label)

            predictions = torch.cat(predictions, dim=0)  # it has become one tensor
            labels = torch.cat(labels, dim=0)  # it has become one tensor

        return predictions, labels

    def execute_attack(self):
        """
            Placeholder method to be overridden by specific attack implementations.

            Raises:
                NotImplementedError: If the method is not overridden.
        """
        raise NotImplementedError("Must override execute_attack")

    def evaluate_metrics(self, true_p, false_p):
        """
           Evaluates and returns precision, recall, false positive rate, and F1 score.

           Args:
               true_p (int): Number of true positives.
               false_p (int): Number of false positives.

           Returns:
               tuple: A tuple containing precision, recall, false positive rate, and F1 score.
        """
        size = len(self.in_eval_pre[0])

        total_positives = true_p + false_p

        precision = true_p / total_positives if total_positives > 0 else 0
        recall = true_p / size
        fpr = false_p / size
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def evaluate_tp_for_each_node(self, in_predictions):
        """
            Records and returns the number of true positives for each node's in-sample evaluation group.

            Args:
                in_predictions (tensor): A tensor containing 0 and 1 representing the boolean result of
                                         each data sample regarded as in or out sample.

            Returns:
                dict: A dictionary where keys are node ids and values are counts of true positives for each node.
        """
        nodes_tp_dict = {}

        for key, index in self.index_mapping.items():
            node_tp = in_predictions[index].sum().item()
            nodes_tp_dict[key] = node_tp

        return nodes_tp_dict


