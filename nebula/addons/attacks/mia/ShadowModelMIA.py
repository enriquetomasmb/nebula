import json
import os
import torch
from nebula.core.training.lightning import Lightning

from torch.utils.data import TensorDataset, DataLoader
from nebula.addons.attacks.mia.base_MIA import MembershipInferenceAttack
from nebula.addons.attacks.mia.AttackModel.SoftmaxMLPClassifier import SoftmaxMLPClassifier


class ShadowModelBasedAttack(MembershipInferenceAttack):
    """
    Subclass of MembershipInferenceAttack for conducting Shadow Model-Based Membership Inference Attacks.
    """
    def __init__(self, model, global_dataset, in_eval, out_eval, indexing_map, max_epochs, shadow_train, shadow_test, num_s, attack_model_type, trainer, trainerconfig, trainerlogger):
        """
           Initializes the ShadowModelBasedAttack class with the given parameters.

           Args:
               model (torch.nn.Module): The target model to be attacked.
               global_dataset (Dataset): The global dataset used (e.g., MNIST, FMNIST, CIFAR10).
               in_eval (list): DataLoader for in-sample evaluation.
               out_eval (DataLoader): DataLoader for out-sample evaluation.
               indexing_map (dict): Mapping of indices to decompose in-samples for each node.
               max_epochs (int): Maximum number of training epochs for the shadow models.
               shadow_train (list): List of dataloaders for shadow model training datasets.
               shadow_test (list): List of dataloaders for shadow model test datasets.
               num_s (int): Number of shadow models to be trained.
               attack_model_type (str): The type of model used for the attack (e.g., "Neural Network").
        """

        super().__init__(model, global_dataset, in_eval, out_eval, indexing_map)

        self.shadow_test_res = None
        self.shadow_train_res = None
        self.max_epochs = max_epochs
        self.num_shadow = num_s
        self.shadow_train = shadow_train  # A list containing different shadow model's train dataloaders
        self.shadow_test = shadow_test  # A list containing different shadow model's test dataloaders
        self.attack_model_type = attack_model_type
        self.trainer = trainer
        self.trainerconfig = trainerconfig
        self.trainerlogger = trainerlogger

        self._generate_attack_dataset()

    def _generate_attack_dataset(self):
        """
            Generates the attack dataset using shadow models by training multiple shadow models
            and collecting their predictions and labels.

            This method trains the specified number of shadow models using the provided shadow
            train and test dataloaders, and then collects their predictions and labels to create
            the attack dataset.
        """
        model_class = type(self.model)

        s_tr_pre = []
        s_tr_label = []
        s_te_pre = []
        s_te_label = []

        for i in range(self.num_shadow):
            shadow_model = model_class()

            #shadow_trainer = Trainer(max_epochs=self.max_epochs, accelerator="auto", devices="auto", logger=False, enable_checkpointing=False)
            shadow_trainer = self.trainer(shadow_model, self.shadow_train[i], config = self.trainerconfig, logger = self.trainerlogger)

            #shadow_trainer.fit(shadow_model, self.shadow_train[i])
            shadow_trainer.train()
            # shadow_train_result = shadow_trainer.callback_metrics
            # shadow_trainer.test(shadow_model, self.shadow_test[i])
            # shadow_test_result = shadow_trainer.callback_metrics

            '''shadow_merged_results = {**shadow_train_result, **shadow_test_result}
               shadow_res_dict = {key: value.item() if hasattr(value, 'item') else value for key, value in
                               shadow_merged_results.items()}'''

            tr_pre, tr_label = self._compute_predictions(shadow_model.to(self.device), self.shadow_train[i])
            te_pre, te_label = self._compute_predictions(shadow_model.to(self.device), self.shadow_test[i])

            s_tr_pre.append(tr_pre)
            s_tr_label.append(tr_label)

            s_te_pre.append(te_pre)
            s_te_label.append(te_label)

        self.shadow_train_res = (torch.cat(s_tr_pre, dim=0), torch.cat(s_tr_label, dim=0))
        self.shadow_test_res = (torch.cat(s_te_pre, dim=0), torch.cat(s_te_label, dim=0))

    def MIA_shadow_model_attack(self):
        """
           Conducts the Membership Inference Attack using the shadow model approach.

           This method uses the shadow model predictions to train an attack model, which is then used
           to infer membership of the original dataset samples.

           Returns:
               tuple: A tuple containing precision, recall, and F1 score of the attack.
        """

        shadow_train_pre = self.shadow_train_res[0]
        shadow_test_pre = self.shadow_test_res[0]

        in_labels = torch.ones(shadow_train_pre.shape[0], dtype=torch.long)
        out_labels = torch.zeros(shadow_test_pre.shape[0], dtype=torch.long)

        attack_dataset = TensorDataset(torch.cat((shadow_train_pre, shadow_test_pre), dim=0),
                                       torch.cat((in_labels, out_labels), dim=0))

        attack_dataloader = DataLoader(attack_dataset, batch_size=128, shuffle=True, num_workers=0)

        attack_model = None
        if self.attack_model_type == "Neural Network":
            attack_model = SoftmaxMLPClassifier(10, 64)
        else:
            pass  # Add other possible attack model type further

        #attack_trainer = Trainer(max_epochs=50, accelerator="auto", devices="auto", logger=False,
        #                         enable_checkpointing=False, enable_model_summary=False)
        #attack_trainer.fit(attack_model, attack_dataloader)
        attack_trainer = self.trainer(attack_model, attack_dataloader, config = self.trainerconfig, logger = self.trainerlogger)
        attack_trainer.train()

        def in_out_samples_check(model, dataset):
            predictions, _ = dataset
            dataloader = DataLoader(predictions, batch_size=128, shuffle=False, num_workers=0)

            predicted_label = []
            model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    logits = model(batch)
                    _, predicted = torch.max(logits, 1)  # max value, max value index

                    true_items = predicted == 1
                    predicted_label.append(true_items)

                predicted_label = torch.cat(predicted_label, dim=0)
            return predicted_label

        in_predictions = in_out_samples_check(attack_model.to(self.device), self.in_eval_pre)
        out_predictions = in_out_samples_check(attack_model.to(self.device), self.out_eval_pre)

        true_positives = in_predictions.sum().item()
        false_positives = out_predictions.sum().item()

        precision, recall, f1 = self.evaluate_metrics(true_positives, false_positives)

        return precision, recall, f1
