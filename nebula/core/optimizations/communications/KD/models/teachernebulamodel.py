from abc import ABC

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from nebula.addons.functions import print_msg_box
from nebula.core.models.nebulamodel import NebulaModel


class TeacherNebulaModel(NebulaModel, ABC):

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}

    def process_metrics(self, phase, y_pred, y, loss=None):
        """
        Calculate and log metrics for the given phase.
        The metrics are calculated in each batch.
        Args:
            phase (str): One of 'Train', 'Validation', or 'Test'
            y_pred (torch.Tensor): Model predictions
            y (torch.Tensor): Ground truth labels
            loss (torch.Tensor, optional): Loss value
        """

        y_pred_classes = torch.argmax(y_pred, dim=1)
        if phase == "Train":
            # self.log(name=f"{phase}/Loss", value=loss, add_dataloader_idx=False)
            self.logger.log_data({f"Teacher/{phase}/Loss": loss.item()}, step=self.global_step)
            self.train_metrics.update(y_pred_classes, y)
        elif phase == "Validation":
            self.val_metrics.update(y_pred_classes, y)
        elif phase == "Test (Local)":
            self.test_metrics.update(y_pred_classes, y)
            if self.cm is not None:
                self.cm.update(y_pred_classes, y)
        elif phase == "Test (Global)":
            self.test_metrics_global.update(y_pred_classes, y)
            if self.cm is not None:
                self.cm.update(y_pred_classes, y)
        else:
            raise NotImplementedError

    def log_metrics_end(self, phase):
        """
        Log metrics for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', 'Test (Local)', or 'Test (Global)'
        """
        if phase == "Train":
            output = self.train_metrics.compute()
        elif phase == "Validation":
            output = self.val_metrics.compute()
        elif phase == "Test (Local)":
            output = self.test_metrics.compute()
        elif phase == "Test (Global)":
            output = self.test_metrics_global.compute()
        else:
            raise NotImplementedError

        output = {f"Teacher/{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}

        self.logger.log_data(output, step=self.global_number[phase])

        metrics_str = ""
        for key, value in output.items():
            metrics_str += f"{key}: {value:.4f}\n"
        print_msg_box(
            metrics_str,
            indent=2,
            title=f"Teacher/{phase} Metrics | Step: {self.global_number[phase]}",
        )

    def generate_confusion_matrix(self, phase, print_cm=False, plot_cm=False):
        """
        Generate and plot the confusion matrix for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', 'Test (Local)', or 'Test (Global)'
            print_cm (bool): Print confusion matrix
            plot_cm (bool): Plot confusion matrix
        """
        if phase == "Test (Local)":
            if self.cm is None:
                raise ValueError(f"Confusion matrix not available for {phase} phase.")
            cm = self.cm.compute().cpu()
        elif phase == "Test (Global)":
            if self.cm_global is None:
                raise ValueError(f"Confusion matrix not available for {phase} phase.")
            cm = self.cm_global.compute().cpu()
        else:
            raise NotImplementedError

        if print_cm:
            print(f"\nTeacher/{phase}/ConfusionMatrix\n", cm)
        if plot_cm:
            cm_numpy = cm.numpy()
            cm_numpy = cm_numpy.astype(int)
            classes = list(range(self.num_classes))
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = plt.subplot()
            sns.heatmap(cm_numpy, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            ax.xaxis.set_ticklabels(classes, rotation=90)
            ax.yaxis.set_ticklabels(classes, rotation=0)
            self.logger.log_figure(fig, step=self.global_number[phase], name=f"Teacher/{phase}/CM")
            plt.close()
        if phase == "Test (Local)":
            self.cm.reset()
        else:
            self.cm_global.reset()
