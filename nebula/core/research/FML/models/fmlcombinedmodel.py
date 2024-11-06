from abc import ABC
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassF1Score,
)
from torchmetrics import MetricCollection
import torch
from nebula.addons.functions import print_msg_box
from nebula.core.models.nebulamodel import NebulaModel
from nebula.core.research.FML.utils.KD import FMLDistillKL


class FMLCombinedNebulaModel(NebulaModel, ABC):

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        beta=0.2,
        alpha=0.2,
        model_meme=None,
        model_local=None,
    ):

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        if metrics is None:
            metrics = MetricCollection(
                [
                    MulticlassAccuracy(num_classes=num_classes),
                    MulticlassPrecision(num_classes=num_classes),
                    MulticlassRecall(num_classes=num_classes),
                    MulticlassF1Score(num_classes=num_classes),
                ]
            )
        self.train_metrics = metrics.clone(prefix="Train/")
        self.train_metrics_meme = metrics.clone(prefix="Train/Meme/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.val_metrics_meme = metrics.clone(prefix="Validation/Meme/")
        self.test_metrics = metrics.clone(prefix="Test (Local)/")
        self.test_metrics_meme = metrics.clone(prefix="Test (Local)/Meme/")
        self.test_metrics_global = metrics.clone(prefix="Test (Global)/")
        self.test_metrics_global_meme = metrics.clone(prefix="Test (Global)/Meme/")
        del metrics

        self.model_meme = model_meme
        self.model_local = model_local
        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        self.beta = beta
        self.alpha = alpha
        self.T = T
        self.automatic_optimization = False
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = FMLDistillKL(self.T)

    def process_metrics(self, phase, y_pred, y, loss=None, model_name="Local"):
        """
        Calculate and log metrics for the given phase.
        The metrics are calculated in each batch.
        Args:
            phase (str): One of 'Train', 'Validation', or 'Test'
            y_pred (torch.Tensor): Model predictions
            y (torch.Tensor): Ground truth labels
            loss (torch.Tensor, optional): Loss value
            model_name (str, optional): Name of the model. Defaults to 'Local'
        """

        y_pred_classes = torch.argmax(y_pred, dim=1)
        if phase == "Train":
            # self.log(name=f"{phase}/Loss", value=loss, add_dataloader_idx=False)
            self.logger.log_data({f"{phase}/{model_name}/Loss": loss.item()}, step=self.global_step)
            # Actualizar las métricas correspondientes
            if model_name == "Local":
                self.train_metrics.update(y_pred_classes, y)
            elif model_name == "Meme":
                self.train_metrics_meme.update(y_pred_classes, y)
        elif phase == "Validation":
            if model_name == "Local":
                self.val_metrics.update(y_pred_classes, y)
            elif model_name == "Meme":
                self.val_metrics_meme.update(y_pred_classes, y)
        elif phase == "Test (Local)":
            if model_name == "Local":
                self.test_metrics.update(y_pred_classes, y)
            elif model_name == "Meme":
                self.test_metrics_meme.update(y_pred_classes, y)
            self.cm.update(y_pred_classes, y) if self.cm is not None else None
        elif phase == "Test (Global)":
            if model_name == "Local":
                self.test_metrics_global.update(y_pred_classes, y)
            elif model_name == "Meme":
                self.test_metrics_global_meme.update(y_pred_classes, y)
            self.cm_global.update(y_pred_classes, y) if self.cm_global is not None else None
        else:
            raise NotImplementedError

        del y, y_pred_classes

    def log_metrics_end(self, phase):
        """
        Log metrics for both models at the end of the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', 'Test (Local)', or 'Test (Global)'
        """
        if phase == "Train":
            # Computar y registrar métricas del modelo personalizado
            output_local = self.train_metrics.compute()
            output_local = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_local.items()}
            self.logger.log_data(output_local, step=self.global_number[phase])

            # Computar y registrar métricas del modelo meme
            output_meme = self.train_metrics_meme.compute()
            output_meme = {f"{phase}/Meme/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_meme.items()}
            self.logger.log_data(output_meme, step=self.global_number[phase])

        elif phase == "Validation":
            # Computar y registrar métricas del modelo personalizado
            output_local = self.val_metrics.compute()
            output_local = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_local.items()}
            self.logger.log_data(output_local, step=self.global_number[phase])

            # Computar y registrar métricas del modelo meme
            output_meme = self.val_metrics_meme.compute()
            output_meme = {f"{phase}/Meme/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_meme.items()}
            self.logger.log_data(output_meme, step=self.global_number[phase])

        elif phase == "Test (Local)":
            # Computar y registrar métricas del modelo personalizado
            output_local = self.test_metrics.compute()
            output_local = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_local.items()}
            self.logger.log_data(output_local, step=self.global_number[phase])

            # Computar y registrar métricas del modelo meme
            output_meme = self.test_metrics_meme.compute()
            output_meme = {f"{phase}/Meme/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_meme.items()}
            self.logger.log_data(output_meme, step=self.global_number[phase])

        elif phase == "Test (Global)":
            # Computar y registrar métricas del modelo personalizado
            output_local = self.test_metrics_global.compute()
            output_local = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_local.items()}
            self.logger.log_data(output_local, step=self.global_number[phase])

            # Computar y registrar métricas del modelo meme
            output_meme = self.test_metrics_global_meme.compute()
            output_meme = {f"{phase}/Meme/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output_meme.items()}
            self.logger.log_data(output_meme, step=self.global_number[phase])

        else:
            raise NotImplementedError(f"Phase {phase} not implemented in log_metrics_end.")

        # Combinar y mostrar las métricas de ambos modelos
        metrics_str = ""
        combined_output = {**output_local, **output_meme}
        for key, value in combined_output.items():
            metrics_str += f"{key}: {value:.4f}\n"

        print_msg_box(metrics_str, indent=2, title=f"{phase} Metrics | Step: {self.global_number[phase]}")

        del output_local, output_meme

    def forward(self, x):
        return self.model_local(x)

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        if phase == "Train":
            optimizer_local, optimizer_meme = self.optimizers()

        output_local = self.model_local(images)
        output_meme = self.model_meme(images)

        # Pérdidas de clasificación
        loss_cls_local = self.criterion_cls(output_local, labels)
        loss_cls_meme = self.criterion_cls(output_meme, labels)

        # Pérdidas de distilación mutua
        loss_div_local = self.criterion_div(output_local, output_meme.detach())
        loss_div_meme = self.criterion_div(output_meme, output_local.detach())

        # Pérdidas totales
        loss_local = self.alpha * loss_cls_local + (1 - self.alpha) * loss_div_local
        loss_meme = self.beta * loss_cls_meme + (1 - self.beta) * loss_div_meme

        if phase == "Train":
            optimizer_local.zero_grad()
            self.manual_backward(loss_local, retain_graph=True)
            optimizer_local.step()

            optimizer_meme.zero_grad()
            self.manual_backward(loss_meme)
            optimizer_meme.step()

        self.process_metrics(phase, output_local, labels, loss_local.detach(), model_name="Local")
        self.process_metrics(phase, output_meme, labels, loss_meme.detach(), model_name="Meme")

        del loss_cls_local, loss_div_local, output_local
        del loss_cls_meme, loss_div_meme, output_meme
        del images, labels
