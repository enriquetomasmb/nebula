import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from nebula.core.models.knoledgeDistillationBaseNebulaModel.teacherfedstellarmodel import TeacherFedstellarModel
from nebula.core.knoledgeDistillation.AT import Attention

class TeacherCIFAR10ModelCNN(TeacherFedstellarModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
            self,
            input_channels=3,
            num_classes=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.config = {
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True
        }

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate
        self.criterion_cls = torch.torch.nn.CrossEntropyLoss()

        # Define layers of the model
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.4)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5)
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x, is_feat=False):
        """
        Forward pass of the model.
            is_feat: bool, if True return the features of the model.
        """
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        flattened = conv3.view(conv3.size(0), -1)  # Flatten the layer
        logits = self.fc_layer(flattened)

        if is_feat:
            return [conv1, conv2, conv3], logits
        else:
            return logits

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     betas=(self.config['beta1'], self.config['beta2']), amsgrad=self.config['amsgrad'])
        return optimizer

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        y_pred = self.forward(images)
        loss = self.criterion_cls(y_pred, labels)

        # Get metrics for each batch and log them
        self.process_metrics(phase, y_pred, labels, loss)

        return loss


class MDTeacherCIFAR10ModelCNN(TeacherFedstellarModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
            self,
            input_channels=3,
            num_classes=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None,
            p=2,
            beta=1000
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)
        self.p = p
        self.beta = beta
        self.config = {
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True
        }

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.torch.nn.CrossEntropyLoss()
        self.criterion_kd = Attention(self.p)
        self.student_model = None
        # Define layers of the model
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.3)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.4)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.5)
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x, is_feat=False):
        """
        Forward pass of the model.
            is_feat: bool, if True return the features of the model.
        """
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        flattened = conv3.view(conv3.size(0), -1)  # Flatten the layer
        logits = self.fc_layer(flattened)

        if is_feat:
            return [conv1, conv2, conv3], logits
        else:
            return logits

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     betas=(self.config['beta1'], self.config['beta2']), amsgrad=self.config['amsgrad'])
        return optimizer

    def set_student_model(self, student_model):
        """
        For cyclic dependency problem, a copy of the student model is created, the teacher_model attribute is removed.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, 'teacher_model'):
            del self.student_model.teacher_model

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        feat_t, logit_t = self(images, is_feat=True)
        loss_cls = self.criterion_cls(logit_t, labels)
        # If beta is greater than the limit, apply mutual distillation
        if self.student_model is not None:
            with torch.no_grad():
                feat_s, logit_s = self.student_model(images, is_feat=True)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s
            g_t = feat_t
            # Compute the knowledge distillation loss
            loss_group = self.criterion_kd(g_t, g_s)
            loss_kd = sum(loss_group)
        else:
            loss_kd = 0 * loss_cls
        # Combine the losses
        loss = loss_cls + self.beta * loss_kd
        self.process_metrics(phase, logit_t, labels, loss)
        return loss