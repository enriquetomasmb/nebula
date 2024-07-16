#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy("file_system")

import torch

from nebula.core.knoledgeDistillation.KD import DistillKL
from nebula.core.models.cifar10.knoledgeDistillation.TeacherCNN import TeacherCIFAR10ModelCNN, \
    MDTeacherCIFAR10ModelCNN
from nebula.core.models.knoledgeDistillationBaseNebulaModel.studentfedstellarmodelV2 import StudentFedstellarModelV2
from nebula.core.models.nebulamodel import NebulaModel

class StudentCIFAR10ModelCNN(StudentFedstellarModelV2):
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
            seed=None,teacher_model=None,
            T=2,
            beta=1,
            decreasing_beta=False,
            limit_beta=0.1,
            mutual_distilation="KD",
            teacher_beta=100,
            send_logic=None
    ):
        if teacher_model is None:
            if mutual_distilation is not None and mutual_distilation == "MD":
                    teacher_model = MDTeacherCIFAR10ModelCNN(beta=teacher_beta)
            elif mutual_distilation is not None and mutual_distilation == "KD":
                teacher_model = TeacherCIFAR10ModelCNN()

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, teacher_model, T, beta, decreasing_beta, limit_beta, send_logic)

        self.config = {
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True
        }
        
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.mutual_distilation = mutual_distilation
        self.criterion_cls = torch.torch.nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(self.T)

        # Define layers of the model
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25)
        )

        self.fc_layer = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, 512),
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.config['beta1'], self.config['beta2']), amsgrad=self.config['amsgrad'])
        return optimizer

    def step(self, batch, batch_idx, phase):
        if phase == "Train":
            self.model_updated_flag2 = True
        images, labels = batch
        y_pred = self.forward(images)
        loss_ce = self.criterion_cls(y_pred, labels)
        # If beta is greater than the limit, apply knowledge distillation
        if self.beta > self.limit_beta and self.teacher_model is not None:
            with torch.no_grad():
                teacher_logits = self.teacher_model(images)
            # Compute the knowledge distillation loss
            loss_kd = self.criterion_div(y_pred, teacher_logits)
            # Combine the losses
            loss = loss_ce + self.beta * loss_kd
        else:
            loss = loss_ce

        # Get metrics for each batch and log them
        self.process_metrics(phase, y_pred, labels, loss)

        return loss

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        if send_logic() == 0: only send the student model
        if send_logic() == 1: only send the fc_layer
        """
        if self.send_logic() == 0:
            original_state = super().state_dict(destination, prefix, keep_vars)
            # Filter out teacher model parameters
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith('teacher_model.')}
        elif self.send_logic() == 1:
            original_state = super().state_dict(destination, prefix, keep_vars)
            filtered_state = {k: v for k, v in original_state.items() if not k.startswith('teacher_model.')}
            filtered_state = {k: v for k, v in filtered_state.items() if k.startswith('fc_layer.')}

        return filtered_state