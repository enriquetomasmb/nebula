
import torch
import copy

from nebula.core.optimizations.communications.KD.utils.AT import Attention

from nebula.core.optimizations.communications.KD.models.teachernebulamodel import TeacherNebulaModel


class TeacherMNISTModelCNN(TeacherNebulaModel):
    """
        Techer model for FashionMNIST.
    """
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
        self.example_input_array = torch.zeros(1, 1, 28, 28)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels, out_channels=64, kernel_size=(5, 5), padding="same"
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(5, 5), padding="same"
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 128, 4096)
        self.l2 = torch.nn.Linear(4096, num_classes)

    def forward(self, x, is_feat=False):
        """Forward pass of the model.
            is_feat: bool, if True return the features of the model.
        """
        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 128)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        if is_feat:
            return [conv1, conv2], logits
        else:
            return logits

    def configure_optimizers(self):
        """ Configure the optimizer for training. """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.process_metrics(phase, logits, labels, loss)
        return loss


class MDTeacherMNISTModelCNN(TeacherNebulaModel):
    """
        Techer model for FashionMNIST.
    """
    def __init__(
            self,
            input_channels=1,
            num_classes=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None,
            p=2,
            beta=1000,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)
        self.p = p
        self.beta = beta
        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.learning_rate = learning_rate

        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_kd = Attention(self.p)

        self.student_model = None

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels, out_channels=64, kernel_size=(5, 5), padding="same"
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(5, 5), padding="same"
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 128, 4096)
        self.l2 = torch.nn.Linear(4096, num_classes)

    def forward(self, x, is_feat=False):
        """Forward pass of the model.
            is_feat: bool, if True return the features of the model.
        """
        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 128)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        if is_feat:
            return [conv1, conv2], logits
        else:
            return logits

    def configure_optimizers(self):
        """ Configure the optimizer for training. """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
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

        # If the student model is not None, apply Mutual Distillation
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

