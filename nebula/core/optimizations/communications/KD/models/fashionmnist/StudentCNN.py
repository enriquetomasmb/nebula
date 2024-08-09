import torch

from nebula.core.optimizations.communications.KD.models.fashionmnist.TeacherCNN import MDTeacherFashionMNISTModelCNN, TeacherFashionMNISTModelCNN
from nebula.core.optimizations.communications.KD.models.studentnebulamodelV2 import StudentNebulaModelV2
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL


class StudentFashionMNISTModelCNN(StudentNebulaModelV2):
    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        teacher_model=None,
        T=2,
        beta=1,
        decreasing_beta=False,
        limit_beta=0.1,
        mutual_distilation="KD",
        teacher_beta=100,
        send_logic=None,
    ):

        if teacher_model is None:
            if mutual_distilation is not None and mutual_distilation == "MD":
                teacher_model = MDTeacherFashionMNISTModelCNN(beta=teacher_beta)
            elif mutual_distilation is not None and mutual_distilation == "KD":
                teacher_model = TeacherFashionMNISTModelCNN()

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            teacher_model,
            T,
            beta,
            decreasing_beta,
            limit_beta,
            send_logic,
        )
        self.mutual_distilation = mutual_distilation
        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(self.T)

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding="same",
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same")
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 64, 2048)
        self.l2 = torch.nn.Linear(2048, num_classes)

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
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        if is_feat:
            return [conv1, conv2], logits
        return logits

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def step(self, batch, batch_idx, phase):
        if phase == "Train":
            self.model_updated_flag2 = True
        images, labels = batch
        student_logits = self(images)
        loss_ce = self.criterion_cls(student_logits, labels)
        # If beta is greater than the limit, apply knowledge distillation
        if self.beta > self.limit_beta:
            # Get the teacher logits
            with torch.no_grad():
                teacher_logits = self.teacher_model(images)
            # Compute the knowledge distillation loss
            loss_kd = self.criterion_div(student_logits, teacher_logits)
            # Combine the losses
            loss = loss_ce + self.beta * loss_kd
        else:
            loss = loss_ce

        self.process_metrics(phase, student_logits, labels, loss)
        return loss
