import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoStudentResnet import ProtoStudentResNet
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoTeacherResnet14 import (
    MDProtoTeacherCIFAR10ModelResnet14,
    ProtoTeacherCIFAR10ModelResnet14,
)


class ProtoStudentCIFAR10ModelResnet8(ProtoStudentResNet):
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
        teacher_model=None,
        T=2,
        beta_kd=1,
        beta_proto=1,
        mutual_distilation="KD",
        teacher_beta=100,
        send_logic=None,
        depth=8,
        num_filters=[16, 16, 32, 64],
        block_name="BasicBlock",
    ):
        if teacher_model is None:
            if mutual_distilation is not None and mutual_distilation == "MD":
                teacher_model = MDProtoTeacherCIFAR10ModelResnet14(beta=teacher_beta)
            elif mutual_distilation is not None and mutual_distilation == "KD":
                teacher_model = ProtoTeacherCIFAR10ModelResnet14()

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            teacher_model,
            T,
            mutual_distilation,
            send_logic,
            depth,
            num_filters,
            block_name,
        )

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.beta_proto = beta_proto
        self.beta_kd = beta_kd
        self.criterion_nll = nn.NLLLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(self.T)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        conv1 = x

        x, f1_pre = self.layer1(x)  # 32x32
        conv2 = x
        x, f2_pre = self.layer2(x)  # 16x16
        conv3 = x
        x, f3_pre = self.layer3(x)  # 8x8
        conv4 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        conv5 = x
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4, conv5],
                )
            return logits, dense, [conv1, conv2, conv3, conv4, conv5]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        conv1 = x

        x, f1_pre = self.layer1(x)  # 32x32
        conv2 = x
        x, f2_pre = self.layer2(x)  # 16x16
        conv3 = x
        x, f3_pre = self.layer3(x)  # 8x8
        conv4 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        conv5 = x
        dense = self.fc_dense(x)

        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        # Return the predicted class based on the closest prototype
        return distances.argmin(dim=1)

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def step(self, batch, batch_idx, phase):

        images, labels_g = batch
        images, labels = images.to(self.device), labels_g.to(self.device)
        logits, protos = self.forward_train(images, softmax=False)

        logits_softmax = F.log_softmax(logits, dim=1)
        protos_copy = protos.clone()
        with torch.no_grad():
            teacher_logits, teacher_protos = self.teacher_model.forward_train(images, softmax=False)

        # Compute loss 1
        loss1 = self.criterion_nll(logits_softmax, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1
            # Compute the loss for the prototypes
            loss2 = self.criterion_mse(proto_new, protos)

        # Compute loss knowledge distillation
        loss3 = self.criterion_div(logits, teacher_logits)

        # Compute loss knowledge distillation for the prototypes
        loss4 = self.criterion_mse(protos_copy, teacher_protos)

        # Combine the losses
        loss = loss1 + self.beta_proto * loss2 + self.beta_kd * (0.5 * loss3 + 0.5 * loss4)

        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Update the prototypes
            self.model_updated_flag2 = True
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss
