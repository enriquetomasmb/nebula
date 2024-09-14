import copy
import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.prototeachernebulamodel import ProtoTeacherNebulaModel
from nebula.core.optimizations.communications.KD.utils.AT import Attention


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ProtoTeacherCIFAR100ModelResNet32(ProtoTeacherNebulaModel):
    def __init__(
        self,
        input_channels=3,
        num_classes=100,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        alpha_kd=1,
        beta_feat=1,
        lambda_proto=1,
        weighting=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            alpha_kd,
            beta_feat,
            lambda_proto,
            weighting,
        )
        self.embedding_dim = 512
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Construcción directa de ResNet-32
        self.layer1 = self._make_layer(BasicBlock, 64, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(256 * BasicBlock.expansion, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        conv1 = x

        x = self.layer1(x)  # 32x32
        conv2 = x
        x = self.layer2(x)  # 16x16
        conv3 = x
        x = self.layer3(x)  # 8x8
        conv4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4],
                )
            return logits, dense, [conv1, conv2, conv3, conv4]

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

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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

    def step(self, batch, batch_idx, phase):

        images, labels_g = batch
        images, labels = images.to(self.device), labels_g.to(self.device)
        logits, protos = self.forward_train(images, softmax=False)

        # Compute loss cross entropy loss
        loss_cls = self.criterion_cls(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss_protos = 0 * loss_cls
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1
            # Compute the loss for the prototypes
            loss_protos = self.criterion_mse(proto_new, protos)

        # Combine the losses
        loss = loss_cls + self.weighting.get_beta(loss_cls) * loss_protos
        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Aggregate the prototypes
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer


class MDProtoTeacherCIFAR100ModelResNet32(ProtoTeacherNebulaModel):
    def __init__(
        self,
        input_channels=3,
        num_classes=100,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        alpha_kd=1,
        beta_feat=100,
        lambda_proto=1,
        weighting=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            T,
            alpha_kd,
            beta_feat,
            lambda_proto,
            weighting,
        )
        self.embedding_dim = 512
        self.p = 2
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_feat = Attention(self.p)
        self.criterion_kd = DistillKL(self.T)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.student_model = None

        # Construcción directa de ResNet-32
        self.layer1 = self._make_layer(BasicBlock, 64, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(256 * BasicBlock.expansion, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, self.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        conv1 = x

        x = self.layer1(x)  # 32x32
        conv2 = x
        x = self.layer2(x)  # 16x16
        conv3 = x
        x = self.layer3(x)  # 8x8
        conv4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4],
                )
            return logits, dense, [conv1, conv2, conv3, conv4]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass para la inferencia del modelo"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Obtener las características intermedias
        logits, dense, features = self.forward_train(x, is_feat=True)

        # Calcular distancias a los prototipos globales
        distances = []
        for key, proto in self.global_protos.items():
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        return distances.argmin(dim=1)

    def configure_optimizers(self):
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
        logits, protos, feat_t = self.forward_train(images, is_feat=True, softmax=False)

        # Compute loss cross entropy loss
        loss_cls = self.criterion_cls(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss_protos = 0 * loss_cls
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1

            # Compute the loss for the prototypes
            loss_protos = self.criterion_mse(proto_new, protos)

        if self.student_model is not None:
            with torch.no_grad():
                student_logits, student_protos, feat_s = self.student_model.forward_train(images, is_feat=True)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            # Compute the mutual distillation loss
            loss_kd = self.criterion_kd(student_logits, logits)
            loss_group = self.criterion_feat(g_t, g_s)
            loss_feat = sum(loss_group)
        else:
            loss_feat = 0 * loss_cls
            loss_kd = 0 * loss_cls

        # Combine the losses
        loss = (
            loss_cls
            + self.weighting.get_alpha(loss_cls) * loss_kd
            + self.weighting.get_beta(loss_cls, loss_kd) * loss_feat
            + self.weighting.get_lambda(loss_cls, loss_kd, loss_feat) * loss_protos
        )
        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Aggregate the prototypes
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss

    def set_student_model(self, student_model):
        """
        Para evitar problemas de dependencia cíclica, se crea una copia del modelo de estudiante y se elimina el atributo teacher_model.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, "teacher_model"):
            del self.student_model.teacher_model
