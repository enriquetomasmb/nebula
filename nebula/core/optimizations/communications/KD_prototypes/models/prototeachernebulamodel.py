import logging
import copy
from abc import ABC
import torch
from nebula.core.optimizations.adaptative_weighted.weighting import Weighting
from nebula.core.optimizations.communications.KD.models.teachernebulamodel import TeacherNebulaModel
from nebula.core.optimizations.adaptative_weighted.adaptativeweighting import AdaptiveWeighting
from nebula.core.optimizations.communications.KD_prototypes.utils.GlobalPrototypeDistillationLoss import \
    GlobalPrototypeDistillationLoss


class ProtoTeacherNebulaModel(TeacherNebulaModel, ABC):

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        alpha_kd=0.5,
        beta_feat=0.3,
        lambda_proto=0.05,
        weighting=None,
    ):

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, T)
        self.automatic_optimization = False
        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        if weighting == "adaptative":
            self.weighting = AdaptiveWeighting(min_val=1, max_val=10)
        else:
            self.weighting = Weighting(alpha_value=alpha_kd, beta_value=beta_feat, lambda_value=lambda_proto)
        self.global_protos = dict()
        self.agg_protos_label = dict()

    def get_protos(self):
        """
        Get the protos for the model.
        """

        if len(self.agg_protos_label) == 0:
            return {k: v.detach() for k, v in self.global_protos.items()}

        proto = dict()
        for label, proto_info in self.agg_protos_label.items():

            if proto_info["count"] > 1:
                proto[label] = (proto_info["sum"] / proto_info["count"]).detach()
            else:
                proto[label] = proto_info["sum"].detach()

        return proto

    def set_protos(self, protos):
        """
        Set the protos for the model.
        """
        self.agg_protos_label = dict()
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}

    def step(self, batch, batch_idx, phase):

        images, labels_g = batch

        if phase == "Train":
            optimizer = self.optimizers()

        images, labels = images.to(self.device), labels_g.to(self.device)
        logits, protos = self.forward_train(images, softmax=False)

        # Compute loss cross entropy loss
        loss_cls = self.criterion_cls(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss_protos = 0 * loss_cls
        else:
            loss_protos = self.criterion_gpd(self.global_protos, protos, labels)

        # Combine the losses
        loss = loss_cls + 0.05 * loss_protos

        if phase == "Train":
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

        self.process_metrics(phase, logits, labels, loss.detach())

        del loss_protos, loss_cls, logits, loss

        if phase == "Train":
            # Aggregate the prototypes
            for i in range(len(labels_g)):
                label = labels_g[i].detach()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]).detach(), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        del labels, labels_g, protos


class MDProtoTeacherNebulaModel(ProtoTeacherNebulaModel, ABC):
    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        p=2,
        alpha_kd=1,
        beta_feat=1,
        lambda_proto=1,
        weighting=None,
    ):

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, T, alpha_kd, beta_feat, lambda_proto, weighting)

        self.student_model = None
        self.p = p

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
                if label.detach() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.detach()].data
                i += 1

            # Compute the loss for the prototypes
            loss_protos = self.criterion_mse(proto_new, protos)

        if self.student_model is not None:
            with torch.no_grad():
                student_logits, student_protos, feat_s = self.student_model.forward_train(images, is_feat=True, softmax=False)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s
            g_t = feat_t
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
                label = labels_g[i].detach()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss

    def set_student_model(self, student_model):
        """
        For cyclic dependency problem, a copy of the student model is created, the teacher_model attribute is removed.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, "teacher_model"):
            del self.student_model.teacher_model
