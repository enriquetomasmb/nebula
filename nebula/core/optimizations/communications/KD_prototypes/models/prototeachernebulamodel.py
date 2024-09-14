import logging
from abc import ABC
from nebula.core.optimizations.adaptative_weighted.weighting import Weighting
from nebula.core.optimizations.communications.KD.models.teachernebulamodel import TeacherNebulaModel
from nebula.core.optimizations.adaptative_weighted.adaptativeweighting import AdaptiveWeighting


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
        alpha_kd=1,
        beta_feat=1,
        lambda_proto=1,
        weighting=None,
    ):

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, T)

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
            return {k: v.cpu() for k, v in self.global_protos.items()}

        proto = dict()
        for label, proto_info in self.agg_protos_label.items():

            if proto_info["count"] > 1:
                proto[label] = (proto_info["sum"] / proto_info["count"]).to("cpu")
            else:
                proto[label] = proto_info["sum"].to("cpu")

        logging.info(f"[ProtoFashionMNISTModelCNN.get_protos] Protos: {proto}")
        return proto

    def set_protos(self, protos):
        """
        Set the protos for the model.
        """
        self.agg_protos_label = dict()
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}
