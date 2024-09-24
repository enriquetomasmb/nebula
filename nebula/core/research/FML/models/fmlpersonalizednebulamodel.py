from abc import ABC

from nebula.core.models.nebulamodel import NebulaModel


class FMLPersonalizedNebulaModel(NebulaModel, ABC):

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        alpha=1,
    ):

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)
        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        self.alpha = alpha
        self.T = T
