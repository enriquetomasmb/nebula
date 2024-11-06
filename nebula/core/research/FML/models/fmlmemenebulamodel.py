from abc import ABC

from nebula.core.models.nebulamodel import NebulaModel


class FMLMemeNebulaModel(NebulaModel, ABC):

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
    ):

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)
        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        self.beta = beta
        self.T = T
