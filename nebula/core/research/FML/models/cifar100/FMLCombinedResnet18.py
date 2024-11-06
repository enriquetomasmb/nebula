import torch

from nebula.core.research.FML.models.cifar100.FMLMemeResnet18 import FMLCIFAR100MemeModelResNet18
from nebula.core.research.FML.models.cifar100.FMLPersonalizedResnet18 import FMLCIFAR100PersonalizedModelResNet18
from nebula.core.research.FML.models.fmlcombinedmodel import FMLCombinedNebulaModel


class FMLCIFAR100CombinedModelResNet18(FMLCombinedNebulaModel):
    def __init__(
        self,
        input_channels=3,
        num_classes=100,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=10,
        beta=0.2,  # 2006.16765v3 ("Dynamic alphabeta at different stage of training can improve both global and local performance")
        alpha=0.2,
        model_meme=None,
        model_local=None,
    ):

        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, T, beta, alpha, model_meme, model_local)

        if self.model_meme is None:
            self.model_meme = FMLCIFAR100MemeModelResNet18(
                input_channels,
                num_classes,
                learning_rate,
                metrics,
                confusion_matrix,
                seed,
                T,
                beta,
            )

        if self.model_local is None:
            self.model_local = FMLCIFAR100PersonalizedModelResNet18(
                input_channels,
                num_classes,
                learning_rate,
                metrics,
                confusion_matrix,
                seed,
                T,
                alpha,
            )

        self.example_input_array = torch.rand(1, 3, 32, 32)

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        opt_local = torch.optim.Adam(
            self.model_local.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )

        opt_meme = torch.optim.Adam(
            self.model_meme.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )

        return opt_local, opt_meme
