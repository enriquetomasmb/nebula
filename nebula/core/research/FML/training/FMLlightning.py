import logging
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
import torch
from nebula.core.training.lightning import Lightning, ParameterSettingError


class FMLLightning(Lightning):
    """
    Learner with PyTorch Lightning. Implements lightning control module for FML implementation.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None, logger=None):
        super().__init__(model, data, config, logger)

    def set_model_parameters(self, params, initialize=False):
        if initialize:
            try:
                self.model.load_state_dict(params)
                return None
            except Exception as e:
                raise ParameterSettingError("Error setting parameters") from e

        if hasattr(self.model, "model_meme") and self.model.model_meme is not None:
            try:
                self.model.model_meme.load_state_dict(params)
            except Exception as e:
                raise ParameterSettingError("Error setting parameters") from e
        else:
            logging.error("[FMLLightning] (set_model_parameters) Personalized model does not have meme model.")
        return None

    def get_model_parameters(self, bytes=False, initialize=False):
        if initialize:
            if bytes:
                return self.serialize_model(self.model.state_dict())
            return self.model.state_dict()

        if hasattr(self.model, "model_meme") and self.model.model_meme is not None:
            if bytes:
                return self.serialize_model(self.model.model_meme.state_dict())
            return self.model.model_meme.state_dict()

        logging.error("[FMLLightning] (get_model_parameters) Personalized model does not have meme model.")
        return None

    def create_trainer(self):
        num_gpus = torch.cuda.device_count()
        if self.config.participant["device_args"]["accelerator"] == "gpu" and num_gpus > 0:
            gpu_index = self.config.participant["device_args"]["idx"] % num_gpus
            logging.info(f"Creating trainer with accelerator GPU ({gpu_index}")
            self._trainer = Trainer(
                callbacks=[RichModelSummary(max_depth=1), LearningRateMonitor(logging_interval="epoch")],
                max_epochs=self.epochs,
                accelerator=self.config.participant["device_args"]["accelerator"],
                devices=[gpu_index],
                logger=self._logger,
                enable_checkpointing=False,
                enable_model_summary=False,
                # deterministic=True
            )
        else:
            logging.info("Creating trainer with accelerator CPU")
            self._trainer = Trainer(
                callbacks=[RichModelSummary(max_depth=1), LearningRateMonitor(logging_interval="epoch")],
                max_epochs=self.epochs,
                accelerator=self.config.participant["device_args"]["accelerator"],
                devices="auto",
                logger=self._logger,
                enable_checkpointing=False,
                enable_model_summary=False,
                # deterministic=True
            )
        logging.info(f"Trainer strategy: {self._trainer.strategy}")
