import logging
from nebula.core.training.lightning import Lightning, ParameterSettingError
from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)


class FMLLightning(Lightning):
    """
    Learner with PyTorch Lightning. Implements lightning control module for FML implementation.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None):
        super().__init__(model, data, config)

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
            logging_training.error("[FMLLightning] (set_model_parameters) Personalized model does not have meme model.")
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

        logging_training.error("[FMLLightning] (get_model_parameters) Personalized model does not have meme model.")
        return None
