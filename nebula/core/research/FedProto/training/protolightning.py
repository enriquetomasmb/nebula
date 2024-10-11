import logging

from nebula.core.training.lightning import Lightning
from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)


class ParameterProtosSettingError(Exception):
    """Custom exception for errors setting model parameters."""


class ProtoLightning(Lightning):
    """
    Learner with PyTorch Lightning.

    """

    def __init__(self, model, data, config=None):
        super().__init__(model, data, config)

    def set_model_parameters(self, params, initialize=False):
        if initialize:
            try:
                self.model.load_state_dict(params)
                if hasattr(self.model, "set_protos"):
                    self.model.set_protos(dict())
                return None
            except Exception as e:
                raise ParameterProtosSettingError("Error setting parameters") from e

        if hasattr(self.model, "set_protos"):
            self.model.set_protos(params)
        else:
            logging_training.error("[ProtoLightning] (set_model_parameters) Error setting parameters")
        return None

    def get_model_parameters(self, bytes=False, initialize=False):
        if initialize:
            if bytes:
                return self.serialize_model(self.model.state_dict())
            return self.model.state_dict()

        if hasattr(self.model, "get_protos"):
            if bytes:
                return self.serialize_model(self.model.get_protos())
            return self.model.get_protos()

        logging_training.error("[ProtoLightning] (get_model_parameters) Error getting parameters")
        return None
