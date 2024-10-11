import logging

from nebula.core.training.lightning import Lightning
from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)


class ParameterProtoAndModelSettingError(Exception):
    """Custom exception for errors setting model parameters."""


class ProtoQuantizationLightning(Lightning):
    """
    Learner with PyTorch Lightning. Implements quantization of model parameters.

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
            # logging.info(f"Initializing model with parameters: {params.keys()}")
            self.model.load_state_dict(params)
            return

        # Convert parameters back to float32
        logging_training.info("[Learner] Decoding parameters...")
        for key, value in params.items():
            if key != "protos":
                params[key] = value.float()
        # Imprimimos la key de los parametros para debug
        # logging.info("[Learner] Keys of parameters: {}".format(params.keys()))

        try:
            self.model.load_state_dict(params)
        except Exception as e:
            raise ParameterProtoAndModelSettingError("Error setting parameters") from e

    def get_model_parameters(self, bytes=False, initialize=False):
        if initialize:
            # logging.info("[Learner] Getting parameters to initialize model...")
            # model = self.model.state_dict()
            # logging.info("Keys: {}".format(list(model.keys())))
            if bytes:
                return self.serialize_model(self.model.state_dict())
            return self.model.state_dict()

        model = self.model.state_dict()
        # Convert parameters to float16 before saving to reduce data size
        logging_training.info("[Learner] Encoding parameters...")
        # print keys for debug
        # logging.info("[Learner] Keys of parameters: {}".format(model.keys()))
        # quantize parameters to half precision, only if the key is not 'protos'
        for key, value in model.items():
            if key != "protos":
                model[key] = value.half()

        if bytes:
            return self.serialize_model(model)

        return model
