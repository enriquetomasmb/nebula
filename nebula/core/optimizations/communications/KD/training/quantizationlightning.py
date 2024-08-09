import gzip
import io
import logging
import torch
from nebula.core.training.lightning import Lightning


class ParameterQuantizationSettingError(Exception):
    """Custom exception for errors setting model parameters."""


class SerializationError(Exception):
    """Custom exception for errors serializing model parameters."""


class QuantizationLightning(Lightning):
    """
    Learner with PyTorch Lightning. Implements quantization of model parameters.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None, logger=None):
        super().__init__(model, data, config, logger)

    def serialize_model(self, model):

        # From https://pytorch.org/docs/stable/notes/serialization.html
        try:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                torch.save(model, f)
            return buffer.getvalue()
        except Exception as e:
            raise SerializationError("Error serializing model") from e

    def set_model_parameters(self, params, initialize=False):
        if initialize:
            self.model.load_state_dict(params)
            return

        # Convert parameters back to float32
        logging.info("[Learner] Decoding parameters...")

        if hasattr(self.model, "set_protos"):
            self.model.set_protos(params)
            return
        try:
            self.model.load_state_dict(params)
        except Exception as e:
            raise ParameterQuantizationSettingError("Error setting parameters") from e

    def get_model_parameters(self, bytes=False, initialize=False):
        if initialize:
            if bytes:
                return self.serialize_model(self.model.state_dict())
            return self.model.state_dict()

        model = self.model.state_dict()
        # Convert parameters to float16 before saving to reduce data size
        if hasattr(self.model, "teacher_model") and self.model.teacher_model is not None:
            model = {k: v.half() for k, v in model.items()}

        if bytes:
            return self.serialize_model(model)
        return model
