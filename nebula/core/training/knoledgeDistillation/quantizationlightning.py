import torch
import logging
import io
from collections import OrderedDict
import gzip

from nebula.core.training.lightning import Lightning

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

        # Convert parameters to float16 before saving to reduce data size
        logging.info("[Learner] Encoding parameters...")
        # print keys for debug
        logging.info("[Learner] Keys of parameters: {}".format(model.keys()))
        # quantize parameters to half precision
        if hasattr(self.model, "teacher_model") and self.model.teacher_model is not None:
            model = {k: v.half() for k, v in model.items()}
        # From https://pytorch.org/docs/stable/notes/serialization.html
        try:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
                torch.save(model, f)
            return buffer.getvalue()
        except:
            raise Exception("Error serializing model")

    def deserialize_model(self, data):
        # From https://pytorch.org/docs/stable/notes/serialization.html
        try:
            buffer = io.BytesIO(data)

            with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                params_dict = torch.load(f, map_location="cpu")
                # Convert parameters back to float32
                logging.info("[Learner] Decoding parameters...")
                params_dict = {k: v.float() for k, v in params_dict.items()}
                # Imprimimos la key de los parametros para debug
                logging.info("[Learner] Keys of parameters: {}".format(params_dict.keys()))
            return OrderedDict(params_dict)

        except:
            raise Exception("Error decoding parameters")
