import logging
import os
import pickle
from collections import OrderedDict
import random
import traceback
import hashlib
import numpy as np
import io
import gzip
import torch

from nebula.core.training.lightning import Lightning

class ProtoLightning(Lightning):
    """
    Learner with PyTorch Lightning.

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
        except:
            raise Exception("Error serializing model")

    def deserialize_model(self, data):
        # From https://pytorch.org/docs/stable/notes/serialization.html
        try:
            buffer = io.BytesIO(data)
            with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                params_dict = torch.load(f, map_location="cpu")
            return OrderedDict(params_dict)
        except:
            raise Exception("Error decoding parameters")

    def set_model_parameters(self, params, initialize=False):
        if hasattr(self.model, 'set_protos'):
            self.model.set_protos(params)
            return

        try:
            self.model.load_state_dict(params)
        except:
            raise Exception("Error setting parameters")

    def get_model_parameters(self, bytes=False):

        if bytes:
            if hasattr(self.model, 'get_protos'):
                return self.serialize_model(self.model.get_protos())
            return self.serialize_model(self.model.state_dict())
        else:
            if hasattr(self.model, 'get_protos'):
                return self.model.get_protos()
            return self.model.state_dict()