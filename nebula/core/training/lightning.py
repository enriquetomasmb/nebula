import gc
import logging
from collections import OrderedDict
import random
import traceback
import hashlib
import io
import gzip
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
import copy
from torch.nn import functional as F
from nebula.core.utils.deterministic import enable_deterministic


class Lightning:
    DEFAULT_MODEL_WEIGHT = 1
    BYPASS_MODEL_WEIGHT = 0

    def __init__(self, model, data, config=None, logger=None):
        # self.model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.data = data
        self.config = config
        self._logger = logger
        self._trainer = None
        self.epochs = 1
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        self.round = 0
        enable_deterministic(self.config)

    @property
    def logger(self):
        return self._logger

    def get_round(self):
        return self.round

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data

    def create_trainer(self):
        num_gpus = torch.cuda.device_count()
        if self.config.participant["device_args"]["accelerator"] == "gpu" and num_gpus > 0:
            gpu_index = self.config.participant["device_args"]["idx"] % num_gpus
            logging.info("Creating trainer with accelerator GPU ({})".format(gpu_index))
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

    def validate_neighbour_model(self, neighbour_model_param):
        avg_loss = 0
        running_loss = 0
        bootstrap_dataloader = self.data.bootstrap_dataloader()
        num_samples = 0
        neighbour_model = copy.deepcopy(self.model)
        neighbour_model.load_state_dict(neighbour_model_param)

        # enable evaluation mode, prevent memory leaks.
        # no need to switch back to training since model is not further used.
        if torch.cuda.is_available():
            neighbour_model = neighbour_model.to("cuda")
        neighbour_model.eval()

        # bootstrap_dataloader = bootstrap_dataloader.to('cuda')
        with torch.no_grad():
            for inputs, labels in bootstrap_dataloader:
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")
                outputs = neighbour_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                running_loss += loss.item()
                num_samples += inputs.size(0)

        avg_loss = running_loss / len(bootstrap_dataloader)
        logging.info("Computed neighbor loss over {} data samples".format(num_samples))
        return avg_loss

    def get_hash_model(self):
        """
        Returns:
            str: SHA256 hash of model parameters
        """
        return hashlib.sha256(self.serialize_model()).hexdigest()

    def set_epochs(self, epochs):
        self.epochs = epochs

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
        try:
            self.model.load_state_dict(params)
        except:
            raise Exception("Error setting parameters")

    def get_model_parameters(self, bytes=False, initialize=False):
        if bytes:
            return self.serialize_model(self.model.state_dict())
        else:
            return self.model.state_dict()

    def train(self):
        try:
            self.create_trainer()
            self._trainer.fit(self.model, self.data)
        except Exception as e:
            logging.error(f"Error training model: {e}")
            logging.error(traceback.format_exc())

    def test(self):
        try:
            self.create_trainer()
            self._trainer.test(self.model, self.data, verbose=True)
        except Exception as e:
            logging.error(f"Error testing model: {e}")
            logging.error(traceback.format_exc())

    def get_model_weight(self):
        return len(self.data.train_dataloader().dataset)

    def on_round_start(self):
        self._logger.log_data({"Round": self.round})
        # self.reporter.enqueue_data("Round", self.round)
        pass

    def on_round_end(self):
        self._logger.global_step = self._logger.global_step + self._logger.local_step
        self._logger.local_step = 0
        self.round += 1
        logging.info("Flushing memory cache at the end of round...")
        torch.cuda.empty_cache()
        gc.collect()
        pass

    def on_learning_cycle_end(self):
        self._logger.log_data({"Round": self.round})
        # self.reporter.enqueue_data("Round", self.round)
        pass
