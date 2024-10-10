import gc
import logging
from collections import OrderedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback
import hashlib
import io
import gzip
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ProgressBar, ModelSummary
import copy
from torch.nn import functional as F
from nebula.core.utils.deterministic import enable_deterministic
from nebula.config.config import TRAINING_LOGGER

logging_training = logging.getLogger(TRAINING_LOGGER)

class NebulaProgressBar(ProgressBar):
    """Nebula progress bar for training.
    Logs the percentage of completion of the training process using logging.
    """

    def __init__(self):
        super().__init__()
        self.enable = True

    def disable(self):
        """Disable the progress bar logging."""
        self.enable = False

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the training epoch starts."""
        super().on_train_epoch_start(trainer, pl_module)
        if self.enable:
            logging_training.info(f"Starting Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.enable:
            # Calculate percentage complete for the current epoch
            percent = ((batch_idx + 1) / self.total_train_batches) * 100  # +1 to count current batch
            logging_training.info(f"Epoch {trainer.current_epoch} - {percent:.01f}% complete")

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of the training epoch."""
        super().on_train_epoch_end(trainer, pl_module)
        if self.enable:
            logging_training.info(f"Epoch {trainer.current_epoch} finished")

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        if self.enable:
            logging_training.info(f"Starting validation for Epoch {trainer.current_epoch}")

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if self.enable:
            logging_training.info(f"Validation for Epoch {trainer.current_epoch} finished")

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)
        if not self.has_dataloader_changed(dataloader_idx):
            return

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called at the end of each test batch."""
        super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self.enable:
            total_batches = self.total_test_batches_current_dataloader
            if total_batches == 0:
                logging_training.warning(f"Total test batches is 0 for dataloader {dataloader_idx}, cannot compute progress.")
                return

            percent = ((batch_idx + 1) / total_batches) * 100  # +1 to count the current batch
            logging_training.info(f"Test Epoch {trainer.current_epoch}, Dataloader {dataloader_idx} - {percent:.01f}% complete")

    def on_test_epoch_start(self, trainer, pl_module):
        super().on_test_epoch_start(trainer, pl_module)
        if self.enable:
            logging_training.info(f"Starting testing for Epoch {trainer.current_epoch}")

    def on_test_epoch_end(self, trainer, pl_module):
        super().on_test_epoch_end(trainer, pl_module)
        if self.enable:
            logging_training.info(f"Testing for Epoch {trainer.current_epoch} finished")


class Lightning:
    DEFAULT_MODEL_WEIGHT = 1
    BYPASS_MODEL_WEIGHT = 0

    def __init__(self, model, data, config=None, logger=None):
        # self.model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.data = data
        self.config = config
        self._logger = logger
        self.__trainer = None
        self.epochs = 1
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
            logging_training.info("Creating trainer with accelerator GPU ({})".format(gpu_index))
            self.__trainer = Trainer(
                callbacks=[ModelSummary(max_depth=1), LearningRateMonitor(logging_interval="epoch"), NebulaProgressBar()],
                max_epochs=self.epochs,
                accelerator=self.config.participant["device_args"]["accelerator"],
                devices=[gpu_index],
                logger=self._logger,
                enable_checkpointing=False,
                enable_model_summary=False,
                # deterministic=True
            )
        else:
            logging_training.info("Creating trainer with accelerator CPU")
            self.__trainer = Trainer(
                callbacks=[ModelSummary(max_depth=1), LearningRateMonitor(logging_interval="epoch"), NebulaProgressBar()],
                max_epochs=self.epochs,
                accelerator=self.config.participant["device_args"]["accelerator"],
                devices="auto",
                logger=self._logger,
                enable_checkpointing=False,
                enable_model_summary=False,
                # deterministic=True
            )
        logging_training.info(f"Trainer strategy: {self.__trainer.strategy}")

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
        logging_training.info("Computed neighbor loss over {} data samples".format(num_samples))
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

    def get_model_parameters(self, bytes=False):
        if bytes:
            return self.serialize_model(self.model.state_dict())
        else:
            return self.model.state_dict()

    async def train(self):
        try:
            self.create_trainer()
            logging.info(f"{'='*10} [Training] Started (check training logs for progress) {'='*10}")
            with ThreadPoolExecutor() as pool:
                future = asyncio.get_running_loop().run_in_executor(pool, self._train_sync)
                await asyncio.wait_for(future, timeout=3600)
            self.__trainer = None
            logging.info(f"{'='*10} [Training] Finished (check training logs for progress) {'='*10}")
        except Exception as e:
            logging_training.error(f"Error training model: {e}")
            logging_training.error(traceback.format_exc())

    def _train_sync(self):
        try:
            self.__trainer.fit(self.model, self.data)
        except Exception as e:
            logging_training.error(f"Error in _train_sync: {e}")
            tb = traceback.format_exc()
            logging_training.error(f"Traceback: {tb}")
            # If "raise", the exception will be managed by the main thread

    async def test(self):
        try:
            self.create_trainer()
            logging.info(f"{'='*10} [Testing] Started (check training logs for progress) {'='*10}")
            with ThreadPoolExecutor() as pool:
                future = asyncio.get_running_loop().run_in_executor(pool, self._test_sync)
                loss, accuracy = await asyncio.wait_for(future, timeout=3600)
            self.__trainer = None
            logging.info(f"{'='*10} [Testing] Finished (check training logs for progress) {'='*10}")
            return loss, accuracy
        except Exception as e:
            logging_training.error(f"Error testing model: {e}")
            logging_training.error(traceback.format_exc())

    def _test_sync(self):
        try:
            self.__trainer.test(self.model, self.data, verbose=True)
            
            metrics = self.__trainer.callback_metrics
            self.__trainer = None
            loss = metrics.get('val_loss/dataloader_idx_0', None).item()
            accuracy = metrics.get('val_accuracy/dataloader_idx_0', None).item()
            return loss, accuracy
        except Exception as e:
            logging_training.error(f"Error in _test_sync: {e}")
            tb = traceback.format_exc()
            logging_training.error(f"Traceback: {tb}")
            # If "raise", the exception will be managed by the main thread

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
        self.model.on_round_end()
        logging.info("Flushing memory cache at the end of round...")
        torch.cuda.empty_cache()
        gc.collect()
        pass

    def on_learning_cycle_end(self):
        self._logger.log_data({"Round": self.round})
        # self.reporter.enqueue_data("Round", self.round)
        pass
