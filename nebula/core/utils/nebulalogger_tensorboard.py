import logging
from lightning.pytorch.loggers import TensorBoardLogger
from datetime import datetime


class NebulaTensorBoardLogger(TensorBoardLogger):

    def __init__(self, scenario_start_time, *args, **kwargs):
        self.scenario_start_time = scenario_start_time
        self.local_step = 0
        self.global_step = 0
        super().__init__(*args, **kwargs)

    def get_step(self):
        return int((datetime.now() - datetime.strptime(self.scenario_start_time, "%d/%m/%Y %H:%M:%S")).total_seconds())

    def log_data(self, data, step=None):
        if step is None:
            step = self.get_step()
        # logging.debug(f"Logging data for global step {step} | local step {self.local_step} | global step {self.global_step}")
        try:
            super().log_metrics(data, step)
        except ValueError as e:
            pass
        except Exception as e:
            logging.error(f"Error logging statistics data [{data}] for step [{step}]: {e}")

    def log_metrics(self, metrics, step=None):
        if step is None:
            self.local_step += 1
            step = self.global_step + self.local_step
        # logging.debug(f"Logging metrics for global step {step} | local step {self.local_step} | global step {self.global_step}")
        if "epoch" in metrics:
            metrics.pop("epoch")
        try:
            super().log_metrics(metrics, step)
        except Exception as e:
            logging.error(f"Error logging metrics [{metrics}] for step [{step}]: {e}")

    def log_figure(self, figure, step=None, name=None):
        if step is None:
            step = self.get_step()
        try:
            self.experiment.add_figure(name, figure, step)
        except Exception as e:
            logging.error(f"Error logging figure [{name}] for step [{step}]: {e}")

    def get_logger_config(self):
        return {"scenario_start_time": self.scenario_start_time, "local_step": self.local_step, "global_step": self.global_step}

    def set_logger_config(self, logger_config):
        if logger_config is None:
            return
        try:
            self.scenario_start_time = logger_config["scenario_start_time"]
            self.local_step = logger_config["local_step"]
            self.global_step = logger_config["global_step"]
        except Exception as e:
            logging.error(f"Error setting logger config: {e}")
