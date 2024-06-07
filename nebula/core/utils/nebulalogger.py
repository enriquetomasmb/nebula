from aim.pytorch_lightning import AimLogger
from datetime import datetime
import logging
from lightning.pytorch.utilities import rank_zero_only
from aim.sdk.run import Run
from aim import Image
from lightning.pytorch.loggers.logger import rank_zero_experiment
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nebula.core.engine import Engine


class NebulaLogger(AimLogger):

    def __init__(self, config, engine: "Engine", scenario_start_time, *args, **kwargs):
        self.config = config
        self.engine = engine
        self.scenario_start_time = scenario_start_time
        self.local_step = 0
        self.global_step = 0
        super().__init__(*args, **kwargs)

    def finalize(self, status: str = "") -> None:
        super().finalize(status)
        logging.info(f"Finalizing logger: {status}")

    def get_step(self):
        return int((datetime.now() - datetime.strptime(self.scenario_start_time, "%d/%m/%Y %H:%M:%S")).total_seconds())

    def log_data(self, data, step=None):
        time_start = datetime.now()
        try:
            logging.debug(f"Logging data: {data}")
            super().log_metrics(data)
        except Exception as e:
            logging.error(f"Error logging statistics data [{data}]: {e}")
        logging.debug(f"Time taken to log data: {datetime.now() - time_start}")

    def log_figure(self, figure, step=None, name=None):
        time_start = datetime.now()
        try:
            logging.debug(f"Logging figure: {name}")
            self.experiment.track(Image(figure), name=name)
        except Exception as e:
            logging.error(f"Error logging figure: {e}")
        logging.debug(f"Time taken to log figure: {datetime.now() - time_start}")
