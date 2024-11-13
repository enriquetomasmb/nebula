from lightning.pytorch.loggers import TensorBoardLogger

import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_data(log_dir, attribute='Resources/total_energy_consumption'):
    total_energy = 0
    # Record the maximum step in all files
    overall_latest_step = -1  

    # Traverse all folders in the log directory (one folder per process)
    for subdir in os.listdir(log_dir):
        subdir_path = os.path.join(log_dir, subdir)
        # Extract data from an event file
        event_acc = EventAccumulator(subdir_path)
        # Load all the data
        event_acc.Reload()

        # Get data for a specific tag
        if attribute in event_acc.scalars.Keys():
            energy_events = event_acc.Scalars(attribute)
            if energy_events:
                latest_event = max(energy_events, key=lambda e: e.step)
                total_energy += latest_event.value
                if latest_event.step > overall_latest_step:
                    overall_latest_step = latest_event.step
    
    return total_energy, overall_latest_step 


class AccumaltedLogger(TensorBoardLogger):

    def __init__(self, *args, **kwargs):
        self.local_step = 0
        self.global_step = 0
        super().__init__(*args, **kwargs)
    def log_metrics(self, metrics, step=None):
        # Any custom code to log metrics
        # FL round information
        self.local_step = step
        step = self.global_step + self.local_step

        AccumaltedLogger.all_global_step=step

        # logging.info(f'(statisticslogger.py) log_metrics: step={step}, metrics={metrics}')
        if "epoch" in metrics:
            metrics.pop("epoch")
        super().log_metrics(metrics, step)  # Call the original log_metrics
    
    # aggragate all information
    def aggregate_nodes_data(self):
        Total_nodes_energy_consumption,step_energy=read_tensorboard_data(os.path.join(self._root_dir,self._name),"Sustainability/Total_energy_consumption")

        Total_nodes_carbon_emission,step_carbon=read_tensorboard_data(os.path.join(self._root_dir,self._name),"Sustainability/Total_carbon_emission")

        if step_carbon<step_energy:
            step=step_energy
        else:
            step=step_carbon
        # print("total_nodes_energy_consumption",total_nodes_energy_consumption)
        metrics={"Sustainability/Total_nodes_energy_consumption":Total_nodes_energy_consumption,
                 "Sustainability/ Total_ nodes_carbon_emission":Total_nodes_carbon_emission}
        super().log_metrics(metrics, step)