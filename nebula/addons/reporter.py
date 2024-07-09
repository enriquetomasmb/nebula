import importlib
import json
import logging
import queue
import threading
import time
import requests
import sys
import psutil
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nebula.core.network.communications import CommunicationsManager


class Reporter(threading.Thread):
    def __init__(self, config, trainer, cm: "CommunicationsManager"):
        threading.Thread.__init__(self, daemon=True, name="reporter_thread-" + config.participant["device_args"]["name"])
        logging.info(f"Starting reporter thread")
        self.config = config
        self.trainer = trainer
        self.cm = cm
        self.frequency = self.config.participant["reporter_args"]["report_frequency"]
        self.grace_time = self.config.participant["reporter_args"]["grace_time_reporter"]
        self.data_queue = queue.Queue()
        self.url = f'http://{self.config.participant["scenario_args"]["controller"]}/nebula/dashboard/{self.config.participant["scenario_args"]["name"]}/node/update'

    def enqueue_data(self, name, value):
        self.data_queue.put((name, value))

    def run(self):
        time.sleep(self.grace_time)
        while True:
            time.sleep(self.frequency)
            if self.config.participant["scenario_args"]["controller"] == "nebula-frontend":
                self.__report_status_to_controller()
            self.__report_resources()
            self.__report_data_queue()

    def report_scenario_finished(self):
        try:
            response = requests.post(
                url = f'http://{self.config.participant["scenario_args"]["controller"]}/nebula/dashboard/{self.config.participant["scenario_args"]["name"]}/node/done',
                data = json.dumps({"ip": self.config.participant["network_args"]["ip"] , "port": self.config.participant["network_args"]["port"]}),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f'NEBULA Participant {self.config.participant["device_args"]["idx"]}',
                },
            )
        except requests.exceptions.ConnectionError:
            logging.error(f"Error connecting to the controller at {self.url}")
            return
        if response.status_code != 200:
            logging.error(f"Error received from controller: {response.status_code} (probably there is overhead in the controller, trying again in the next round)")
            logging.debug(response.text)
            return                

    def __report_data_queue(self):
        try:
            while not self.data_queue.empty():
                name, value = self.data_queue.get()
                self.trainer.logger.log_data({name: value})
                self.data_queue.task_done()
        except queue.Empty:
            pass

    def __report_status_to_controller(self):
        try:
            response = requests.post(
                self.url,
                data=json.dumps(self.config.participant),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f'NEBULA Participant {self.config.participant["device_args"]["idx"]}',
                },
            )
        except requests.exceptions.ConnectionError:
            logging.error(f"Error connecting to the controller at {self.url}")
            return
        if response.status_code != 200:
            logging.error(f"Error received from controller: {response.status_code} (probably there is overhead in the controller, trying again in the next round)")
            logging.debug(response.text)
            return

    def __report_resources(self):
        cpu_percent = psutil.cpu_percent()
        cpu_temp = 0
        try:
            if sys.platform == "linux":
                cpu_temp = psutil.sensors_temperatures()["coretemp"][0].current
        except Exception as e:
            pass

        pid = os.getpid()
        cpu_percent_process = psutil.Process(pid).cpu_percent(interval=1)

        memory_process = psutil.Process(pid).memory_info().rss / (1024**2)
        memory_percent_process = psutil.Process(pid).memory_percent()
        memory_percent = psutil.virtual_memory().percent
        memory_used = psutil.virtual_memory().used / (1024**2)

        disk_percent = psutil.disk_usage("/").percent

        net_io_counters = psutil.net_io_counters()
        bytes_sent = net_io_counters.bytes_sent
        bytes_recv = net_io_counters.bytes_recv
        packets_sent = net_io_counters.packets_sent
        packets_recv = net_io_counters.packets_recv

        resources = {
            "CPU/CPU global (%)": cpu_percent,
            "CPU/CPU process (%)": cpu_percent_process,
            "CPU/CPU temperature (°)": cpu_temp,
            "RAM/RAM global (%)": memory_percent,
            "RAM/RAM global (MB)": memory_used,
            "RAM/RAM process (%)": memory_percent_process,
            "RAM/RAM process (MB)": memory_process,
            "Disk/Disk (%)": disk_percent,
            "Network/Network (bytes sent)": bytes_sent,
            "Network/Network (bytes received)": bytes_recv,
            "Network/Network (packets sent)": packets_sent,
            "Network/Network (packets received)": packets_recv,
            "Network/Connections": len(self.cm.get_addrs_current_connections(only_direct=True)),
        }
        self.trainer.logger.log_data(resources)

        if importlib.util.find_spec("pynvml") is not None:
            try:
                import pynvml

                pynvml.nvmlInit()
                devices = pynvml.nvmlDeviceGetCount()
                for i in range(devices):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem_percent = round(gpu_mem.used / gpu_mem.total * 100, 3)
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    gpu_clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    gpu_memory_clocks = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    gpu_fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                    gpu_info = {
                        f"GPU/GPU{i} (%)": gpu_percent,
                        f"GPU/GPU{i} temperature (°)": gpu_temp,
                        f"GPU/GPU{i} memory (%)": gpu_mem_percent,
                        f"GPU/GPU{i} power": gpu_power,
                        f"GPU/GPU{i} clocks": gpu_clocks,
                        f"GPU/GPU{i} memory clocks": gpu_memory_clocks,
                        f"GPU/GPU{i} fan speed": gpu_fan_speed,
                    }
                    self.trainer.logger.log_data(gpu_info)
            except Exception:
                pass
