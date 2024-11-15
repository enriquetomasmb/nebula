import json
import logging
import os
import platform
import re
import subprocess

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def normalize_cpu_string(cpu_string):
    # Remove specific substrings
    cpu_string = cpu_string.replace("(R)", "").replace("(TM)", "").replace("CPU", "")
    # Remove special characters and extra spaces
    normalized_string = re.sub(r"[^\w\s]", "", cpu_string)
    # Convert to lowercase
    normalized_string = normalized_string.lower()
    # Remove extra spaces
    normalized_string = " ".join(normalized_string.split())
    return normalized_string


def get_cup_benchmark(fileName="CPU_benchmark_v4.csv"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, fileName)

    # Read the CSV file and create a dictionary to map the CPU model to the TDP
    cpu_tdp_df = pd.read_csv(csv_file_path)  # The file is in the same directory as the node.py
    cpu_tdp_df["cpuName"] = cpu_tdp_df["cpuName"].apply(normalize_cpu_string)

    cpu_tdp_dict = dict(zip(cpu_tdp_df["cpuName"], cpu_tdp_df["TDP"], strict=False))
    return cpu_tdp_dict


def get_cpu_tdp(cpu_model):
    cpu_model = normalize_cpu_string(cpu_model)
    cpu_tdp_dict = get_cup_benchmark(fileName="CPU_benchmark_v4.csv")
    tdp = cpu_tdp_dict.get(cpu_model)
    logging.info(f"cpu tdp  {tdp}")
    logging.info(f"cpu device {cpu_model}")

    # print(cpu_tdp_dict)
    if tdp is None:
        logging.warning(f"TDP for CPU model '{cpu_model}' not found. Using average TDP value.")
        tdp = 62.38
    return tdp  # If the TDP cannot be found, the average TDP is returned


def get_cpu_models():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Linux":
        command = "grep 'model name' /proc/cpuinfo | head -n 1 | awk -F ': ' '{ print $2 }'"
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
            brand = result.stdout.strip()
        except Exception:
            brand = "Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz"
        return brand

    elif platform.system() == "Darwin":
        command = "sysctl -n machdep.cpu.brand_string"
        model = subprocess.check_output(command, shell=True).strip().decode()
        return model
    return "Unknown CPU"


def get_world_map(
    jsonFilePath="ne_10m_admin_0_countries/global_energy_mix.json",
    shpFilePath="ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp",
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, jsonFilePath), encoding="utf-8") as file:
        carbon_intensity_data = json.load(file)
        world_map = gpd.read_file(os.path.join(current_dir, shpFilePath))
    return carbon_intensity_data, world_map


def get_carbon_intensity(longitude, latitude):
    point = Point(longitude, latitude)
    logging.info(f"longitude{longitude, latitude}")
    carbon_intensity_data, world_map = get_world_map(
        jsonFilePath="ne_10m_admin_0_countries/global_energy_mix.json",
        shpFilePath="ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp",
    )
    for _, row in world_map.iterrows():
        try:
            if row["geometry"].contains(point):
                return carbon_intensity_data[row["ISO_A3"]]["carbon_intensity"]
        except Exception as e:
            print(f"error {e}")
            logging.exception(f"error {e}")
            return 100
    return 500


def get_sustain_energy_consumption(gpu_powers, cpu_models_tdp, cpu_util, last_time, pue):
    total_tdp_all_cpus = sum(cpu_models_tdp * cpu_percent * 0.01 for cpu_percent in cpu_util)
    gpu_energy_consumption = last_time * gpu_powers / 3600000
    cpu_energy_consumption = pue * 0.01 * total_tdp_all_cpus * last_time / 3600000
    return cpu_energy_consumption, gpu_energy_consumption


def get_sustain_carbon_emission(carbon_intensity, renewable_energy, cpu_energy_consumption, gpu_energy_consumption):
    cpu_carbon_emission = cpu_energy_consumption * carbon_intensity * (100 - renewable_energy) * 0.01
    gpu_carbon_emission = gpu_energy_consumption * carbon_intensity * (100 - renewable_energy) * 0.01
    return cpu_carbon_emission, gpu_carbon_emission
