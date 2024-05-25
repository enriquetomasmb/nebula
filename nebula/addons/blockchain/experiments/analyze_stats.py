import csv

import pandas as pd
import matplotlib.pyplot as plt
import requests
from tabulate import tabulate


def print_table(title: str, values: list, headers: list) -> None:
    print(f"\n{'-' * 25} {title.upper()} {'-' * 25}", flush=True)
    print(tabulate(sorted(values), headers=headers, tablefmt="grid"))


def export_stats(measure_performance: bool = False):

    df = pd.read_csv("container_stats.csv")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S")

    df = df.dropna(subset=["Timestamp"])

    df = df.drop_duplicates(keep="last")

    df.set_index("Timestamp", inplace=True)

    df.sort_values(by="Timestamp", inplace=True)

    """ PARTICIPANTS """
    for container in df["Container"].unique():
        first_cpu_usage = df[df["Container"] == container][f"CPU Usage"].iloc[0]
        df.loc[df["Container"] == container, f"CPU Usage"] -= first_cpu_usage

    participant_containers = df[df["Container"].str.contains("participant")]

    num_participant_containers = df[df["Container"].str.contains("participant")]["Container"].nunique()

    participant_containers.reset_index(inplace=True)

    participant_containers = participant_containers.drop_duplicates(subset=["Timestamp", "Container"], keep="last")
    participant_containers.set_index("Timestamp", inplace=True)

    participant_containers_sum = participant_containers.groupby("Timestamp").filter(lambda group: len(group) >= num_participant_containers).groupby("Timestamp")[["CPU Usage", "Memory Usage"]].sum()

    participant_containers_sum["Container"] = "participant_sum"

    """ VALIDATORS """
    validator_containers = df[df["Container"].str.contains("validator")]

    num_validator_containers = df[df["Container"].str.contains("validator")]["Container"].nunique()

    validator_containers.reset_index(inplace=True)

    validator_containers = validator_containers.drop_duplicates(subset=["Timestamp", "Container"], keep="last")

    validator_containers.set_index("Timestamp", inplace=True)

    validator_containers_sum = validator_containers.groupby("Timestamp").filter(lambda group: len(group) >= num_validator_containers).groupby("Timestamp")[["CPU Usage", "Memory Usage"]].sum()

    validator_containers_sum["Container"] = "validator_sum"

    """ OTHERS """
    other_containers = df[~df["Container"].str.contains("participant|validator")]

    df_sum_and_others = pd.concat([other_containers, participant_containers_sum, validator_containers_sum])

    """ PLOT CPU """

    for container in df_sum_and_others["Container"].unique():
        container_data = df_sum_and_others[df_sum_and_others["Container"] == container]
        plt.plot(container_data.index, container_data["CPU Usage"], label=container)

    plt.title("CPU Usage over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("CPU Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cpu_stats.png")
    plt.show()

    """ PLOT MEMORY """

    for container in df_sum_and_others["Container"].unique():
        df_sum_and_others[df_sum_and_others["Container"] == container]["Memory Usage"].plot(label=container)

    plt.title("Memory Usage over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Memory Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("memory_stats.png")
    plt.show()

    avg_usages = dict()
    for container in df_sum_and_others["Container"].unique():

        avg_cpu_usage = df_sum_and_others[df_sum_and_others["Container"] == container]["CPU Usage"].mean()
        avg_memory_usage = df_sum_and_others[df_sum_and_others["Container"] == container]["Memory Usage"].mean()

        avg_usages[container] = {"cpu": avg_cpu_usage, "memory": avg_memory_usage}

    """ GAS TOTAL """
    response = requests.get(url=f"http://172.25.0.105:8081/gas", headers={"Content-type": "application/json", "Accept": "application/json"}, timeout=10)
    response.raise_for_status()
    response_json = response.json()
    gas_wei = response_json["Sum (WEI)"]
    gas_usd = response_json["Sum (USD)"]

    """ GAS TIMESERIES """
    response = requests.get(url=f"http://172.25.0.105:8081/gas_series", headers={"Content-type": "application/json", "Accept": "application/json"}, timeout=10)
    response.raise_for_status()
    gas_series = response.json()

    """ TIME """
    response = requests.get(url=f"http://172.25.0.105:8081/time", headers={"Content-type": "application/json", "Accept": "application/json"}, timeout=10)
    response.raise_for_status()
    time_series = response.json()

    """ REPUTATION """
    response = requests.get(url=f"http://172.25.0.105:8081/reputation", headers={"Content-type": "application/json", "Accept": "application/json"}, timeout=10)

    response.raise_for_status()
    reputation_series = response.json()

    with open("rounds.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                measure_performance,
                num_validator_containers,  # n_validators
                num_participant_containers,  # n_participants
                9,  # rounds
                1,  # epochs
                "full mesh",  # topology
                0,  # malicious
                1,  # BLOCK TIME
                gas_wei,  # gas in wei
                gas_usd,  # gas in $
                avg_usages["boot"]["memory"] if measure_performance else 0,
                avg_usages["nebula-frontend"]["memory"] if measure_performance else 0,
                avg_usages["oracle"]["memory"] if measure_performance else 0,
                avg_usages["participant_sum"]["memory"] if measure_performance else 0,
                avg_usages["rpc"]["memory"] if measure_performance else 0,
                avg_usages["validator_sum"]["memory"] if measure_performance else 0,
                avg_usages["boot"]["cpu"] if measure_performance else 0,
                avg_usages["nebula-frontend"]["cpu"] if measure_performance else 0,
                avg_usages["oracle"]["cpu"] if measure_performance else 0,
                avg_usages["participant_sum"]["cpu"] if measure_performance else 0,
                avg_usages["rpc"]["cpu"] if measure_performance else 0,
                avg_usages["validator_sum"]["cpu"] if measure_performance else 0,
                0,  # Accuracy
                gas_series,
                reputation_series,
                time_series,
            ]
        )

    with open("rounds.csv", "r", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader)
        runs_rows = list([row for row in reader])

    print_table("All Runs", runs_rows, headers)


if __name__ == "__main__":
    export_stats()
