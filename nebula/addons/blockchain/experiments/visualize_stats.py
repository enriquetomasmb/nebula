import ast
import csv

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
plt.rc("axes", axisbelow=True)
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 8, "ytick.labelsize": 8})


def bar_color(bar_name: str):
    color_dict = {
        "Validators": (158, 42, 43),
        "Non-Validator": (224, 159, 62),
        "Oracle": (184, 166, 142),
        "Front-End": (153, 168, 140),
        "Boot": (95, 15, 64),
        "Core": (51, 92, 103),
    }

    for key in color_dict.keys():
        if key in bar_name:
            return tuple((c / 255 / 1.2 for c in color_dict[key]))

    return color_dict.get(bar_name, (0, 0, 0))


def create_bar_chart(bar_names: list, bar_heights: list, y_title: str):
    plt.clf()
    fig, ax = plt.subplots(figsize=(6.4, 2.5))

    cmap = plt.get_cmap("Blues_r")

    for i in range(len(bar_names)):
        color = cmap(float(i) / len(bar_names) / 1.3)
        bar = ax.bar(bar_names[i], bar_heights[i], label=bar_names[i], color=color)
        height = bar[0].get_height()
        if not i:
            ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height * 2, "{:.2f}".format(height), va="center", ha="center", color="black", fontsize=8)
        else:
            ax.text(bar[0].get_x() + bar[0].get_width() / 2.0, height / 1.8, "{:.2f}".format(height), va="center", ha="center", color="white", fontsize=8)

    ax.set_ylabel(y_title)
    plt.yticks([i / 10 for i in range(0, 6)])
    plt.title("Aggregation Time by Algorithm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"00_avg_aggregation_time_algorithms.png", dpi=200)


def utility_chart(
    normalized: bool,
    x_title: str,
    y_title: str,
    plot_title: str,
    file_name: str,
    keep_columns: list,
    divisor: int = 1,
) -> None:
    plt.clf()

    df = pd.read_csv("rounds.csv")

    df = df[df["Measure Performance"] == True]
    df = df[df["Number of Cores"] == 10]
    df = df[df["Number of Malicious Cores"] == 0]
    df = df[df["Topology"] == "full mesh"]

    keep_columns.append("Number of Validators")
    df = df[keep_columns]

    x_axis_column = "Number of Validators"

    df = df.groupby(x_axis_column).mean().reset_index()

    if "CPU" in plot_title:
        df = df.groupby(x_axis_column).max().reset_index()
        averages = df.drop(columns=[x_axis_column]).max()
    else:
        df = df.groupby(x_axis_column).mean().reset_index()
        averages = df.drop(columns=[x_axis_column]).mean()

    sorted_averages = averages.sort_values(ascending=False)

    x_axis_data = df[x_axis_column].copy()

    df = df[sorted_averages.index]

    df.insert(0, x_axis_column, x_axis_data)

    if normalized:
        df_normalized = df.drop(columns=[x_axis_column]).div(df.drop(columns=[x_axis_column]).sum(axis=1) * divisor, axis=0)
    else:
        df_normalized = df.drop(columns=[x_axis_column]).div(divisor, axis=0)

    df_normalized.insert(0, x_axis_column, x_axis_data)
    df_normalized = df_normalized[df_normalized[x_axis_column].isin([1, 2]) == False]

    fig, ax = plt.subplots(figsize=(6.4, 3.5))

    for i in range(1, len(df.columns)):
        color = bar_color(df.columns[i])

        bars = ax.bar(df[x_axis_column], df_normalized.iloc[:, i], bottom=df_normalized.iloc[:, 1:i].sum(axis=1), label=df.columns[i].replace("AVG CPU-Time of ", "").replace("AVG Memory of ", ""), color=color)

        for j, bar in enumerate(bars):
            yval = bar.get_height()
            cumulative_height = df_normalized.iloc[:, 1 : i + 1].sum(axis=1)[j]
            if normalized:
                if i <= 1:
                    ax.text(bar.get_x() + bar.get_width() / 2, cumulative_height - yval / 3, f"{round(yval * 100, 1)}%", va="center", ha="center", color="white", fontsize=8, weight="semibold")
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.95 + 0.05 * i, f"{round(yval * 100, 1)}%", va="center", ha="center", color=bar_color(df.columns[i]), fontsize=8, weight="semibold")
            elif j > 0:
                if i <= 2:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, cumulative_height - yval / 2, round(yval, 2), va="center", ha="center", color="white" if i <= 2 else bar_color(df.columns[i]), fontsize=8, weight="semibold")
                else:
                    shift = 2.5 if "CPU" in df.columns[i] else 7
                    ax.text(bar.get_x() + bar.get_width() / 2.0, cumulative_height + shift * (1 + i % 3), round(yval, 2), va="center", ha="center", color="white" if i <= 2 else bar_color(df.columns[i]), fontsize=8, weight="semibold")

    if normalized:
        ax.yaxis.set_major_formatter(lambda x, _: "{:.0%}".format(x))

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(plot_title)
    plt.grid(True)

    if normalized:
        plt.yticks(np.arange(0, 1.31, 0.1))
    else:
        if "CPU" in plot_title:
            plt.yticks(np.arange(0, 61, 10))
        else:
            plt.yticks(np.arange(0, 161, 20))

    plt.xticks(range(0, 11, 1))

    legend = ax.legend(loc="lower left" if normalized else "upper left", edgecolor=None, fontsize=8)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((1, 1, 1, 1))
    plt.tight_layout()
    plt.savefig(f"00_{file_name}.png", dpi=200)


def reputation_single_run(run: int):
    plt.clf()
    with open("rounds.csv", "r") as file:
        csv_reader = csv.DictReader(file)

        for _ in range(run):
            next(csv_reader)

        nth_row = next(csv_reader)
        nested_list_str = nth_row["Time Series Reputation"]

    nested_list = ast.literal_eval(nested_list_str)
    df = pd.DataFrame(nested_list, columns=["Container", "Reputation", "Aggregation Round"])

    df = df.groupby(["Container", "Aggregation Round"])["Reputation"].mean().reset_index()

    malicious_reputations = []
    honest_reputations = []

    for idx, container in enumerate(df["Container"].unique()):
        container_df = df[df["Container"] == container]
        if idx <= 1:
            malicious_reputations.append(container_df["Reputation"].values)
        else:
            honest_reputations.append(container_df["Reputation"].values)

    malicious_reputations = np.array(malicious_reputations)
    honest_reputations = np.array(honest_reputations)

    malicious_avg = np.mean(malicious_reputations, axis=0)
    honest_avg = np.mean(honest_reputations, axis=0)

    aggregation_rounds = df["Aggregation Round"].unique()

    plt.plot(aggregation_rounds, malicious_avg, color="darkred", linestyle="-", label="Average Malicious", linewidth=1)
    plt.plot(aggregation_rounds, honest_avg, color="steelblue", linestyle="-", label="Average Honest", linewidth=1)
    data_points = np.vstack((malicious_reputations, honest_reputations))
    for i, round in enumerate(aggregation_rounds):
        plt.boxplot(data_points[:, i], positions=[round], widths=0.25, manage_ticks=False, sym=".")

    plt.legend(fontsize=8)
    plt.xlabel("Aggregation Round")
    plt.ylabel("Reputation (%)")
    plt.xticks([r for r in df["Aggregation Round"].unique()])
    plt.yticks(np.arange(0, 101, 10))
    plt.title("Reputation of Honest and Malicious Cores with Reputation Poisoning")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"00_reputation_series_chart_{run}.png", dpi=200)


def reputation_single_run_box(run: int):
    plt.clf()
    with open("rounds.csv", "r") as file:
        csv_reader = csv.DictReader(file)

        for _ in range(run):
            next(csv_reader)

        nth_row = next(csv_reader)
        nested_list_str = nth_row["Time Series Reputation"]

    nested_list = ast.literal_eval(nested_list_str)
    df = pd.DataFrame(nested_list, columns=["Container", "Reputation", "Aggregation Round"])

    df = df.groupby(["Container", "Aggregation Round"])["Reputation"].mean().reset_index()

    df["Type"] = "Honest"
    for idx, container in enumerate(df["Container"].unique()):
        if idx <= 1:
            df.loc[df["Container"] == container, "Type"] = "Malicious"

    sns.boxplot(x="Aggregation Round", y="Reputation", hue="Type", data=df)

    plt.xlabel("Aggregation Round")
    plt.ylabel("Reputation (%)")
    plt.xticks([r for r in df["Aggregation Round"].unique()])
    plt.yticks(np.arange(0, 101, 10))
    plt.title("Reputation by Participant with Reputation Poisoning")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"00_reputation_series_chart_{run}_box.png", dpi=200)


def reputation_single_run_heatmap(run: int):
    with open("rounds.csv", "r") as file:
        csv_reader = csv.DictReader(file)

        for _ in range(run):
            next(csv_reader)

        nth_row = next(csv_reader)
        nested_list_str = nth_row["Time Series Reputation"]

    nested_list = ast.literal_eval(nested_list_str)
    df = pd.DataFrame(nested_list, columns=["Container", "Reputation", "Aggregation Round", "Sender"])

    total_reputation = df.groupby("Container")["Reputation"].mean().sort_values(ascending=True)

    new_names = {old_name: f"Core {i + 1}" for i, old_name in enumerate(total_reputation.index)}

    df["Container"] = df["Container"].map(new_names)
    df["Sender"] = df["Sender"].map(new_names)

    df["Container"] = pd.Categorical(df["Container"], categories=sorted(new_names.values(), key=lambda x: int(x.split()[1])))
    df["Sender"] = pd.Categorical(df["Sender"], categories=sorted(new_names.values(), key=lambda x: int(x.split()[1])))

    pivot_df = df.pivot_table(values="Reputation", index="Container", columns="Sender")

    pivot_df = pivot_df.sort_index(ascending=True)
    pivot_df = pivot_df.sort_index(axis=1, ascending=True)

    core_10 = pivot_df.pop("Core 10")
    pivot_df["Core 10"] = core_10

    core_10 = pivot_df.loc["Core 10"]
    pivot_df = pivot_df.drop("Core 10")
    pivot_df.loc["Core 10"] = core_10

    final_order_1 = ["Core 3", "Core 4", "Core 5", "Core 1", "Core 2", "Core 6", "Core 7", "Core 8", "Core 9", "Core 10"]
    rename_map = {name: f"Core {i + 1}" for i, name in enumerate(final_order_1)}
    pivot_df.rename(index=rename_map, columns=rename_map, inplace=True)
    pivot_df = pivot_df.reindex(index=rename_map.values(), columns=rename_map.values())

    final_order_2 = [f"Malicious {nr}" for nr in range(1, 5)] + [f"Benign {nr}" for nr in range(1, 7)]
    rename_map = {name: final_order_2[i] for i, name in enumerate([f"Core {i}" for i in range(1, 11)])}
    pivot_df.rename(index=rename_map, columns=rename_map, inplace=True)
    pivot_df = pivot_df.reindex(index=rename_map.values(), columns=rename_map.values())

    plt.figure(figsize=(6.4, 3.5))
    sns.heatmap(pivot_df, cmap="RdBu", annot=True, fmt=".1f", annot_kws={"size": 8})
    plt.title("Heatmap of Average Reported Opinion with Malicious Cores")
    plt.gca().invert_yaxis()
    plt.ylabel("Rating Core", fontsize=10)
    plt.xlabel("Rated Core", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"00_reputation_series_chart_{run}_heat.png", dpi=200)


def accuracy_MNIST(fed_avg: list, blockchain: list, krum: list, title: str, filename: str):
    x_percentage = [i * 10 for i in range(len(fed_avg))]

    cmap = plt.get_cmap("Blues_r")

    fig, ax = plt.subplots(figsize=(6.4, 3.5))

    plt.plot(x_percentage, fed_avg, marker="o", linestyle="--", markersize=3, linewidth=0.75, color=cmap(0), label="FedAvg")

    plt.plot(x_percentage, blockchain, marker="o", linestyle=":", markersize=3, linewidth=1, color=cmap(0.25), label="Blockchain Reputation")

    plt.plot(x_percentage, krum, marker="o", linestyle="-.", markersize=3, linewidth=0.75, color=cmap(0.5), label="Krum")

    for idx, value in enumerate(blockchain):
        plt.text(x_percentage[idx], value + 0.03, str(value), ha="center", fontsize=8)

    for idx, value in enumerate(fed_avg):
        if idx == 0:
            plt.text(x_percentage[0], value - 0.12, str(value), ha="center", fontsize=8)
        else:
            plt.text(x_percentage[idx], value + 0.03, str(value), ha="center", fontsize=8)

    for idx, value in enumerate(krum):
        plt.text(x_percentage[idx], value - 0.09, str(value), ha="center", fontsize=8)

    plt.xlabel("Percentage of Malicious Cores")
    plt.ylabel("Accuracy")
    plt.title(title)

    plt.xticks(x_percentage)
    plt.yticks([i / 10 for i in range(0, 12)])

    formatter = ticker.FuncFormatter(lambda x, pos: "{:d}%".format(int(x)))
    ax.xaxis.set_major_formatter(formatter)

    plt.legend()
    plt.grid(visible=True, color="lightgrey", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)


def total_gas_usage_single_run(run: int):
    plt.clf()

    with open("rounds.csv", "r") as file:
        csv_reader = csv.DictReader(file)

        for _ in range(run):
            next(csv_reader)

        nth_row = next(csv_reader)
        nested_list_str = nth_row["Time Series Gas"]

    nested_list = ast.literal_eval(nested_list_str)
    df = pd.DataFrame(nested_list, columns=["Gas in Wei", "Aggregation Round"])

    total_gas_per_round = df.groupby("Aggregation Round")["Gas in Wei"].sum().reset_index()

    total_gas_per_round["Gas in Wei"] = total_gas_per_round["Gas in Wei"] * float(0.000002494)

    plt.figure(figsize=(6.4, 3.5))
    plt.plot(total_gas_per_round["Aggregation Round"], total_gas_per_round["Gas in Wei"], marker="o", linestyle="--", dashes=(1, 2), markersize=3, linewidth=1)

    for i in range(len(total_gas_per_round)):
        plt.text(total_gas_per_round["Aggregation Round"].iloc[i], total_gas_per_round["Gas in Wei"].iloc[i] + 1, str(round(total_gas_per_round["Gas in Wei"].iloc[i])), ha="center", fontsize=8)

    plt.xlabel("Aggregation Round")
    plt.ylabel("Total Gas Costs over all Cores (USD)")

    plt.xticks([r for r in df["Aggregation Round"].unique()])
    plt.yticks(list(range(0, int(total_gas_per_round["Gas in Wei"].max()) + 10, 5)))

    plt.title("Total Gas Costs per Aggregation Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"00_run-{run}_total_gas_cost_per_round.png", dpi=200)


def aggregation_time_single_run(run: int):
    plt.clf()

    with open("rounds.csv", "r") as file:
        csv_reader = csv.DictReader(file)

        for _ in range(run):
            next(csv_reader)

        nth_row = next(csv_reader)
        nested_list_str = nth_row["Time Series Aggregation"]

    nested_list = ast.literal_eval(nested_list_str)
    df = pd.DataFrame(nested_list, columns=["Time", "Aggregation Round"])

    avg_time_per_aggregation = df.groupby("Aggregation Round").mean().reset_index()

    plt.figure(figsize=(6.4, 3.5))
    plt.plot(avg_time_per_aggregation["Aggregation Round"], avg_time_per_aggregation.iloc[:, 1:], marker="o", linestyle="--", dashes=(1, 2), markersize=3, linewidth=1)
    plt.xlabel("Aggregation Round")
    plt.ylabel("AVG Time used for Aggregation")
    plt.xticks([r for r in df["Aggregation Round"].unique()])
    plt.title("Average Time used for Aggregation per Round")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("00_avg_aggregation_time_per_round.png", dpi=200)


def aggregation_time_by_block_time():
    plt.clf()

    df = pd.read_csv("rounds.csv")

    df = df[df["Measure Performance"] == False]
    df = df[df["Number of Rounds"] == 5]
    df = df[df["Number of Cores"] == 10]
    df = df[df["Number of Malicious Cores"] == 0]
    df = df[df["Topology"] == "full mesh"]

    avg_times = []

    for index, row in df.iterrows():
        nested_list_str = row["Time Series Aggregation"]
        nested_list = ast.literal_eval(nested_list_str)

        nested_df = pd.DataFrame(nested_list, columns=["Time", "Aggregation Round"])

        avg_time = nested_df["Time"].mean()
        avg_times.append(avg_time)

    df["Average Time"] = avg_times
    avg_time_per_aggregation = df.groupby("Block Time")["Average Time"].mean().reset_index()
    # Plotting
    plt.figure(figsize=(6.4, 3.5))
    plt.plot(avg_time_per_aggregation["Block Time"], avg_time_per_aggregation["Average Time"], marker="o", linestyle="--", dashes=(1, 2), markersize=3, linewidth=1)

    for i in range(len(avg_time_per_aggregation)):
        plt.text(avg_time_per_aggregation["Block Time"].iloc[i], avg_time_per_aggregation["Average Time"].iloc[i] + 0.25, str(round(avg_time_per_aggregation["Average Time"].iloc[i], 2)), ha="right")
    plt.yticks(list(range(0, int(avg_time_per_aggregation["Average Time"].max()) + 2, 1)))

    plt.xlabel("Block Time in Seconds")
    plt.ylabel("Average Time used for Aggregation (sec)")
    plt.xticks([r for r in df["Block Time"].unique()])
    plt.title("Average Time used for Aggregation by Block Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("00_avg_aggregation_time_per_block_time.png", dpi=200)


def total_avg_gas_costs_by_n_participants():
    plt.clf()

    df = pd.read_csv("rounds.csv")

    df = df[df["Measure Performance"] == False]
    df = df[df["Number of Rounds"] == 5]
    df = df[df["Number of Malicious Cores"] == 0]
    df = df[df["Topology"] == "full mesh"]

    total_gas_per_aggregation = df.groupby("Number of Cores")["Total Gas (USD)"].mean().reset_index()

    total_gas_per_aggregation["Total Gas (USD)"] = total_gas_per_aggregation["Total Gas (USD)"] / 0.00001971
    total_gas_per_aggregation["Total Gas (USD)"] = total_gas_per_aggregation["Total Gas (USD)"] * 0.000002494

    plt.figure(figsize=(6.4, 3.5))
    plt.plot(total_gas_per_aggregation["Number of Cores"], total_gas_per_aggregation["Total Gas (USD)"], marker="o", linestyle="--", dashes=(1, 2), markersize=3, linewidth=1)

    for i in range(len(total_gas_per_aggregation)):
        plt.text(total_gas_per_aggregation["Number of Cores"].iloc[i], total_gas_per_aggregation["Total Gas (USD)"].iloc[i] + 5, str(int(total_gas_per_aggregation["Total Gas (USD)"].iloc[i])), ha="right")

    plt.xlabel("Number of Cores")
    plt.ylabel("Total Gas Costs of Scenario (USD)")

    plt.xticks([r for r in df["Number of Cores"].unique()])
    plt.yticks(list(range(0, int(total_gas_per_aggregation["Total Gas (USD)"].max()) + 30, 20)))
    plt.grid()
    plt.title("Total Gas Costs by Number of Cores")
    plt.tight_layout()
    plt.savefig("00_total_avg_gas_costs_by_n_participants.png", dpi=200)


utility_chart(
    normalized=True,
    x_title="Number of Validator Nodes",
    y_title="Relative CPU Time (%)",
    plot_title="Relative CPU Utilization",
    file_name="normalized_cpu_all",
    keep_columns=["AVG CPU-Time of Boot Node", "AVG CPU-Time of Front-End", "AVG CPU-Time of Oracle", "AVG CPU-Time of Non-Validator Node", "AVG CPU-Time of Validators", "AVG CPU-Time of Cores"],
)

utility_chart(
    normalized=True,
    x_title="Number of Validator Nodes",
    y_title="Average Relative Memory Utilization (%)",
    plot_title="Average Relative Memory Utilization",
    file_name="normalized_memory_all",
    keep_columns=["AVG Memory of Boot Node", "AVG Memory of Front-End", "AVG Memory of Oracle", "AVG Memory of Cores", "AVG Memory of Non-Validator Node", "AVG Memory of Validators"],
)

utility_chart(
    normalized=False,
    x_title="Number of Validator Nodes",
    y_title="Average Absolute Memory Utilization (MiB)",
    plot_title="Average Absolute Memory Utilization of Blockchain Network",
    file_name="absolute_memory_blockchain",
    keep_columns=["AVG Memory of Boot Node", "AVG Memory of Oracle", "AVG Memory of Non-Validator Node", "AVG Memory of Validators"],
    divisor=8388608,
)

utility_chart(
    normalized=False,
    x_title="Number of Validator Nodes",
    y_title="Absolute CPU Time (sec)",
    plot_title="Absolute CPU Utilization of Blockchain Network",
    file_name="absolute_cpu_blockchain",
    keep_columns=["AVG CPU-Time of Boot Node", "AVG CPU-Time of Oracle", "AVG CPU-Time of Non-Validator Node", "AVG CPU-Time of Validators"],
    divisor=10**9,
)

create_bar_chart(["FedAvg", "TrimMedian", "Krum", "Blockchain Reputation"], [0.02829, 0.10424, 0.32781, 0.47], "Time (seconds)")

accuracy_MNIST(
    fed_avg=[0.8914, 0.3655, 0.2122, 0.1686, 0.1565, 0.169],
    blockchain=[0.9590, 0.9558, 0.9563, 0.9597, 0.9599, 0.9545],
    krum=[0.9201, 0.9144, 0.9193, 0.9359, 0.9047, 0.9176],
    title="Accuracy of Aggregation Algorithms during Noise Injection (Non-IID)",
    filename="00_accuracy_model_poisoning_non_iid.png",
)

accuracy_MNIST(fed_avg=[0.8995, 0.4271, 0.2284, 0.1988, 0.1909], blockchain=[0.9616, 0.9612, 0.9452, 0.9583, 0.957], krum=[0.9598, 0.9598, 0.9303, 0.9409, 0.9264], title="Accuracy of Aggregation Algorithms during Noise Injection (IID)", filename="00_accuracy_model_poisoning_IID.png")

reputation_single_run(87)
# reputation_single_run_box(87)
# reputation_single_run_scatter(86)
reputation_single_run_heatmap(90)
aggregation_time_by_block_time()
aggregation_time_single_run(run=2)
total_gas_usage_single_run(run=2)
total_avg_gas_costs_by_n_participants()
