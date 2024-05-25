import csv
import docker
import time
import multiprocessing
import json

start = time.time()
# 130 -> 634 + 10 = 644 - 130 = 514
for x in range(130, 0, -1):
    print(f"Elapsed time: {x} sec. left")
    time.sleep(1)


def write_stats(queue):
    with open("container_stats.csv", "w+", newline="") as csvfile:
        fieldnames = ["Timestamp", "Container", "CPU Usage", "Memory Usage"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        while True:
            stat = queue.get()
            if stat is None:
                break
            writer.writerow(stat)
            csvfile.flush()


def stream_stats(container, queue):
    for stat in container.stats(stream=True):
        stat = json.loads(stat.decode("utf-8"))
        cpu_usage = stat["cpu_stats"]["cpu_usage"]["total_usage"]
        memory_usage = stat["memory_stats"]["usage"]
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        queue.put({"Timestamp": timestamp, "Container": container.name, "CPU Usage": cpu_usage, "Memory Usage": memory_usage})
        time.sleep(0.5)


client = docker.DockerClient(base_url="unix:///var/run/docker.sock")

queue = multiprocessing.Queue()

writer_process = multiprocessing.Process(target=write_stats, args=(queue,))
writer_process.start()

processes = []
for container in client.containers.list():
    p = multiprocessing.Process(target=stream_stats, args=(container, queue))
    p.start()
    processes.append(p)

start = time.time()
# 150 => 530 - 150
for x in range(380, 0, -1):
    print(f"Elapsed time: {x} sec. left")
    time.sleep(1)

for p in processes:
    p.terminate()

queue.put(None)

writer_process.join()

print("Stats written to container_stats.csv")
