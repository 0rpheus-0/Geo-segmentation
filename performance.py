import torch
from torch.profiler import profile, ProfilerActivity
from constants import DEVICE, INFER_WIDTH, INFER_HEIGHT, INFER_CHANNEL
import psutil
import pynvml
import time
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import os
from torch.autograd import DeviceType

ave_size_image = 209244
image_count = 1024
batch_sizes = [1, 2, 4, 8, 16, 32]
# torch.set_num_threads(12)


def init_nvml():
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except:
        print("NVML init failed")
        return None


def batch_performance(
    model,
    iterations,
    batch_size,
    device,
):
    model = model.to(device)
    input_tensor = torch.randn(batch_size, INFER_CHANNEL, INFER_HEIGHT, INFER_WIDTH).to(
        device
    )
    model.eval()
    torch.no_grad()

    gpu_available = device.type == "cuda"
    if gpu_available:
        handle = init_nvml()
        gpu_available = handle is not None

    if gpu_available:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU]
    if gpu_available:
        activities.append(ProfilerActivity.CUDA)

    process = psutil.Process(os.getpid())

    forward_times = []
    cpu_load = []
    gpu_load = [] if gpu_available else None

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        start = time.time()
        for _ in range(iterations):
            start_forward = time.time()
            _ = model(input_tensor)
            end_forward = time.time()

            forward_times.append(end_forward - start_forward)
            cpu_load.append(process.cpu_percent(interval=None))
            if gpu_available:
                gpu_load.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                torch.cuda.synchronize()
        end = time.time()
        mem = process.memory_info().rss / 1024**2
        gpu_mem_used = (
            pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
            if gpu_available
            else None
        )
        if gpu_available:
            pynvml.nvmlShutdown()

    cpu_count = psutil.cpu_count(logical=True)
    cpu_load = [x / cpu_count for x in cpu_load]

    ave_cpu_load = stat.mean(cpu_load)
    ave_gpu_load = stat.mean(gpu_load) if gpu_available else None

    # print(prof.key_averages().table())

    key_averages_to_df(prof.key_averages(), gpu_available=gpu_available).to_excel(
        f"./log/bs{batch_size}.xlsx", index=False
    )

    total_time = end - start
    forward_time = sum(forward_times)
    overhead = total_time - forward_time
    throughput = image_count / forward_time
    throughput_tb_day = throughput * ave_size_image * 86400 / 1024**4
    return {
        "Batch Size": batch_size,
        "Iterations": iterations,
        "Total Time (s)": round(total_time, 4),
        "Forward Time (s)": round(forward_time, 4),
        "Overhead (s)": round(overhead, 4),
        "Throughput (image/s)": round(throughput, 2),
        "Throughput (TB/day)": round(throughput_tb_day, 6),
        "CPU Load (%)": ave_cpu_load,
        "CPU Mem Used (MB)": round(mem, 4),
        "GPU Load (%)": ave_gpu_load if ave_gpu_load is not None else "N/A",
        "GPU Mem Used (MB)": round(gpu_mem_used, 2)
        if gpu_mem_used is not None
        else "N/A",
        "Device": device,
    }


def test_batch_sizes(
    model,
    batch_sizes,
    image_count,
    device,
):
    results = []
    for batch_size in batch_sizes:
        iterations = int(image_count / batch_size)
        print(f"Testing batch size {batch_size}...")
        res = batch_performance(model, iterations, batch_size, device)
        results.append(res)

    df = pd.DataFrame(results)
    return df


def key_averages_to_df(key_averages_list, threshold_percent=1.0, gpu_available=False):
    self_total_cpu = sum(evt.self_cpu_time_total for evt in key_averages_list)
    self_total_cuda = sum(
        evt.self_device_time_total
        for evt in key_averages_list
        if evt.device_type == DeviceType.CUDA and not evt.is_user_annotation
    )
    records = []

    for evt in key_averages_list:
        record = {
            "Name": evt.key,
            "Calls": evt.count,
            "CPU total": evt.cpu_time_total_str,
            "Self CPU total": evt.self_cpu_time_total_str,
            "CPU Mem": format_bytes(evt.cpu_memory_usage),
            "Self CPU Mem": format_bytes(evt.self_cpu_memory_usage),
        }

        if evt.cpu_time_total is not None:
            record["CPU total %"] = round(evt.cpu_time_total / self_total_cpu * 100, 3)
            record["Self CPU total %"] = round(
                evt.self_cpu_time_total / self_total_cpu * 100, 3
            )
        else:
            record["CPU total %"] = 0
            record["Self CPU total %"] = 0

        record["GPU total"] = evt.device_time_total_str if gpu_available else 0
        record["Self GPU total"] = (
            format_time(evt.self_device_time_total) if gpu_available else 0
        )
        record["CUDA Mem"] = format_bytes(
            evt.device_memory_usage if gpu_available else 0
        )
        record["Self CUDA Mem"] = format_bytes(
            evt.self_device_memory_usage if gpu_available else 0
        )

        if gpu_available and evt.device_time_total is not None:
            record["Self GPU total %"] = round(
                evt.self_device_time_total / self_total_cuda * 100, 3
            )
        else:
            record["GPU total %"] = 0
            record["Self GPU total %"] = 0

        records.append(record)

    df = pd.DataFrame(records)

    df = df[
        (df["CPU total %"] >= threshold_percent)
        | (df["Self GPU total %"] >= threshold_percent)
    ]

    return df.reset_index(drop=True)


def format_time(microseconds: float):
    if microseconds >= 1_000_000:
        return f"{microseconds / 1_000_000:.3f} s"
    elif microseconds >= 1_000:
        return f"{microseconds / 1_000:.3f} ms"
    elif microseconds >= 1:
        return f"{microseconds:.3f} us"
    else:
        return f"{microseconds * 1_000:.3f} ns"


def format_bytes(num_bytes: float):
    sign = "-" if num_bytes < 0 else ""
    abs_bytes = abs(num_bytes)

    if abs_bytes >= 1 << 40:
        return f"{sign}{abs_bytes / (1 << 40):.3f} Tb"
    elif abs_bytes >= 1 << 30:
        return f"{sign}{abs_bytes / (1 << 30):.3f} Gb"
    elif abs_bytes >= 1 << 20:
        return f"{sign}{abs_bytes / (1 << 20):.3f} Mb"
    elif abs_bytes >= 1 << 10:
        return f"{sign}{abs_bytes / (1 << 10):.3f} Kb"
    else:
        return f"{sign}{abs_bytes:.0f} b"


model = torch.jit.load("models_unet_rgb/best_model_new.pt", map_location=DEVICE)


def plot_results(df, save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    batch_sizes = df["Batch Size"]

    axes[0, 0].plot(batch_sizes, df["Throughput (image/s)"], marker="o")
    axes[0, 0].set_title("Пропускная способность")
    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 0].set_ylabel("Throughput (image/s)")
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim(0)

    axes[0, 1].plot(batch_sizes, df["Forward Time (s)"], marker="o", color="orange")
    axes[0, 1].set_title("Время forward pass")
    axes[0, 1].set_xlabel("Batch Size")
    axes[0, 1].set_ylabel("Forward Time (s)")
    axes[0, 1].grid(True)

    axes[1, 0].plot(batch_sizes, df["CPU Mem Used (MB)"], marker="o", color="green")
    axes[1, 0].set_title("Память")
    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("CPU Mem Used (MB)")
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0)

    axes[1, 1].plot(batch_sizes, df["CPU Load (%)"], marker="o", color="red")
    axes[1, 1].set_title("Загрузка CPU")
    axes[1, 1].set_xlabel("Batch Size")
    axes[1, 1].set_ylabel("CPU Load (%)")
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(0, 100)

    axes[2, 0].plot(batch_sizes, df["GPU Mem Used (MB)"], marker="o", color="green")
    axes[2, 0].set_title("Память GPU")
    axes[2, 0].set_xlabel("Batch Size")
    axes[2, 0].set_ylabel("GPU Mem Used (MB)")
    axes[2, 0].grid(True)
    axes[2, 0].set_ylim(0)

    axes[2, 1].plot(batch_sizes, df["GPU Load (%)"], marker="o", color="red")
    axes[2, 1].set_title("Загрузка GPU")
    axes[2, 1].set_xlabel("Batch Size")
    axes[2, 1].set_ylabel("GPU Load (%)")
    axes[2, 1].grid(True)
    axes[2, 1].set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    df = test_batch_sizes(
        model,
        batch_sizes,
        image_count,
        DEVICE,
    )
    df.to_excel("log/batch_performance.xlsx", index=False)
    plot_results(df, save_path="log/performance_graphs.png")
