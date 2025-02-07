import datetime
import torch
from diffusers import DiffusionPipeline
import time
import numpy as np
import pynvml
import psutil
import gc
import csv

MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"

def save_results_to_csv(results, filename="benchmark_results.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["Batch Size", "Resolution", "Avg Latency (s)", "P90 Latency (s)", "Images per Second", "Avg GPU Memory (MB)", "Avg CPU Memory (MB)"])
        # Write the data
        for result in results:
            writer.writerow([
                result["batch_size"],
                result["resolution"],
                result["avg_latency"],
                result["p90_latency"],
                result["images_per_second"],
                result["avg_gpu_memory_mb"],
                result["avg_cpu_memory_mb"]
            ])

def get_gpu_memory_usage():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2  # Convert to MB

def get_cpu_memory_usage():
    return psutil.virtual_memory().used / 1024**2  # Convert to MB

def benchmark_stable_diffusion(num_iterations=10, batch_sizes=[1, 2, 4, 8], resolutions=[(512, 512), (1024, 1024)]):
    model_id = MODEL_ID
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "A serene landscape with mountains and a lake at sunset, highly detailed, professional photography, 8k resolution."

    results = []

    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size}")
        for resolution in resolutions:
            print(f"Resolution: {resolution}")
            latencies = []
            gpu_memories = []
            cpu_memories = []

            # Warmup
            pipe(
                [prompt] * batch_size,
                height=resolution[0],
                width=resolution[1],
                num_inference_steps=4,
                guidance_scale=0.0
            )

            for i in range(num_iterations):
                print(f"Iteration: {i}")
                torch.cuda.synchronize()
                start_time = time.time()
                gpu_mem_start = get_gpu_memory_usage()
                cpu_mem_start = get_cpu_memory_usage()

                _ = pipe(
                    [prompt] * batch_size,
                    height=resolution[0],
                    width=resolution[1],
                    num_inference_steps=4,
                    guidance_scale=0.0,
                ).images

                torch.cuda.synchronize()
                end_time = time.time()
                gpu_mem_end = get_gpu_memory_usage()
                cpu_mem_end = get_cpu_memory_usage()

                latency = end_time - start_time
                latencies.append(latency)
                gpu_memories.append(max(gpu_mem_start, gpu_mem_end))
                cpu_memories.append(max(cpu_mem_start, cpu_mem_end))

                # Clear CUDA cache to get more accurate memory usage
                torch.cuda.empty_cache()
                gc.collect()

            avg_latency = np.mean(latencies)
            p90_latency = np.percentile(latencies, 90)
            images_per_second = batch_size / avg_latency
            avg_gpu_memory = np.mean(gpu_memories)
            avg_cpu_memory = np.mean(cpu_memories)

            print({
                "batch_size": batch_size,
                "resolution": resolution,
                "avg_latency": avg_latency,
                "p90_latency": p90_latency,
                "images_per_second": images_per_second,
                "avg_gpu_memory_mb": avg_gpu_memory,
                "avg_cpu_memory_mb": avg_cpu_memory
            })
            results.append({
                "batch_size": batch_size,
                "resolution": resolution,
                "avg_latency": avg_latency,
                "p90_latency": p90_latency,
                "images_per_second": images_per_second,
                "avg_gpu_memory_mb": avg_gpu_memory,
                "avg_cpu_memory_mb": avg_cpu_memory
            })

    return results

if __name__ == "__main__":
    results = benchmark_stable_diffusion()
    # Get the current date
    current_date = datetime.datetime.now()
    formatted_date = current_date.strftime("%Y_%m_%d")
    save_results_to_csv(results, filename=f"sd_benchmark_results_{formatted_date}.csv")