# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "acquire-zarr>=0.5.2",
#     "zarr",
#     "rich",
#     "tensorstore",
#     "click",
#     "psutil",
#     "s3fs",
#     "matplotlib",
#     "pandas",
# ]
# ///
#!/usr/bin/env python3
"""Compare write performance of TensorStore vs. acquire-zarr for sustained writes.

This benchmark demonstrates memory stability and throughput consistency over
extended acquisition sessions using real microscopy data.
"""

import json
import os
from pathlib import Path
import platform
import psutil
import shutil
import subprocess
import time
from typing import Tuple

import acquire_zarr as aqz
import click
import numpy as np
import tensorstore
import zarr
from rich import print


class CyclicArray:
    """Array-like wrapper that cycles through a fixed dataset."""
    def __init__(self, data: np.ndarray, n_frames: int):
        self.data = data
        self.t = self.data.shape[0]  # Size of first dimension
        self.shape = (n_frames,) + self.data.shape[1:]

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) > 0:
            if isinstance(idx[0], int):
                return self.data[(idx[0] % self.t,) + idx[1:]]
        elif isinstance(idx, int):
            return self.data[idx % self.t]

        return self.data[idx]

    def compare_array(self, arr: np.ndarray) -> None:
        """Compare an array with a CyclicArray."""
        assert self.shape == arr.shape

        for i in range(0, arr.shape[0], self.t):
            start = i
            stop = min(i + self.t, arr.shape[0])
            np.testing.assert_array_equal(
                self.data[0 : (stop - start)], arr[start:stop]
            )


class MemoryMonitor:
    """Track memory and throughput over time."""
    def __init__(self, sample_interval_s: float = 10.0, max_memory_mb: float = 10000):
        self.interval = sample_interval_s
        self.max_memory = max_memory_mb
        self.last_sample = 0
        self.samples = []  # (elapsed_s, rss_mb, throughput_mbs)
        self.process = psutil.Process()
        self.start_time = time.time()
        self.bytes_written = 0

    def update(self, bytes_this_write: int) -> tuple[bool, float | None]:
        """Update bytes written, sample if interval elapsed.
        Returns (should_stop, memory_mb)
        """
        self.bytes_written += bytes_this_write
        now = time.time()

        if now - self.last_sample >= self.interval:
            elapsed = now - self.start_time
            rss_mb = self.process.memory_info().rss / (1024**2)
            throughput = (self.bytes_written / elapsed / (1024**2)) if elapsed > 0 else 0
            self.samples.append((elapsed, rss_mb, throughput))
            self.last_sample = now

            print(f"  Memory: {rss_mb:.1f} MB, Throughput: {throughput:.2f} MB/s")

            should_stop = rss_mb > self.max_memory
            return should_stop, rss_mb

        return False, None

    def to_csv(self, path: str):
        """Write samples to CSV."""
        import csv
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_s', 'memory_mb', 'throughput_mbs'])
            writer.writerows(self.samples)


def load_s3_frames(n_frames: int = 128, cache_path: str = "s3_cache.npy") -> np.ndarray:
    """Load frames from S3 dataset, cache locally."""
    if Path(cache_path).exists():
        print(f"Loading cached frames from {cache_path}")
        return np.load(cache_path)

    print("Downloading frames from S3...")
    # Open the zarr store from S3 (public, no auth needed)
    import s3fs
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(
        root="aind-open-data/exaSPIM_LCTHY1_2025-02-14_20-36-25/exaSPIM/tile_000000_ch_488.zarr",
        s3=s3
    )
    zarray = zarr.open(store, mode='r')["0"]

    # Load first n_frames, crop to 2048x2048
    z_max = min(n_frames, zarray.shape[0])
    h = zarray.shape[1]
    w = zarray.shape[0]
    start_h = (h - 2048) // 2
    start_w = (w - 2048) // 2

    print("Loading data from S3...", end="")
    frames = zarray[:z_max, start_h:start_h+2048, start_w:start_w+2048]
    print("done")

    frames = np.stack(frames).astype(np.uint16)
    np.save(cache_path, frames)
    print(f"Cached {len(frames)} frames to {cache_path}")
    return frames


def run_tensorstore_test(
        data: CyclicArray, path: str, metadata: dict, max_memory_mb: float = 10000
) -> Tuple[float, np.ndarray, MemoryMonitor]:
    """Write data using TensorStore, stop if memory exceeds threshold."""
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": path},
        "metadata": metadata,
        "delete_existing": True,
        "create": True,
    }
    ts = tensorstore.open(spec).result()
    print(ts)

    monitor = MemoryMonitor(sample_interval_s=10.0, max_memory_mb=max_memory_mb)
    total_start = time.perf_counter_ns()
    futures = []
    elapsed_times = []

    chunk_length = ts.schema.chunk_layout.write_chunk.shape[0]
    write_chunk_shape = (chunk_length, *ts.domain.shape[1:])
    chunk = np.empty(write_chunk_shape, dtype=np.uint16)
    bytes_per_write = chunk.nbytes

    stopped_early = False
    for i in range(data.shape[0]):
        start_plane = time.perf_counter_ns()
        chunk_idx = i % chunk_length
        chunk[chunk_idx] = data[i]

        if chunk_idx == chunk_length - 1:
            slc = slice(i - chunk_length + 1, i + 1)
            futures.append(ts[slc].write(chunk))
            chunk = np.empty(write_chunk_shape, dtype=np.uint16)

            should_stop, mem = monitor.update(bytes_per_write)
            if should_stop:
                print(f"\n!!! TensorStore stopped at frame {i}: memory exceeded {max_memory_mb} MB (current: {mem:.1f} MB)")
                stopped_early = True
                break

        elapsed = time.perf_counter_ns() - start_plane
        elapsed_times.append(elapsed)
        print(f"TensorStore: Plane {i} written in {elapsed / 1e6:.3f} ms")

    if not stopped_early:
        start_futures = time.perf_counter_ns()
        for future in futures:
            future.result()
        elapsed = time.perf_counter_ns() - start_futures
        elapsed_times.append(elapsed)
        print(f"TensorStore: Final futures took {elapsed / 1e6:.3f} ms")

    total_elapsed = time.perf_counter_ns() - total_start
    tot_ms = total_elapsed / 1e6
    print(f"TensorStore: Total write time: {tot_ms:.3f} ms")

    return tot_ms, np.array(elapsed_times) / 1e6, monitor


def run_acquire_zarr_test(
        data: CyclicArray,
        path: str,
        tchunk_size: int = 1,
        xy_chunk_size: int = 2048,
        xy_shard_size: int = 1,
) -> Tuple[float, np.ndarray, MemoryMonitor]:
    """Write data using acquire-zarr and track memory."""
    settings = aqz.StreamSettings(
        store_path=path,
        arrays=[
            aqz.ArraySettings(
                dimensions=[
                    aqz.Dimension(
                        name="t",
                        kind=aqz.DimensionType.TIME,
                        array_size_px=0,
                        chunk_size_px=tchunk_size,
                        shard_size_chunks=1,
                    ),
                    aqz.Dimension(
                        name="y",
                        kind=aqz.DimensionType.SPACE,
                        array_size_px=2048,
                        chunk_size_px=xy_chunk_size,
                        shard_size_chunks=xy_shard_size,
                    ),
                    aqz.Dimension(
                        name="x",
                        kind=aqz.DimensionType.SPACE,
                        array_size_px=2048,
                        chunk_size_px=xy_chunk_size,
                        shard_size_chunks=xy_shard_size,
                    ),
                ],
                data_type=aqz.DataType.UINT16,
            )
        ],
        overwrite=True,
    )

    stream = aqz.ZarrStream(settings)
    monitor = MemoryMonitor(sample_interval_s=10.0)
    elapsed_times = []

    total_start = time.perf_counter_ns()
    chunk = np.empty((tchunk_size, 2048, 2048), dtype=np.uint16)
    bytes_per_write = chunk.nbytes

    for i in range(data.shape[0]):
        start_plane = time.perf_counter_ns()
        chunk_idx = i % tchunk_size
        chunk[chunk_idx] = data[i]

        if chunk_idx == tchunk_size - 1:
            stream.append(chunk)
            monitor.update(bytes_per_write)

        elapsed = time.perf_counter_ns() - start_plane
        elapsed_times.append(elapsed)
        print(f"Acquire-zarr: Plane {i} written in {elapsed / 1e6:.3f} ms")

    start_close = time.perf_counter_ns()
    stream.close()
    elapsed = time.perf_counter_ns() - start_close
    elapsed_times.append(elapsed)
    print(f"Acquire-zarr: Final close took {elapsed / 1e6:.3f} ms")

    total_elapsed = time.perf_counter_ns() - total_start
    tot_ms = total_elapsed / 1e6
    print(f"Acquire-zarr: Total write time: {tot_ms:.3f} ms")

    return tot_ms, np.array(elapsed_times) / 1e6, monitor


def get_git_commit_hash():
    """Get the current git commit hash, or None if not in a git repo."""
    cwd = os.getcwd()
    os.chdir(Path(__file__).parent)

    hash_out = None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        hash_out = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    finally:
        os.chdir(cwd)

    return hash_out


def get_system_info() -> dict:
    """Collect system information for benchmark context."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # try to get CPU brand on different platforms
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["cpu_brand"] = result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_brand"] = line.split(":")[1].strip()
                        break
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["cpu_brand"] = result.stdout.split("\n")[1].strip()
    except Exception:
        info["cpu_brand"] = "Unknown"

    return info


def plot_results(az_csv: str, ts_csv: str, output_prefix: str = "benchmark"):
    """Generate plots comparing memory and throughput over time."""
    import pandas as pd
    import matplotlib.pyplot as plt

    az_df = pd.read_csv(az_csv)
    ts_df = pd.read_csv(ts_csv)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Memory over time
    ax1.plot(az_df['elapsed_s'], az_df['memory_mb'], label='acquire-zarr', linewidth=2)
    ax1.plot(ts_df['elapsed_s'], ts_df['memory_mb'], label='TensorStore', linewidth=2)
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Memory Usage Over Time', fontsize=14, fontweight='bold')

    # Throughput over time
    ax2.plot(az_df['elapsed_s'], az_df['throughput_mbs'], label='acquire-zarr', linewidth=2)
    ax2.plot(ts_df['elapsed_s'], ts_df['throughput_mbs'], label='TensorStore', linewidth=2)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Write Throughput Over Time', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_prefix}_comparison.png")
    plt.close()


def compare(
        t_chunk_size: int,
        xy_chunk_size: int,
        xy_shard_size: int,
        frame_count: int,
        do_compare: bool = True,
        max_memory_mb: float = 10000,
        cache_path: str = "s3_cache.npy",
) -> dict:
    print("tchunk_size:", t_chunk_size)
    print("xy_chunk_size:", xy_chunk_size)
    print("xy_shard_size:", xy_shard_size)
    print("frame_count:", frame_count)
    print("max_memory_mb:", max_memory_mb)
    print("compare_data:", do_compare)

    print("\nLoading data from S3:")
    frames = load_s3_frames(n_frames=128, cache_path=cache_path)
    data = CyclicArray(frames, frame_count)

    print("\nRunning acquire-zarr test:")
    az_path = "acquire_zarr_test.zarr"
    print("Saving to", Path(az_path).absolute())

    time_az_ms, frame_write_times_az, monitor_az = run_acquire_zarr_test(
        data, az_path, t_chunk_size, xy_chunk_size, xy_shard_size
    )
    monitor_az.to_csv("acquire_zarr_timeseries.csv")

    # use the exact same metadata that was used for the acquire-zarr test
    az = zarr.open(az_path)["0"]

    print("\nRunning TensorStore test:")
    ts_path = "tensorstore_test.zarr"
    time_ts_ms, frame_write_times_ts, monitor_ts = run_tensorstore_test(
        data,
        ts_path,
        {**az.metadata.to_dict(), "data_type": "uint16"},
        max_memory_mb,
    )
    monitor_ts.to_csv("tensorstore_timeseries.csv")

    # Data comparison (optional)
    comparison_result = None
    if do_compare:
        print("\nComparing the written data:", end=" ")
        try:
            ts = zarr.open(ts_path)
            data.compare_array(az)
            data.compare_array(ts)
            print("[OK]\n")

            metadata_match = ts.metadata == az.metadata
            print(f"Metadata matches: {metadata_match}")
            if metadata_match:
                print(ts.metadata)

            comparison_result = {
                "data_match": True,
                "metadata_match": metadata_match,
            }
        except Exception as e:
            print(f"[ERROR] Comparison failed: {e}")
            comparison_result = {
                "data_match": False,
                "metadata_match": False,
                "error": str(e),
            }
    else:
        print("\nSkipping data comparison")

    # clean up test data
    del az

    try:
        print("\nCleaning up test data...", end="")
        shutil.rmtree(az_path)
        shutil.rmtree(ts_path)
        print("[OK]")
    except Exception:
        print("[ERROR] Failed to remove test data")

    data_size_gib = (2048 * 2048 * 2 * frame_count) / (1 << 30)

    # Calculate statistics
    az_stats = {
        "total_time_ms": time_az_ms,
        "throughput_gib_per_s": 1000 * data_size_gib / time_az_ms,
        "frame_write_time_50th_percentile_ms": float(
            np.percentile(frame_write_times_az, 50)
        ),
        "frame_write_time_99th_percentile_ms": float(
            np.percentile(frame_write_times_az, 99)
        ),
        "peak_memory_mb": max(s[1] for s in monitor_az.samples) if monitor_az.samples else 0,
        "avg_memory_mb": sum(s[1] for s in monitor_az.samples) / len(monitor_az.samples) if monitor_az.samples else 0,
    }

    ts_stats = {
        "total_time_ms": time_ts_ms,
        "throughput_gib_per_s": 1000 * data_size_gib / time_ts_ms,
        "frame_write_time_50th_percentile_ms": float(
            np.percentile(frame_write_times_ts, 50)
        ),
        "frame_write_time_99th_percentile_ms": float(
            np.percentile(frame_write_times_ts, 99)
        ),
        "peak_memory_mb": max(s[1] for s in monitor_ts.samples) if monitor_ts.samples else 0,
        "avg_memory_mb": sum(s[1] for s in monitor_ts.samples) / len(monitor_ts.samples) if monitor_ts.samples else 0,
        "stopped_early": len(monitor_ts.samples) > 0 and monitor_ts.samples[-1][1] > max_memory_mb,
    }

    print("\nPerformance comparison:")
    print(
        f"  acquire-zarr: {az_stats['total_time_ms']:.3f} ms, {az_stats['throughput_gib_per_s']:.3f} GiB/s\n"
        f"    Frame write time - 50th: {az_stats['frame_write_time_50th_percentile_ms']:.3f} ms, "
        f"99th: {az_stats['frame_write_time_99th_percentile_ms']:.3f} ms\n"
        f"    Memory - peak: {az_stats['peak_memory_mb']:.1f} MB, avg: {az_stats['avg_memory_mb']:.1f} MB"
    )
    print(
        f"  TensorStore: {ts_stats['total_time_ms']:.3f} ms, {ts_stats['throughput_gib_per_s']:.3f} GiB/s\n"
        f"    Frame write time - 50th: {ts_stats['frame_write_time_50th_percentile_ms']:.3f} ms, "
        f"99th: {ts_stats['frame_write_time_99th_percentile_ms']:.3f} ms\n"
        f"    Memory - peak: {ts_stats['peak_memory_mb']:.1f} MB, avg: {ts_stats['avg_memory_mb']:.1f} MB\n"
        f"    Stopped early: {ts_stats['stopped_early']}"
    )
    print(f"  TS/AZ Ratio: {time_ts_ms / time_az_ms:.3f}")

    # Structure results for JSON output
    results = {
        "test_parameters": {
            "t_chunk_size": t_chunk_size,
            "xy_chunk_size": xy_chunk_size,
            "xy_shard_size": xy_shard_size,
            "frame_count": frame_count,
            "data_size_gib": data_size_gib,
            "max_memory_mb": max_memory_mb,
        },
        "acquire_zarr": az_stats,
        "tensorstore": ts_stats,
        "ratio_ts_to_az": time_ts_ms / time_az_ms,
        "timestamp": time.time(),
        "git_commit_hash": get_git_commit_hash(),
        "system_info": get_system_info(),
    }

    if comparison_result is not None:
        results["comparison"] = comparison_result

    return results


@click.command()
@click.option("--t-chunk-size", default=64, help="Time dimension chunk size")
@click.option("--xy-chunk-size", default=64, help="Spatial dimension chunk size")
@click.option("--xy-shard-size", default=16, help="Spatial dimension shard size")
@click.option("--frame-count", default=1024, help="Number of frames to write")
@click.option("--max-memory-mb", default=10000, help="Max memory before stopping TensorStore test")
@click.option("--output", default="results.json", help="Output file for results")
@click.option("--nocompare/--compare", default=False, help="Disable data comparison")
@click.option("--plot/--no-plot", default=True, help="Generate comparison plots")
@click.option("--cache-frames", default="s3_cache.npy", help="Path to cache S3 frames")
def main(t_chunk_size, xy_chunk_size, xy_shard_size, frame_count, max_memory_mb,
         output, nocompare, plot, cache_frames):
    """Compare write performance of TensorStore vs. acquire-zarr."""
    results = compare(
        t_chunk_size, xy_chunk_size, xy_shard_size, frame_count,
        not nocompare, max_memory_mb, cache_frames
    )

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {output}")

    if plot and Path("acquire_zarr_timeseries.csv").exists():
        plot_results("acquire_zarr_timeseries.csv", "tensorstore_timeseries.csv",
                     output.replace('.json', ''))


if __name__ == "__main__":
    main()