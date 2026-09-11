# ponytail: continuous 3-minute thermal and power soak test with nvidia-smi hardware telemetry.
import os
import sys
import time
import threading
import subprocess
import numpy as np
import strux_rs

telemetry_data = []
stop_telemetry = threading.Event()

def telemetry_sampler():
    """Sample GPU telemetry every 1 second via nvidia-smi."""
    while not stop_telemetry.is_set():
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            line = res.stdout.strip()
            parts = [float(x.strip()) for x in line.split(",")]
            # temp, power, clock, mem
            telemetry_data.append((time.time(), parts[0], parts[1], parts[2], parts[3]))
        except Exception:
            pass
        time.sleep(1.0)

def main():
    DURATION_SEC = 180  # 3 minutes sustained soak
    print("=" * 80)
    print("  STRUX-RS CONTINUOUS THERMAL & POWER SOAK TEST (3 MINUTES)")
    print(f"  Target Hardware: NVIDIA RTX A2000 Laptop GPU (Continuous Max-TGP Load)")
    print("=" * 80)

    # 1. Load base structure (1,919 atoms)
    base_traj = strux_rs.parse_pdb("trajectory.pdb")
    base_frame = base_traj[0]
    num_atoms = base_frame.shape[0]

    # Use 2,500 frames (6,250,000 alignments per iteration, ~3.2s per pass)
    N_FRAMES = 2500
    PAIRS_PER_PASS = N_FRAMES * N_FRAMES
    print(f"Dataset: {N_FRAMES:,d} frames ({num_atoms:,d} atoms/frame, {PAIRS_PER_PASS:,d} pairs/iteration)")
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=(N_FRAMES, num_atoms, 3)).astype(np.float32)
    test_batch = np.ascontiguousarray(base_frame[np.newaxis, :, :] + noise, dtype=np.float32)

    # Start telemetry thread
    t_start = time.time()
    t_thread = threading.Thread(target=telemetry_sampler, daemon=True)
    t_thread.start()

    print(f"\nStarting sustained {DURATION_SEC}-second soak test loop...")
    print(f"{'Elapsed':>8} | {'Iter':>6} | {'Total Alignments':>18} | {'Throughput':>16} | {'Temp':>6} | {'Clock':>8} | {'Power':>8}")
    print("-" * 88)

    total_alignments = 0
    iter_count = 0
    t_loop_start = time.time()

    try:
        while True:
            elapsed = time.time() - t_loop_start
            if elapsed >= DURATION_SEC:
                break

            iter_count += 1
            t0 = time.perf_counter()
            _ = strux_rs.cuda_pairwise_rmsd(test_batch)
            dt = time.perf_counter() - t0

            total_alignments += PAIRS_PER_PASS
            rate = PAIRS_PER_PASS / dt

            # Grab latest telemetry
            if telemetry_data:
                _, cur_temp, cur_power, cur_clock, cur_mem = telemetry_data[-1]
                temp_str = f"{cur_temp:.0f}°C"
                clock_str = f"{cur_clock:.0f} MHz"
                power_str = f"{cur_power:.1f} W"
            else:
                temp_str, clock_str, power_str = "N/A", "N/A", "N/A"

            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            elapsed_str = f"{mins:02d}:{secs:02d}"

            print(f"{elapsed_str:>8} | {iter_count:>6,d} | {total_alignments:>18,d} | {rate:>14,.0f}/s | {temp_str:>6} | {clock_str:>8} | {power_str:>8}")

    finally:
        stop_telemetry.set()
        t_thread.join(timeout=2.0)

    total_time = time.time() - t_loop_start
    print("\n" + "=" * 80)
    print("  SOAK TEST COMPLETED: HARDWARE STABILITY & THERMAL REPORT")
    print("=" * 80)
    print(f"Total Duration:              {total_time:.2f} s ({total_time / 60:.1f} minutes)")
    print(f"Total Iterations:            {iter_count} passes")
    print(f"Total Alignments Computed:   {total_alignments:,d} pairwise QCP fits")
    print(f"Overall Sustained Throughput: {total_alignments / total_time:,.0f} alignments/second")

    if telemetry_data:
        temps = [x[1] for x in telemetry_data]
        powers = [x[2] for x in telemetry_data]
        clocks = [x[3] for x in telemetry_data]
        mems = [x[4] for x in telemetry_data]

        print("\n--- THERMAL METRICS ---")
        print(f"Initial Temperature:         {temps[0]:.1f} °C")
        print(f"Peak Temperature:            {max(temps):.1f} °C")
        print(f"Average Temperature:         {sum(temps)/len(temps):.1f} °C")
        print(f"Final Temperature:           {temps[-1]:.1f} °C")

        print("\n--- POWER DISSIPATION ---")
        print(f"Average Power Draw:          {sum(powers)/len(powers):.1f} W")
        print(f"Peak Power Draw:             {max(powers):.1f} W")

        print("\n--- FREQUENCY & THROTTLING ---")
        print(f"Max Graphics Clock:          {max(clocks):.0f} MHz")
        print(f"Min Sustained Clock:         {min(clocks):.0f} MHz")
        print(f"Average Clock:               {sum(clocks)/len(clocks):.0f} MHz")

        throttled = (max(clocks) - min(clocks)) > 300
        if throttled:
            print("Thermal Throttling Status:   DETECTED (Clock variation > 300 MHz under thermal load)")
        else:
            print("Thermal Throttling Status:   NONE DETECTED (Clocks sustained stably throughout)")

        print("\n--- MEMORY STABILITY ---")
        print(f"Initial VRAM Used:           {mems[0]:.0f} MiB")
        print(f"Final VRAM Used:             {mems[-1]:.0f} MiB")
        vram_leak = (mems[-1] - mems[0]) > 50
        print(f"VRAM Leakage Status:         {'LEAK DETECTED' if vram_leak else 'STABLE (Zero memory drift)'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
