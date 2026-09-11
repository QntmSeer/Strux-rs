# ponytail: compile qcp.cu directly to PTX using NVRTC without requiring a full nvcc install.
import os
import sys

def main():
    try:
        from cuda.bindings import nvrtc
    except Exception as e1:
        try:
            from cuda import nvrtc
        except Exception as e2:
            print(f"ImportError: {e1} / {e2}")
            return 1

    cu_path = "src/cuda/kernels/qcp.cu"
    ptx_path = "src/cuda/kernels/qcp.ptx"

    if not os.path.exists(cu_path):
        print(f"File not found: {cu_path}")
        return 1

    with open(cu_path, "rb") as f:
        src = f.read()

    # NVRTC doesn't need cuda_runtime.h header included
    src_cleaned = src.replace(b"#include <cuda_runtime.h>", b"")

    err, prog = nvrtc.nvrtcCreateProgram(src_cleaned, b"qcp.cu", 0, [], [])
    if err.value != 0:
        print(f"nvrtcCreateProgram error: {err}")
        return 1

    opts = [b"--gpu-architecture=compute_75", b"-O3"]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

    err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    log_buf = bytearray(log_size)
    err = nvrtc.nvrtcGetProgramLog(prog, log_buf)
    log_str = log_buf.decode("utf-8", errors="replace").strip()
    if log_str:
        print("Compiler log:\n", log_str)

    if res.value != 0:
        print(f"Compilation failed with code: {res}")
        return 1

    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx_buf = bytearray(ptx_size)
    err = nvrtc.nvrtcGetPTX(prog, ptx_buf)
    with open(ptx_path, "wb") as f:
        f.write(ptx_buf)

    print(f"SUCCESS: Compiled {cu_path} -> {ptx_path} ({len(ptx_buf)} bytes)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
