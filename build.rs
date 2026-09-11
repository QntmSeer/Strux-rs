// ponytail: build script for compiling or copying PTX when the cuda feature is enabled.

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/kernels/qcp.cu");
    println!("cargo:rerun-if-changed=src/cuda/kernels/qcp.ptx");

    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::fs;
        use std::path::PathBuf;
        use std::process::Command;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let out_ptx = out_dir.join("qcp.ptx");
        let cu_path = PathBuf::from("src/cuda/kernels/qcp.cu");
        let precompiled_ptx = PathBuf::from("src/cuda/kernels/qcp.ptx");

        // 1. If precompiled PTX exists, copy it directly to OUT_DIR
        if precompiled_ptx.exists() {
            fs::copy(&precompiled_ptx, &out_ptx).expect("Failed to copy precompiled PTX");
            println!("cargo:warning=Using precompiled qcp.ptx");
            return;
        }

        // 2. Otherwise try compiling via nvcc
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg("--gpu-architecture=compute_75")
            .arg("-o")
            .arg(&out_ptx)
            .arg(&cu_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Successfully compiled qcp.cu to PTX via nvcc");
            }
            _ => {
                panic!("Neither precompiled qcp.ptx nor nvcc compiler found!");
            }
        }
    }
}
