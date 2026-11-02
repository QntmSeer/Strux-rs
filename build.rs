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

        // 1. Try compiling via nvcc first so edits to qcp.cu take effect; fallback to precompiled PTX
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
                let _ = fs::copy(&out_ptx, &precompiled_ptx);
                println!("cargo:warning=Successfully compiled qcp.cu to PTX via nvcc");
            }
            _ if precompiled_ptx.exists() => {
                fs::copy(&precompiled_ptx, &out_ptx).expect("Failed to copy precompiled PTX");
                println!("cargo:warning=Using precompiled qcp.ptx");
            }
            _ => {
                panic!("Neither nvcc compiler nor precompiled qcp.ptx found!");
            }
        }
    }
}
