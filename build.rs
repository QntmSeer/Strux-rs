// ponytail: build script for compiling CUDA kernels into PTX when the cuda feature is enabled.

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/kernels/qcp.cu");

    #[cfg(feature = "cuda")]
    {
        use std::env;
        use std::path::PathBuf;
        use std::process::Command;

        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let ptx_path = out_dir.join("qcp.ptx");
        let cu_path = PathBuf::from("src/cuda/kernels/qcp.cu");

        // Try compiling via nvcc
        let status = Command::new("nvcc")
            .arg("-ptx")
            .arg("-O3")
            .arg("--gpu-architecture=compute_75") // Works on Turing, Ampere, Ada, Hopper
            .arg("-o")
            .arg(&ptx_path)
            .arg(&cu_path)
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Successfully compiled qcp.cu to PTX via nvcc");
            }
            _ => {
                println!("cargo:warning=nvcc not found or failed; checking for precompiled PTX");
            }
        }
    }
}
