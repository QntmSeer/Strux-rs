// ponytail: Zero-overhead dynamic NVTX range markers for NVIDIA Nsight Systems (nsys).
// Dynamically loads libnvtx3interop.so or libnvToolsExt.so if available; no-op otherwise.
use std::ffi::CString;
use std::sync::OnceLock;

type NvtxPushFn = unsafe extern "C" fn(*const std::os::raw::c_char) -> i32;
type NvtxPopFn = unsafe extern "C" fn() -> i32;

struct NvtxLib {
    _lib: libloading::Library,
    push: NvtxPushFn,
    pop: NvtxPopFn,
}

static NVTX: OnceLock<Option<NvtxLib>> = OnceLock::new();

fn get_nvtx() -> Option<&'static NvtxLib> {
    NVTX.get_or_init(|| {
        let names = ["nvtx3interop", "nvToolsExt", "nvToolsExt64_1"];
        for name in names {
            if let Ok(lib) = unsafe { libloading::Library::new(libloading::library_filename(name)) } {
                unsafe {
                    let push = lib.get::<NvtxPushFn>(b"nvtxRangePushA\0");
                    let pop = lib.get::<NvtxPopFn>(b"nvtxRangePop\0");
                    if let (Ok(push), Ok(pop)) = (push, pop) {
                        return Some(NvtxLib {
                            push: *push,
                            pop: *pop,
                            _lib: lib,
                        });
                    }
                }
            }
        }
        None
    })
    .as_ref()
}

pub struct NvtxRange {
    active: bool,
}

impl NvtxRange {
    #[inline]
    pub fn new(name: &str) -> Self {
        if let Some(nvtx) = get_nvtx() {
            if let Ok(c_str) = CString::new(name) {
                unsafe {
                    (nvtx.push)(c_str.as_ptr());
                }
                return Self { active: true };
            }
        }
        Self { active: false }
    }
}

impl Drop for NvtxRange {
    #[inline]
    fn drop(&mut self) {
        if self.active {
            if let Some(nvtx) = get_nvtx() {
                unsafe {
                    (nvtx.pop)();
                }
            }
        }
    }
}
