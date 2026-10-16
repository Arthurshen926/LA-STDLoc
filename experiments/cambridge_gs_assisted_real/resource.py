"""Single physical GPU1 UUID bootstrap; no scientific imports in this module."""
import ctypes, json, os, subprocess, uuid
from pathlib import Path
PHYSICAL_UUID="GPU-1f21bf0e-0ae0-aad5-c30e-653ee1e13775"

def physical_context_uuid():
    cuda=ctypes.CDLL("libcuda.so.1");ordinal=ctypes.c_int()
    if cuda.cuCtxGetDevice(ctypes.byref(ordinal))!=0:raise ValueError("cuCtxGetDevice failed")
    value=(ctypes.c_ubyte*16)()
    fn=getattr(cuda,"cuDeviceGetUuid_v2",cuda.cuDeviceGetUuid)
    if fn(ctypes.byref(value),ordinal)!=0:raise ValueError("cuDeviceGetUuid failed")
    return "GPU-"+str(uuid.UUID(bytes=bytes(value)))

def assert_gpu(device):
    import torch
    if device!="cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES")!=PHYSICAL_UUID:
        raise ValueError("single UUID logical alias differs")
    if torch.cuda.device_count()!=1 or torch.cuda.current_device()!=0 or physical_context_uuid()!=PHYSICAL_UUID:
        raise ValueError("CUDA physical context differs")

def initialize():
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=PHYSICAL_UUID or torch.cuda.is_initialized():
        raise ValueError("single UUID must be set before initialization")
    output=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"],
        capture_output=True,text=True,check=True,timeout=15).stdout
    if any(line.split(",")[0].strip()==PHYSICAL_UUID for line in output.splitlines()):
        raise ValueError("GPU1 has an existing compute process")
    torch.cuda.set_device(0)
    pinned=torch.empty((32,),pin_memory=True);explicit=torch.zeros((32,),device="cuda:0")
    torch.cuda.synchronize(0);assert_gpu("cuda:0")
    return dict(pid=os.getpid(),device="cuda:0",device_count=1,physical_uuid=physical_context_uuid(),
        pinned=bool(pinned.is_pinned()),prior_nvml_apps=output,
        NSpid=[x for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("NSpid:")])
