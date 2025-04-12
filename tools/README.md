
## Notes on deployment

The script will produce one output file: `system_info.json`. This file contains information about the system, including the GPU type and other relevant details.

### NVIDIA Systems

Make sure that CUDA module is loaded. We rely on `nvidia-smi` availability.

### AMD Systems

We rely on `rocminfo` to query GPU type. Make sure that proper modules are loaded.

On our system, `rocminfo` is not immediately available. On our system, it is not available in `PATH` but we have an environment variable that needs to be used:

```
export PATH=${ROCMINFO_PATH}/bin/:${PATH}
```

### Intel Systems

We rely on `xpu-smi` to query GPU type. Make sure that proper modules are loaded.
