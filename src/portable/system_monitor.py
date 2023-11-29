from psutil import Process
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


class SystemMonitor:
    def __init__(self):
        # Initialize NVML for GPU monitoring
        self.nvml_initialized = SystemMonitor._initialize_nvml()

    @classmethod
    def _initialize_nvml(cls):
        try:
            nvmlInit()
            return True
        except Exception as e:
            print(f"Error initializing NVML: {e}")
            return False

    def get_ram_usage(self):
        return Process().memory_info().rss / (1024 * 1024)

    def get_gpu_memory_usage(self):
        if not self.nvml_initialized:
            print("NVML not initialized.")
            return None

        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(handle)
            return info.used // 1024**3
        except Exception as e:
            print(f"Error retrieving GPU memory info: {e}")
            return None

    def get_gpu_utilization(self):
        gpu_memory = self.get_gpu_memory_usage()
        return gpu_memory
