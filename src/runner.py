import logging

from config.user_configuration import UserConfiguration
from config.log_configuration import LogConfiguration
from config.torch_configuration import TorchConfiguration
from os_environment_manager import OSEnvironmentManager
from package_path_manager import PackagePathManager
from system_monitor import SystemMonitor

ENV = "pre_prod"
NET_ID = "vgn2004"
OS_ENV_DICT = {
    "CUDA_VISIBLE_DEVICES": 0,
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
    "TORCHDYNAMO_DISABLE": 1
}

if __name__ == "__main__":
    # Configure the logger
    LogConfiguration.setup_logging()
    logger = logging.getLogger(__name__)

    # Get initial RAM and GPU utilization
    monitor = SystemMonitor()
    logger.info(f"RAM Usage: {monitor.get_ram_usage()} MB")
    logger.info(f"GPU Utilization: {monitor.get_gpu_utilization()} MB")

    # Setup folder/file path related configurations
    user_config = UserConfiguration(net_id=NET_ID, env=ENV)

    # Add Python packages to sys path
    package_path_manager = PackagePathManager(user_config)
    package_path_manager.add_package_paths_to_system()

    # Add environment variables to OS env
    os_env_manager = OSEnvironmentManager()
    os_env_manager.update_from_dict(OS_ENV_DICT)

    # Setup and commit global torch configurations
    torch_config = TorchConfiguration()
    torch_config.commit()


