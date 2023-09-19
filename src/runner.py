from config.user_configuration import UserConfiguration
from config.log_configuration import LogConfiguration
from os_environment_manager import OSEnvironmentManager
from package_path_manager import PackagePathManager

LogConfiguration.setup_logging()

ENV = "pre_prod"
NET_ID = "vgn2004"
OS_ENV_DICT = {
    "CUDA_VISIBLE_DEVICES": 0,
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
    "TORCHDYNAMO_DISABLE": 1
}


class HPCRunner:
    def __init__(self, net_id, env, os_env_dict):
        self.net_id = net_id
        self.env = env
        self.os_env_dict = os_env_dict

        self.user_config = None

    def configure(self):
        self.user_config = UserConfiguration(net_id=self.net_id, env=self.env)

        package_path_manager = PackagePathManager(self.user_config)
        package_path_manager.add_package_paths_to_system()

        os_env_manager = OSEnvironmentManager()
        os_env_manager.update_from_dict(self.os_env_dict)


