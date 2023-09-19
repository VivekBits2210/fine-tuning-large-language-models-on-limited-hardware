from typing import List
import sys

from config.user_configuration import UserConfiguration

LOCAL_PATH = lambda net_id: f"/home/{net_id}/.local/lib/python3.8/site-packages"


class PackagePathManager:
    INTEL_PATH = "/share/apps/python/3.8.6/intel/lib/python3.8/site-packages"

    def __init__(self, user_config: UserConfiguration):
        self.net_id = user_config.net_id
        self.local_path = LOCAL_PATH(self.net_id)

    def fetch_package_paths(self) -> List[str]:
        return [self.local_path, self.INTEL_PATH]

    def add_package_paths_to_system(self) -> None:
        paths = self.fetch_package_paths()
        for index, path in enumerate(paths):
            sys.path.insert(index, path)
