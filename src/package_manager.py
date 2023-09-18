from typing import List
import sys


class PackageManager:
    INTEL_PATH = "/share/apps/python/3.8.6/intel/lib/python3.8/site-packages"

    def __init__(self, net_id: str):
        self.net_id = net_id
        self.local_path = self._local_path()

    def _local_path(self) -> str:
        return f"/home/{self.net_id}/.local/lib/python3.8/site-packages"

    def fetch_package_paths(self) -> List[str]:
        return [self.local_path, self.INTEL_PATH]

    def add_package_paths_to_system(self) -> None:
        paths = self.fetch_package_paths()
        for index, path in enumerate(paths):
            sys.path.insert(index, path)