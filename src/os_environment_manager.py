import os
from typing import Dict, Any


class OSEnvironmentManager:
    def __init__(self) -> None:
        self.env = dict(os.environ)

    def get_var(self, key) -> str:
        return self.env.get(key)

    def set_var_lazy(self, key: str, value: Any) -> None:
        self.env[key] = str(value)

    def delete_var_lazy(self, key: str) -> None:
        if key in self.env:
            del self.env[key]

    def commit_changes(self) -> None:
        for key, value in self.env.items():
            os.environ[key] = value

    def update_from_dict(self, mapping: Dict[str, Any]) -> None:
        for key, value in mapping.items():
            self.set_var_lazy(key, value)
        self.commit_changes()

    def __repr__(self):
        return "\n".join([f"{key}={value}" for key, value in self.env.items()])

    def __str__(self):
        return self.__repr__()
