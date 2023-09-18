import os
from typing import Dict, Any


def set_environment_variables(mapping: Dict[str, Any]) -> None:
    for key, value in mapping.items():
        os.environ[key] = str(value)
