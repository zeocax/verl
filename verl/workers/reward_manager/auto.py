REWARD_MANAGER_MAPPING = {
    "naive": "NaiveRewardManager",
    "prime": "PrimeRewardManager",
    "batch": "BatchRewardManager",
    "dapo": "DAPORewardManager",
}

import os
import importlib


class AutoRewardManager:

    def __init__(self):
        raise TypeError("AutoRewardManager is designed to be instantiated "
                        "using the `AutoRewardManager.from_name_or_path(reward_manager_name_or_path)` method.")

    @classmethod
    def from_name_or_path(cls, reward_manager_name_or_path: str):
        """
        Args:
            reward_manager_name_or_path: the name of the reward manager or the path to the reward manager file
        """
        if reward_manager_name_or_path in REWARD_MANAGER_MAPPING:
            # maybe we can lazy import the module here like transformers
            reward_manager_name = REWARD_MANAGER_MAPPING[reward_manager_name_or_path]
            if reward_manager_name == "BatchRewardManager":
                from .batch import BatchRewardManager
                return BatchRewardManager
            elif reward_manager_name == "NaiveRewardManager":
                from .naive import NaiveRewardManager
                return NaiveRewardManager
            elif reward_manager_name == "PrimeRewardManager":
                from .prime import PrimeRewardManager
                return PrimeRewardManager
            elif reward_manager_name == "DAPORewardManager":
                from .dapo import DAPORewardManager
                return DAPORewardManager
        elif os.path.exists(reward_manager_name_or_path):
            spec = importlib.util.spec_from_file_location("RewardManager", reward_manager_name_or_path)
            module = importlib.util.module_from_spec(spec)
            return module
        else:
            raise ValueError(
                f"Invalid reward manager name or path: {reward_manager_name_or_path}"
                f"We have implemented the following reward managers: {REWARD_MANAGER_MAPPING.keys()}"
                f"If you want to use a custom reward manager, please reference to the documentation for more details.")
