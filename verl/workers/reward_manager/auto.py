REWARD_MANAGER_MAPPING = {
    "naive": ("naive", "NaiveRewardManager"),
    "prime": ("prime", "PrimeRewardManager"),
    "batch": ("batch", "BatchRewardManager"),
    "dapo": ("dapo", "DAPORewardManager"),
}

import os
import importlib
from verl.utils.import_utils import load_extern_type

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
            module_name, class_name = REWARD_MANAGER_MAPPING[reward_manager_name_or_path]
            module = importlib.import_module(f".{module_name}", package="verl.workers.reward_manager")
            return getattr(module, class_name)
        elif os.path.exists(reward_manager_name_or_path):
            # We assume the custom reward manager is defined as a class named "RewardManager" in the file,
            # Add an arg to specify the class name?
            # Test custom reward manager: "/root/verl/custom_reward_manager.py"
            # Wait to write a test case for this
            return load_extern_type(reward_manager_name_or_path, "RewardManager")
        else:
            raise ValueError(
                f"Invalid reward manager name or path: {reward_manager_name_or_path}"
                f"We have implemented the following reward managers: {REWARD_MANAGER_MAPPING.keys()}"
                f"If you want to use a custom reward manager, please reference to the documentation for more details.")
