REWARD_MANAGER_MAPPING = {
    "naive": ("naive", "NaiveRewardManager"),
    "prime": ("prime", "PrimeRewardManager"),
    "batch": ("batch", "BatchRewardManager"),
    "dapo": ("dapo", "DAPORewardManager"),
}

import logging
import os
logger = logging.getLogger(__name__)

class AutoRewardManager:

    def __init__(self):
        raise TypeError("AutoRewardManager is designed to be instantiated "
                        "using the `AutoRewardManager.from_name_or_custom_cls(reward_manager_name_or_custom_cls)` method.")

    @classmethod
    def from_name_or_custom_cls(cls, reward_manager_name_or_custom_cls: dict):
        """
        Args:
            reward_manager_name_or_custom_cls: the name of the reward manager or the custom class to the reward manager
        """
        if isinstance(reward_manager_name_or_custom_cls, str):
            if reward_manager_name_or_custom_cls in REWARD_MANAGER_MAPPING:
                import importlib
                # maybe we can lazy load the module here like transformers
                module_name, class_name = REWARD_MANAGER_MAPPING[reward_manager_name_or_custom_cls]
                module = importlib.import_module(f".{module_name}", package="verl.workers.reward_manager")
                return getattr(module, class_name)
            else:
                raise ValueError(f"Invalid reward manager name: {reward_manager_name_or_custom_cls}, "
                                 f"we have implemented the following reward managers: {REWARD_MANAGER_MAPPING.keys()}")
            
        elif reward_manager_name_or_custom_cls.custom_cls.get("path", None) is not None \
                or reward_manager_name_or_custom_cls.custom_cls.get("name", None) is not None:
            if reward_manager_name_or_custom_cls.custom_cls.get("path", None) is None \
                or reward_manager_name_or_custom_cls.custom_cls.get("name", None) is None:
                raise ValueError(f"Please provide both a valid custom reward manager path and name")
            from verl.utils.import_utils import load_extern_type
            try:
                return load_extern_type(reward_manager_name_or_custom_cls.custom_cls.path, reward_manager_name_or_custom_cls.custom_cls.name)
            except Exception as e:
                raise ValueError(f"Failed to load custom reward manager: {e}")
        else:
            logger.log(msg=f"No reward manager is provided, using the default reward manager: 'naive'", level=logging.WARNING)
            import importlib
            # maybe we can lazy load the module here like transformers
            module_name, class_name = REWARD_MANAGER_MAPPING["naive"]
            module = importlib.import_module(f".{module_name}", package="verl.workers.reward_manager")
            return getattr(module, class_name)
