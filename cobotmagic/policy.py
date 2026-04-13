import os
import sys

import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

# Ensure imports from `src/` resolve when running scripts from `cobotmagic/`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.openpi.training import config as _config
from src.openpi.policies import policy_config
from src.openpi.shared import download

DEFAULT_POLICY_CONFIG_NAME = "pi0_mobile_aloha_lora_local"
DEFAULT_CHECKPOINT_DIR = "/workspace/project/openpi/checkpoints/pi0_mobile_aloha_local/mobile_aloha_lora/10000"


class Pi0Policy(nn.Module):
    def __init__(
        self,
        config_name: str = DEFAULT_POLICY_CONFIG_NAME,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    ):
        super().__init__()
        config = _config.get_config(config_name)
        checkpoint_dir = download.maybe_download(os.path.expanduser(checkpoint_dir))

        # Create a trained policy.
        self.model = policy_config.create_trained_policy(config, checkpoint_dir)
        self.optimizer = None

    def __call__(self, center_image, left_image, right_image, robot_state, task_prompt, 
                 actions=None, action_is_pad=None):
        model_inputs = {
            "state": robot_state,
            "images": {
                "cam_high": center_image,
                "cam_low": np.zeros_like(center_image),
                "cam_left_wrist": left_image,
                "cam_right_wrist": right_image,
            },
            "prompt": task_prompt,
        }
        a_hat = self.model.infer(model_inputs)['actions']
        return a_hat
        """
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(image, depth_image, robot_state, actions, action_is_pad)
            if self.loss_function == 'l1':
                mse = F.l1_loss(actions, a_hat)
            elif self.loss_function == 'l2':
                mse = F.mse_loss(actions, a_hat)
            else:
                mse = F.smooth_l1_loss(actions, a_hat)

            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict, a_hat
        else:  # inference time
            a_hat = self.model(image, depth_image, robot_state)  # no action, sample from prior
            return a_hat
        """
    
    def configure_optimizers(self):
        return None

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)
