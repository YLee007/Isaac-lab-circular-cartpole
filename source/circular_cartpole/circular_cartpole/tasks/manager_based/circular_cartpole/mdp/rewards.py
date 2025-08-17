# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from .observations import rk_get_pole_rotation_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rk_timeout_pos_reward(
    env: ManagerBasedRLEnv,
    linear_weight: float,
    exp_weight: float,
    pole_index_list: list[int],
    target_pos_list: list[tuple[float, float, float]],
) -> torch.Tensor:
    time_outs = env.termination_manager.time_outs

    if torch.any(time_outs):
        timeout_rewards = rk_pole_rotation_reward_multi(
            env,
            linear_weight=linear_weight,
            exp_weight=exp_weight,
            pole_index_list=pole_index_list,
            target_pos_list=target_pos_list,
        )
        return torch.where(
            time_outs, timeout_rewards, torch.zeros_like(timeout_rewards)
        )

    return torch.zeros_like(time_outs)


def rk_timeout_vel_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    time_outs = env.termination_manager.time_outs

    if torch.any(time_outs):
        # Handle NaN values in joint velocities
        joint_vels = asset.data.joint_vel[:, asset_cfg.joint_ids]
        joint_vels = torch.nan_to_num(joint_vels, nan=0.0, posinf=0.0, neginf=0.0)
        timeout_rewards = torch.sum(torch.abs(joint_vels), dim=1)
        return torch.where(
            time_outs, timeout_rewards, torch.zeros_like(timeout_rewards)
        )

    return torch.zeros_like(time_outs, dtype=torch.float32)


def rk_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    action_diff = env.action_manager.action - env.action_manager.prev_action
    # Handle NaN values by replacing them with zeros
    action_diff = torch.nan_to_num(action_diff, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sum(torch.square(action_diff), dim=1)


def rk_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    # Handle NaN values by replacing them with zeros
    actions = torch.nan_to_num(
        env.action_manager.action, nan=0.0, posinf=0.0, neginf=0.0
    )
    return torch.sum(torch.square(actions), dim=1)


def rk_pole_rotation_reward_multi(
    env: ManagerBasedRLEnv,
    linear_weight: float,
    exp_weight: float,
    pole_index_list: list[int],
    target_pos_list: list[tuple[float, float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for reaching the target rotation."""
    # Get all errors at once if possible, otherwise use lists
    linear_rewards = []
    exp_rewards = []

    for pole_index, target_pos in zip(pole_index_list, target_pos_list):
        error = rk_get_pole_rotation_error(env, pole_index, target_pos, asset_cfg)
        # Clamp error to avoid numerical issues
        error = torch.clamp(error, 0.0, torch.pi)

        # Linear reward component
        linear_reward = (torch.pi - error) / torch.pi

        # Exponential bonus for small errors
        exp_reward = torch.exp(-1.0 * error)

        linear_rewards.append(linear_reward)
        exp_rewards.append(exp_reward)

    # Stack and combine rewards
    linear_rewards_tensor = torch.stack(linear_rewards, dim=-1)
    exp_rewards_tensor = torch.stack(exp_rewards, dim=-1)

    # Apply weights and combine
    weighted_linear = linear_rewards_tensor * linear_weight
    weighted_exp = exp_rewards_tensor * exp_weight
    combined_rewards = weighted_linear + weighted_exp

    # Average across all poles
    final_reward = torch.mean(combined_rewards, dim=-1)
    return final_reward.squeeze(-1)
