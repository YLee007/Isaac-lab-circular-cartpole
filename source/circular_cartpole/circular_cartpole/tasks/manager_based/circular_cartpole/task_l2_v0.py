# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


##
# Pre-defined configs
##

from .circular_cartpole_robot_cfg import (
    CIRCULAR_CARTPOLE_ROBOT_L2_CFG,
    upward_pos,
    joint_effort_scale,
)

##
# Scene definition
##

fixed_pole_target_pos = upward_pos
fixed_pole_index = 1
flex_1_pole_target_pos = upward_pos
flex_1_pole_index = 2
flex_2_pole_target_pos = upward_pos
flex_2_pole_index = 3


@configclass
class CircularCartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = CIRCULAR_CARTPOLE_ROBOT_L2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["base_to_fixed"],
        scale=joint_effort_scale,
        clip={"base_to_fixed": (-3.0 * joint_effort_scale, 3.0 * joint_effort_scale)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        body_pos = ObsTerm(func=mdp.rk_get_body_link_quat_w)
        body_ang_vel = ObsTerm(func=mdp.rk_get_body_link_ang_vel_w)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_fixed_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["base_to_fixed"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )

    reset_flex_1_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["fixed_to_flex_1"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )

    reset_flex_2_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["flex_1_to_flex_2"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alignment_fixed_reward = RewTerm(
        func=mdp.rk_pole_rotation_reward_multi,
        weight=0.2,
        params={
            "linear_weight": 8,
            "exp_weight": 2,
            "pole_index_list": [fixed_pole_index],
            "target_pos_list": [fixed_pole_target_pos],
        },
    )
    alignment_flex_1_reward = RewTerm(
        func=mdp.rk_pole_rotation_reward_multi,
        weight=0.6,
        params={
            "linear_weight": 8,
            "exp_weight": 2,
            "pole_index_list": [flex_1_pole_index],
            "target_pos_list": [flex_1_pole_target_pos],
        },
    )
    alignment_flex_2_reward = RewTerm(
        func=mdp.rk_pole_rotation_reward_multi,
        weight=0.6,
        params={
            "linear_weight": 8,
            "exp_weight": 2,
            "pole_index_list": [flex_2_pole_index],
            "target_pos_list": [flex_2_pole_target_pos],
        },
    )

    action_magnitude = RewTerm(
        func=mdp.rk_action_l2,
        weight=-0.01,
    )
    action_smoothness = RewTerm(
        func=mdp.rk_action_rate_l2,
        weight=-0.01,
    )

    timeout_pos_reward = RewTerm(
        func=mdp.rk_timeout_pos_reward,
        weight=100,
        params={
            "pole_index_list": [
                fixed_pole_index,
                flex_1_pole_index,
                flex_2_pole_index,
            ],
            "target_pos_list": [
                fixed_pole_target_pos,
                flex_1_pole_target_pos,
                flex_2_pole_target_pos,
            ],
            "linear_weight": 8,
            "exp_weight": 2,
        },
    )
    timeout_vel_reward = RewTerm(
        func=mdp.rk_timeout_vel_reward,
        weight=-0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "base_to_fixed",
                    "fixed_to_flex_1",
                    "flex_1_to_flex_2",
                ],
            )
        },
    )


def update_target_weight_func(env, env_ids, data, target_weight, num_steps):
    if env.common_step_counter > num_steps:
        return target_weight
    return mdp.modify_term_cfg.NO_CHANGE


def configure_update_target_weight_params(address, func, target_weight, num_steps):
    return {
        "address": address,
        "modify_fn": func,
        "modify_params": {
            "target_weight": target_weight,
            "num_steps": num_steps,
        },
    }


@configclass
class CurriculumCfg:
    """Curriculum configuration for the MDP."""

    fixed_pole_weight_schedule_1 = CurriculumTerm(
        func=mdp.modify_term_cfg,
        params=configure_update_target_weight_params(
            "rewards.alignment_fixed_reward.weight",
            update_target_weight_func,
            0.4,
            20000,
        ),
    )

    fixed_pole_weight_schedule_2 = CurriculumTerm(
        func=mdp.modify_term_cfg,
        params=configure_update_target_weight_params(
            "rewards.alignment_fixed_reward.weight",
            update_target_weight_func,
            0.6,
            40000,
        ),
    )

    action_magnitude_weight_schedule_1 = CurriculumTerm(
        func=mdp.modify_term_cfg,
        params=configure_update_target_weight_params(
            "rewards.action_magnitude.weight",
            update_target_weight_func,
            -1,
            60000,
        ),
    )
    action_smoothness_weight_schedule_1 = CurriculumTerm(
        func=mdp.modify_term_cfg,
        params=configure_update_target_weight_params(
            "rewards.action_smoothness.weight",
            update_target_weight_func,
            -1,
            60000,
        ),
    )

    timeout_vel_reward_weight_schedule_1 = CurriculumTerm(
        func=mdp.modify_term_cfg,
        params=configure_update_target_weight_params(
            "rewards.timeout_vel_reward.weight",
            update_target_weight_func,
            -40,
            60000,
        ),
    )


##
# Environment configuration
##


@configclass
class CircularCartpoleEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: CircularCartpoleSceneCfg = CircularCartpoleSceneCfg(
        num_envs=4096, env_spacing=1.5
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
