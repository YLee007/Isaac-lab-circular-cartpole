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
    CIRCULAR_CARTPOLE_ROBOT_L1_CFG,
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


@configclass
class CircularCartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = CIRCULAR_CARTPOLE_ROBOT_L1_CFG.replace(
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
        asset_name="robot", joint_names=["base_to_fixed"], scale=joint_effort_scale
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
            "position_range": (-0.5 * math.pi, 0.5 * math.pi),
            "velocity_range": (-0.5 * math.pi, 0.5 * math.pi),
        },
    )

    reset_flex_1_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["fixed_to_flex_1"]),
            "position_range": (-0.5 * math.pi, 0.5 * math.pi),
            "velocity_range": (-0.5 * math.pi, 0.5 * math.pi),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alignment_reward = RewTerm(
        func=mdp.rk_pole_rotation_reward_multi,
        weight=1,
        params={
            "linear_weight": 10,
            "exp_weight": 0,
            "pole_index_list": [fixed_pole_index, flex_1_pole_index],
            "target_pos_list": [fixed_pole_target_pos, flex_1_pole_target_pos],
        },
    )
    vel_penalty_fixed = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_to_fixed"])},
    )
    vel_penalty_flex_1 = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["fixed_to_flex_1"])},
    )
    timeout_pos_reward = RewTerm(
        func=mdp.rk_timeout_pos_reward,
        weight=100,
        params={
            "pole_index_list": [fixed_pole_index, flex_1_pole_index],
            "target_pos_list": [fixed_pole_target_pos, flex_1_pole_target_pos],
            "linear_weight": 10,
            "exp_weight": 0,
        },
    )
    timeout_vel_reward = RewTerm(
        func=mdp.rk_timeout_vel_reward,
        weight=-10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["base_to_fixed", "fixed_to_flex_1"]
            )
        },
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

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 6
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
