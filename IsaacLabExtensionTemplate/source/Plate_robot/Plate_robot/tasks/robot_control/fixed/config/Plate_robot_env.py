from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

# 环境配置类
@configclass
class VolleyballPlateEnvCfg(DirectRLEnvCfg):
    # 环境配置：包括仿真设置，动作空间，观测空间等
    episode_length_s = 8.3333  # 每集时长
    decimation = 2  # 仿真加速
    action_space = 6  # 6个电机驱动，动作空间
    observation_space = 23  # 观测空间的维度
    state_space = 0

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 时间步长
        render_interval=decimation,  # 渲染间隔
        disable_contact_processing=True,  # 禁用接触处理
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # 机器人配置
    robot = ArticulationCfg(
        prim_path="/World/volleyball_plate_0_1/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/chenzk/robotcon/volleyball_plate_0.1.usd",  # 机器人模型路径
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={  # 初始化关节位置
                "first_arm_joint_01": 0.0,
                "first_arm_joint_02": 0.0,
                "first_arm_joint_03": 0.0,
                "motor_joint_01": 0.0,
                "motor_joint_02": 0.0,
                "motor_joint_03": 0.0,
            },
            pos=(1.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "first_arm": ImplicitActuatorCfg(
                joint_names_expr=["first_arm_joint_01", "first_arm_joint_02", "first_arm_joint_03"],  # 控制三个臂部关节
                effort_limit=50.0,
                velocity_limit=2.5,
                stiffness=80.0,
                damping=4.0,
            ),
            "motor": ImplicitActuatorCfg(
                joint_names_expr=["motor_joint_01", "motor_joint_02", "motor_joint_03"],  # 控制三个电机关节
                effort_limit=50.0,
                velocity_limit=2.5,
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )

    # 观测空间和奖励函数
    action_scale = 7.5
    dof_velocity_scale = 0.1

    dist_reward_scale = 1.5
    rot_reward_scale = 1.5
    action_penalty_scale = 0.05
    finger_reward_scale = 2.0

# 环境类定义，继承自 DirectRLEnv
class VolleyballPlateEnv(DirectRLEnv):
    cfg: VolleyballPlateEnvCfg

    def __init__(self, cfg: VolleyballPlateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 获取环境局部坐标的函数
        def get_env_local_pose(env_pos: torch.Tensor, xformable: UsdGeom.Xformable, device: torch.device):
            """计算在环境局部坐标系下的位姿"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 初始化机器人关节的上下限
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # 初始化速度缩放比例
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # 定义机器人关节目标位置
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)

        stage = get_current_stage()
        # 获取机器人和排球的位姿（此处示意，排球资产未设计）
        robot_pose = get_env_local_pose(self.scene.env_origins[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/Robot/robot_link")), self.device)

        # 排球的观测（尚未设计）
        # to_target = volleyball_pos - robot_grasp_pos  # 计算机器人托盘和排球的相对位置

    # 预物理步骤：控制机器人的动作
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * self.cfg.action_scale
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    # 执行动作
    def _apply_action(self):
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # 获取终止条件
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._robot.data.joint_pos[:, 3] > 0.39  # 终止条件示意
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # 计算奖励
    def _get_rewards(self) -> torch.Tensor:
        # 计算机器人和排球之间的相对距离等
        rewards = self._compute_rewards(self.actions)
        return rewards

    # 重置环境状态
    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # 重置机器人和排球的位置
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125, 0.125, (len(env_ids), self._robot.num_joints), self.device
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 获取观测值
    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        # 排球的相对位置和速度（示意代码，排球未设计）
        # to_target = volleyball_pos - robot_grasp_pos

        # 拼接观测数据
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                # 这里可以拼接排球和机器人的相对速度、加速度等
                # to_target,  
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # 辅助函数：计算奖励等
    def _compute_rewards(self, actions):
        # 计算机器人和排球的相对距离、角度等
        d = torch.norm(self.robot_grasp_pos - self.volleyball_pos, p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward

        # 更多奖励计算可以根据排球的状态来修改
        rewards = dist_reward  # 示意，只计算了距离奖励
        return rewards
