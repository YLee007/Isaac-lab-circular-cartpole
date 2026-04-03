# Isaac Lab 项目实践 - 圆周倒立摆（L1、L2、L3）


该项目主要展示了如何借助Isaac Lab的强化学习功能，实现一阶、二阶、三阶圆周倒立摆。项目采用Manager-Based的训练环境，训练策略选用SKRL PPO。

## 3D建模文件
路径为 `models`，由SolidWorks导出，包括URDF~~和USD格式~~文件。

## **一阶圆周倒立摆 V0**
- **目标姿态**：<br>
灰杆朝上、绿杆朝上
- **环境配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\task_l1_v0.py`
- **超参数配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\agents\skrl_ppo_cfg_l1_v0.yaml`
- **示意图**：<br>
  <img src="source\circular_cartpole\imgs\l1_v0.png" alt="一阶圆周倒立摆 V0" width="400" height="400" >


## **一阶圆周倒立摆 V1**
- **目标姿态**：<br>
灰杆朝上、绿杆朝下
- **环境配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\task_l1_v1.py`
- **超参数配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\agents\skrl_ppo_cfg_l1_v1.yaml`
- **示意图**：<br>
  <img src="source\circular_cartpole\imgs\l1_v1.png" alt="一阶圆周倒立摆 V1" width="400" height="400" >


## **二阶圆周倒立摆 V0**
- **目标姿态**：<br>
灰杆朝上、绿杆朝上、蓝杆朝上
- **环境配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\task_l2_v0.py`
- **超参数配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\agents\skrl_ppo_cfg_l2_v0.yaml`
- **示意图**：<br>
  <img src="source\circular_cartpole\imgs\l2_v0.png" alt="二阶圆周倒立摆 V0" width="400" height="400" >

## **三阶圆周倒立摆 V0**
- **目标姿态**：<br>
灰杆朝上、绿杆朝上、蓝杆朝上
- **环境配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\task_l3_v0.py`
- **超参数配置**：`source\circular_cartpole\circular_cartpole\tasks\manager_based\circular_cartpole\agents\skrl_ppo_cfg_l3_v0.yaml`
- **示意图**：<br>
  <img src="source\circular_cartpole\imgs\l3_v0.png" alt="三阶圆周倒立摆 V0" width="400" height="400" >
