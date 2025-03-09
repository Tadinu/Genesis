from dataclasses import dataclass
import time
import torch
from typing_extensions import Optional

# Numpy
import numpy as np

# MuJoCo
import mujoco as mj

# Genesis
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from genesis.system.base_system import BaseSystem
from genesis import CTRL_MODE as GS_CTRL_MODE

@dataclass(frozen=True)
class OSCConfig:
    # Cartesian impedance control gains
    impedance_pos = np.asarray([100.0, 100.0, 100.0])  # [N/m]
    impedance_ori = np.asarray([50.0, 50.0, 50.0])  # [Nm/rad]

    # Joint impedance control gains.
    Kp_null = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0])

    # Damping ratio for both Cartesian and joint impedance control
    damping_ratio = 1.0

    # Gains for the twist computation. These should be between 0 and 1. 0 means no
    # movement, 1 means move the end-effector to the target in one integration step.
    Kpos: float = 0.95

    # Gain for the orientation component of the twist computation. This should be
    # between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
    # orientation in one integration step.
    Kori: float = 0.95

    # Compute damping and stiffness matrices
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)

def get_mass_inv(robot: BaseSystem) -> np.ndarray:
    rigid_solver = robot.system.scene.sim.rigid_solver
    mass_mat = rigid_solver.get_mass_mat(dofs_idx=torch.tensor(robot.dof_ids))
    return mass_mat.inverse().cpu().numpy()

# Ref: [kevinzakka]-https://github.com/kevinzakka/mjctrl/blob/main/opspace.py
# https://github.com/Genesis-Embodied-AI/Genesis/blob/main/tests/utils.py
def control_osc(robot: BaseSystem, ee: RigidLink,
                ee_target: RigidEntity, q0,
                ctrl_mode: Optional[GS_CTRL_MODE] = GS_CTRL_MODE.POSITION,
                gravity_compensation: Optional[bool] = True,
                integration_dt = 1):
    model = robot.system_model
    nv = model.nv
    rigid_solver = robot.system.scene.sim.rigid_solver

    # Pre-allocate numpy arrays
    jac = np.zeros((6, nv))
    twist = np.zeros(6)
    ee_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((nv, nv))
    Mx = np.zeros((6, 6))

    # Spatial velocity (aka twist)
    ee_pos = ee.get_pos()
    ee_quat = ee.get_quat()
    target_pos = ee_target.get_pos()
    target_quat = ee_target.get_quat()
    dx = target_pos - ee_pos
    twist[:3] = OSCConfig.Kpos * dx / integration_dt
    mj.mju_negQuat(ee_quat_conj, ee_quat)
    mj.mju_mulQuat(error_quat, target_quat, ee_quat_conj)
    mj.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= OSCConfig.Kori / integration_dt

    # Jacobian
    jac = np.squeeze(robot.system.get_jacobian(ee).cpu().numpy())

    # Compute the task-space inertia matrix
    M_inv = get_mass_inv(robot)
    Mx_inv = jac @ M_inv @ jac.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    # Compute generalized forces
    qpos = robot.system.get_dofs_position().cpu().numpy()
    qvel = robot.system.get_dofs_velocity().cpu().numpy()
    tau = jac.T @ Mx @ (OSCConfig.Kp * twist - OSCConfig.Kd * (jac @ qvel))

    # Add joint task in nullspace.
    Jbar = M_inv @ jac.T @ Mx
    ddq = OSCConfig.Kp_null * (q0 - qpos) - OSCConfig.Kd_null * qvel
    tau += (np.eye(model.nv) - jac.T @ Jbar.T) @ ddq

    # Add gravity compensation
    if gravity_compensation:
        # NOTE: access [dofs_state] directly here while waiting for a @ti.kernel get of [qf_bias]
        tau += rigid_solver.dofs_state.qf_bias.to_numpy()[robot.dof_ids, 0]

    # Set the control signal and step the simulation
    np.clip(tau, *model.actuator_ctrlrange.T, out=tau)

    if ctrl_mode == GS_CTRL_MODE.POSITION:
        return qpos
    elif ctrl_mode == GS_CTRL_MODE.VELOCITY:
        return qvel
    else:
        return tau[robot.actuator_ids]