"""Configuration space of a robot model.

The :class:`Configuration` class encapsulates a MuJoCo
`model <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjmodel>`__
and `data <https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjdata>`__,
offering easy access to frame transforms and frame Jacobians. A frame refers to a coordinate
system that can be attached to various parts of the robot, such as a body, geom, or site.
"""

import logging
from typing import Optional

import mujoco
import numpy as np

from . import constants as consts
from . import exceptions
from .lie import SE3, SO3

# Genesis
import taichi as ti
from genesis.repr_base import RBC
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from genesis import CTRL_MODE as GS_CTRL_MODE


class Configuration:
    """Encapsulates a model and data for convenient access to kinematic quantities.

    This class provides methods to access and update the kinematic quantities of a robot
    model, such as frame transforms and Jacobians. It performs forward kinematics at every
    time step, ensuring up-to-date information about the robot's state.

    Key functionalities include:

    * Running forward kinematics to update the state.
    * Checking configuration limits.
    * Computing Jacobians for different frames.
    * Retrieving frame transforms relative to the world frame.
    * Integrating velocities to update configurations.
    """

    def __init__(
        self,
        model: mujoco.MjModel
    ):
        """Constructor.

        Args:
            model: Mujoco model.
            q: Configuration to initialize from. If None, the configuration is
                initialized to the default configuration `qpos0`.
        """
        self.model = model

    def check_limits(self, entity: RigidEntity, tol: float = 1e-6, safety_break: bool = True) -> None:
        """Check that the current configuration is within bounds.

        Args:
            tol: Tolerance in [rad].
            safety_break: If True, stop execution and raise an exception if the current
                configuration is outside limits. If False, print a warning and continue
                execution.
        """
        for jnt in range(self.model.njnt):
            jnt_type = self.model.jnt_type[jnt]
            if (
                jnt_type == mujoco.mjtJoint.mjJNT_FREE
                or not self.model.jnt_limited[jnt]
            ):
                continue
            padr = self.model.jnt_qposadr[jnt]
            qval = entity.get_qpos().cpu().numpy()[padr]
            qmin = self.model.jnt_range[jnt, 0]
            qmax = self.model.jnt_range[jnt, 1]
            if qval < qmin - tol or qval > qmax + tol:
                if safety_break:
                    raise exceptions.NotWithinConfigurationLimits(
                        joint_id=jnt,
                        value=qval,
                        lower=qmin,
                        upper=qmax,
                        model=self.model,
                    )
                else:
                    logging.warning(
                        f"Value {qval:.2f} at index {jnt} is outside of its limits: "
                        f"[{qmin:.2f}, {qmax:.2f}]"
                    )

    def get_frame_jacobian(self, entity: RigidEntity, link: RigidLink) -> np.ndarray:
        r"""Compute the Jacobian matrix of a frame velocity.

        Denoting our frame by :math:`B` and the world frame by :math:`W`, the
        Jacobian matrix :math:`{}_B J_{WB}` is related to the body velocity
        :math:`{}_B v_{WB}` by:

        .. math::

            {}_B v_{WB} = {}_B J_{WB} \dot{q}

        Args:
            frame_name: Name of the frame in the MJCF.
            frame_type: Type of frame. Can be a geom, a body or a site.

        Returns:
            Jacobian :math:`{}_B J_{WB}` of the frame.
        """

        jac = entity.get_jacobian(link).cpu().numpy()

        link_quat = link.get_quat().cpu().numpy()
        # Genesis jacobians have a frame of reference centered at the local frame but
        # aligned with the world frame. To obtain a jacobian expressed in the local
        # frame, aka body jacobian, we need to left-multiply by A[T_fw].
        R_wf = SO3(wxyz=link_quat)
        A_fw = SE3.from_rotation(R_wf.inverse()).adjoint()
        jac = A_fw @ jac

        return jac

    def get_transform_frame_to_world(self, entity: RigidEntity | RBC, frame_name: Optional[str] = None) -> SE3:
        """Get the pose of a frame at the current configuration.

        Args:
            frame_name: Name of the frame in the MJCF.
        Returns:
            The pose of the frame in the world frame.
        """
        frame = entity.get_link(frame_name) if isinstance(entity, RigidEntity) and frame_name else entity
        return SE3.from_rotation_and_translation(
            rotation=SO3(wxyz=frame.get_quat().cpu().numpy()),
            translation=frame.get_pos().cpu().numpy(),
        )

    def get_transform(
        self,
        entity: RigidEntity | RBC,
        base: RigidEntity | RBC
    ) -> SE3:
        """Get the pose of a frame with respect to another frame at the current
        configuration.

        Args:
            entity: Entity | RBC
            base: Entity | RBC

        Returns:
            The pose of `source_name` in `dest_name`.
        """
        transform_source_to_world = self.get_transform_frame_to_world(entity)
        transform_dest_to_world = self.get_transform_frame_to_world(base)
        return transform_dest_to_world.inverse() @ transform_source_to_world

    def integrate(self, entity: RigidEntity, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate a velocity starting from the current configuration.

        Args:
            velocity: The velocity in tangent space.
            dt: Integration duration in [s].

        Returns:
            The new configuration after integration.
        """
        q = entity.get_qpos().cpu().numpy().astype('float64')
        mujoco.mj_integratePos(self.model, q, velocity, dt)
        return q

    def apply_ctrl(self, entity: RigidEntity, velocity: np.ndarray, dt: float,
                   force: Optional[np.ndarray] = None,
                   ctrl_type: Optional[GS_CTRL_MODE] = GS_CTRL_MODE.VELOCITY) -> np.ndarray:
        """Integrate a velocity and update the current configuration inplace.

        Args:
            arm_dof: Number of arm joints.
            hand_dof: Number of hand joints.
            velocity: The velocity in tangent space.
            dt: Integration duration in [s].
        """
        if ctrl_type == GS_CTRL_MODE.POSITION:
            # position-control
            q = self.integrate(entity, velocity, dt)
            entity.control_dofs_position(q)
        elif ctrl_type == GS_CTRL_MODE.VELOCITY:
            # velocity-control
            entity.control_dofs_velocity(velocity)
        elif ctrl_type == GS_CTRL_MODE.FORCE:
            # force-control
            entity.control_dofs_force(force)

    @property
    def nv(self) -> int:
        """The dimension of the tangent space."""
        return self.model.nv

    @property
    def nq(self) -> int:
        """The dimension of the configuration space."""
        return self.model.nq
