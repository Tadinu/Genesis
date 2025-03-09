"""Relative frame task implementation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from ..configuration import Configuration
from ..lie import SE3
from .exceptions import TargetNotSet, TaskDefinitionError
from .frame_task import FrameTask

# Genesis
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink


class RelativeFrameTask(FrameTask):
    """Regulate the pose of a frame relative to another frame.

    Attributes:
        frame: Entity of the frame to regulate, typically the name of body, geom
            or site in the robot model.
        base: RBC of the base frame the task is relative to.
        transform_target_to_base: Target pose in the base frame.
    """

    transform_target_to_base: Optional[SE3]

    def __init__(
        self,
        entity: RigidEntity,
        frame: RigidLink,
        base: RigidLink,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(entity=entity, frame=frame, position_cost=position_cost, orientation_cost=orientation_cost,
                         gain=gain, lm_damping=lm_damping)
        self.name = f"RelativeFrameTask_{entity.name}_{frame.name}_{base.name}",
        self.base = base
        self.transform_target_to_base = None

    def set_target(self, transform_target_to_base: SE3) -> None:
        """Set the target pose in the base frame.

        Args:
            transform_target_to_base: Transform from the task target frame to the
                base frame.
        """
        self.transform_target_to_base = transform_target_to_base.copy()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Set the target pose from a given robot configuration.

        Args:
            configuration: Robot configuration :math:`q`.
        """
        self.set_target(
            configuration.get_transform(self.frame, self.base)
        )

    def compute_error(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_base is None:
            raise TargetNotSet(self.__class__.__name__)

        transform_frame_to_base = configuration.get_transform(self.frame, self.base)
        return transform_frame_to_base.rminus(self.transform_target_to_base)

    def compute_jacobian(self, configuration: Configuration) -> np.ndarray:
        if self.transform_target_to_base is None:
            raise TargetNotSet(self.__class__.__name__)

        jacobian_frame_in_frame = configuration.get_frame_jacobian(self.entity, self.frame)
        jacobian_base_in_base = configuration.get_frame_jacobian(self.entity, self.base)

        transform_frame_to_base = configuration.get_transform(self.frame, self.base)
        transform_frame_to_target = (
            self.transform_target_to_base.inverse() @ transform_frame_to_base
        )

        return transform_frame_to_target.jlog() @ (
            jacobian_frame_in_frame
            - transform_frame_to_base.inverse().adjoint() @ jacobian_base_in_base
        )
