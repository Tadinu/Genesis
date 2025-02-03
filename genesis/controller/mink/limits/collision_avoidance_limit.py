"""Collision avoidance limit."""

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Union, Iterable

import mujoco as mj
import numpy as np

from ..configuration import Configuration
from .limit import Constraint, Limit

# Genesis
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.ext.trimesh.collision import CollisionManager, ContactData

# Type aliases.
CollisionPair = tuple[str, str]
CollisionPairs = Sequence[CollisionPair]


@dataclass(frozen=True)
class Contact:
    """Struct to store contact information between two geoms.

    Attributes:
        dist: Smallest signed distance between geom1 and geom2. If no collision of
            distance smaller than distmax is found, this value is equal to distmax [1].
        fromto: Segment connecting the closest points on geom1 and geom2. The first
            three elements are the coordinates of the closest point on geom1, and the
            last three elements are the coordinates of the closest point on geom2.
        geom1: ID of geom1.
        geom2: ID of geom2.
        distmax: Maximum distance between geom1 and geom2.

    References:
        [1] MuJoCo API documentation. `mj_geomDistance` function.
            https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html
    """

    dist: float
    fromto: np.ndarray
    collision_1: str
    collision_2: str
    distmax: float

    @property
    def normal(self) -> np.ndarray:
        """Contact normal pointing from geom1 to geom2."""
        normal = self.fromto[3:] - self.fromto[:3]
        mj.mju_normalize3(normal)
        return normal

    @property
    def inactive(self) -> bool:
        """Returns True if no distance smaller than distmax is detected between geom1
        and geom2."""
        return self.dist == self.distmax


def compute_contact_normal_jacobian(
    entity: RigidEntity,
    contact: ContactData,
) -> np.ndarray:
    jac1 = entity.get_jacobian(contact.names[0], np.array(contact.point)).cpu().numpy()
    jac2 = entity.get_jacobian(contact.names[1], np.array(contact.point)).cpu().numpy()
    return contact.normal @ (jac2 - jac1)


def _is_welded_together(model: mj.MjModel, geom_id1: int, geom_id2: int) -> bool:
    """Returns true if the geoms are part of the same body, or if their bodies are
    welded together."""
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body1]
    weld2 = model.body_weldid[body2]
    return weld1 == weld2


def _are_geom_bodies_parent_child(
    model: mj.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geom bodies have a parent-child relationship."""
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]

    # body_weldid is the ID of the body's weld.
    body_weldid1 = model.body_weldid[body_id1]
    body_weldid2 = model.body_weldid[body_id2]

    # weld_parent_id is the ID of the parent of the body's weld.
    weld_parent_id1 = model.body_parentid[body_weldid1]
    weld_parent_id2 = model.body_parentid[body_weldid2]

    # weld_parent_weldid is the weld ID of the parent of the body's weld.
    weld_parent_weldid1 = model.body_weldid[weld_parent_id1]
    weld_parent_weldid2 = model.body_weldid[weld_parent_id2]

    cond1 = body_weldid1 == weld_parent_weldid2
    cond2 = body_weldid2 == weld_parent_weldid1
    return cond1 or cond2


def _is_pass_contype_conaffinity_check(
    model: mj.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    """Returns true if the geoms pass the contype/conaffinity check."""
    cond1 = bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2])
    cond2 = bool(model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1])
    return cond1 or cond2


class CollisionAvoidanceLimit(Limit):
    """Normal velocity limit between geom pairs.

    Attributes:
        model: MuJoCo model.
        geom_pairs: Set of collision pairs in which to perform active collision
            avoidance. A collision pair is defined as a pair of geom groups. A geom
            group is a set of geom names. For each geom pair, the solver will
            attempt to compute joint velocities that avoid collisions between every
            geom in the first geom group with every geom in the second geom group.
            Self collision is achieved by adding a collision pair with the same
            geom group in both pair fields.
        gain: Gain factor in (0, 1] that determines how fast the geoms are
            allowed to move towards each other at each iteration. Smaller values
            are safer but may make the geoms move slower towards each other.
        minimum_distance_from_collisions: The minimum distance to leave between
            any two geoms. A negative distance allows the geoms to penetrate by
            the specified amount.
        collision_detection_distance: The distance between two geoms at which the
            active collision avoidance limit will be active. A large value will
            cause collisions to be detected early, but may incur high computational
            cost. A negative value will cause the geoms to be detected only after
            they penetrate by the specified amount.
        bound_relaxation: An offset on the upper bound of each collision avoidance
            constraint.
    """

    def __init__(
        self,
        entity: RigidEntity,
        model: mj.MjModel,
        collision_managers: dict[str, CollisionManager],
        collision_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
    ):
        """Initialize collision avoidance limit.

        Args:
            model: MuJoCo model.
            geom_pairs: Set of collision pairs in which to perform active collision
                avoidance. A collision pair is defined as a pair of geom groups. A geom
                group is a set of geom names. For each collision pair, the mapper will
                attempt to compute joint velocities that avoid collisions between every
                geom in the first geom group with every geom in the second geom group.
                Self collision is achieved by adding a collision pair with the same
                geom group in both pair fields.
            gain: Gain factor in (0, 1] that determines how fast the geoms are
                allowed to move towards each other at each iteration. Smaller values
                are safer but may make the geoms move slower towards each other.
            minimum_distance_from_collisions: The minimum distance to leave between
                any two geoms. A negative distance allows the geoms to penetrate by
                the specified amount.
            collision_detection_distance: The distance between two geoms at which the
                active collision avoidance limit will be active. A large value will
                cause collisions to be detected early, but may incur high computational
                cost. A negative value will cause the geoms to be detected only after
                they penetrate by the specified amount.
            bound_relaxation: An offset on the upper bound of each collision avoidance
                constraint.
        """
        self.entity = entity
        self.model = model
        self.collision_managers = collision_managers
        self.gain = gain
        self.minimum_distance_from_collisions = minimum_distance_from_collisions
        self.collision_detection_distance = collision_detection_distance
        self.bound_relaxation = bound_relaxation
        self.collision_pairs = collision_pairs
        self.max_num_contacts = 10 * len(self.collision_pairs)

    def compute_qp_inequalities(
        self,
        configuration: Configuration,
        dt: float,
    ) -> Constraint:
        upper_bound = np.full((self.max_num_contacts,), np.inf)
        coefficient_matrix = np.zeros((self.max_num_contacts, self.model.nv))
        for idx, (collision_1, collision_2) in enumerate(self.collision_pairs):
            dist, contact = self._compute_contact_with_minimum_distance(collision_1, collision_2)
            if dist >= self.collision_detection_distance:
                continue
            hi_bound_dist = dist
            if hi_bound_dist > self.minimum_distance_from_collisions:
                dist = hi_bound_dist - self.minimum_distance_from_collisions
                upper_bound[idx] = (self.gain * dist / dt) + self.bound_relaxation
            else:
                upper_bound[idx] = self.bound_relaxation
            jac = compute_contact_normal_jacobian(
                self.entity, contact
            )
            coefficient_matrix[idx] = -jac
        return Constraint(G=coefficient_matrix, h=upper_bound)

    # Private methods.
    def _compute_contact_with_minimum_distance(
        self, collision_1: str, collision_2: str
    ) -> tuple[float, ContactData]:
        """Returns the smallest signed distance between a geom pair."""
        dist = self.collision_managers[collision_1].min_distance_other(self.collision_managers[collision_2])
        _, contact_data = self.collision_managers[collision_1].in_collision_other(self.collision_managers[collision_2],
                                                                                  return_data=True)
        min_depth = dist
        min_contact = None
        for contact in contact_data:
            if contact.depth <= min_depth:
                min_depth = contact.depth
                min_contact = contact
        return dist, min_contact
