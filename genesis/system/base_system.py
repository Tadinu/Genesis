import abc
from typing_extensions import Optional
from pathlib import Path

import numpy as np
import mujoco as mj

import genesis as gs
from genesis.controller import gink
from trimesh.collision import CollisionManager

_HERE = Path(__file__).parent


class BaseSystem(abc.ABC):
    IDENTITY_WXYZ = np.array([1., 0., 0., 0.])
    ZERO_XYZ = np.zeros(3)

    def __init__(self, scene: gs.Scene, system_model: mj.MjModel,
                 system_spec: mj.MjSpec,
                 system_name: str, system_xml_path: str,
                 pos: np.ndarray = ZERO_XYZ,
                 quat: np.ndarray = IDENTITY_WXYZ,
                 joint_names: list[str] = None,
                 q0: list[float] = None,
                 gravity_compensation: float = 1.,
                 collision: bool = True,
                 use_mujoco_dynamics: bool = False):
        self.scene = scene
        self.system_model: mj.MjModel = system_model
        self.system_spec: mj.MjSpec = system_spec  # containing pre-compiled model metadata, more easily accessible
        self.system_data: mj.MjData = mj.MjData(system_model) if use_mujoco_dynamics else None
        self.use_mujoco_dynamics: bool = use_mujoco_dynamics
        self.configuration: gink.Configuration = None
        self.tasks: dict[str, gink.Task] = None
        self.limits: list[gink.Limit] = []
        self.collision_pairs: dict[str, str] = {}
        self.collision_managers: dict[str, CollisionManager] = {}
        self.HOME_QPOS: list[float] = q0 if q0 else []
        self.OBSTACLE_NAMES: list[str] = []

        # Get the dof and actuator ids for the active joints to be controlled
        self.N_DOFS: int = len(joint_names) if joint_names else 0
        self.dof_ids: Optional[np.ndarray] = np.array(
            [system_model.joint(name).id for name in joint_names]) if joint_names else None
        self.actuator_ids: Optional[np.ndarray] = np.array(
            [system_model.actuator(name).id for name in joint_names]) if joint_names else None

        # System
        self.system = self.scene.add_entity(name=system_name,
                                            morph=gs.morphs.MuJoCoMorph(model=self.system_model,
                                                                        file=system_xml_path,
                                                                        pos=pos, quat=quat,
                                                                        collision=collision),
                                            material=gs.materials.Rigid(gravity_compensation=gravity_compensation))

    def init(self) -> None:
        """
        Initialize the system. Run only after building [self.scene].
        """
        self._setup()
        self._init_targets()

    def _setup(self) -> None:
        # Robot kinematics
        self.configuration = gink.Configuration(model=self.system_model)
        self.system.set_qpos(self.HOME_QPOS)
        if self.use_mujoco_dynamics:
            self.system_model.opt.timestep = self.scene.dt
            if self.HOME_QPOS:
                self.system_data.qpos = self.HOME_QPOS

        # Control
        self._config_control()

        # Tasks
        self._config_tasks()

        # Limits (position/velocity, joints, collision, etc.)
        self._config_gink_limits()

    @abc.abstractmethod
    def _config_control(self) -> None:
        pass

    def _config_tasks(self) -> None:
        pass

    def _config_gink_limits(self) -> None:
        # Joint limits
        self.limits = [
            gink.ConfigurationLimit(entity=self.system, model=self.system_model)
        ]

    @abc.abstractmethod
    def _init_targets(self) -> None:
        pass

    @abc.abstractmethod
    def update_targets(self) -> None:
        pass

    def update_tasks(self) -> None:
        pass

    def step(self) -> None:
        if self.use_mujoco_dynamics:
            mj.mj_step(self.system_model, self.system_data)
