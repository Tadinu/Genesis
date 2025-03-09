import argparse
from pathlib import Path

import numpy as np
import mujoco as mj
from loop_rate_limiters import RateLimiter

import genesis as gs
from genesis.controller import gink
from genesis.options import SimOptions
from genesis.options.morphs import Primitive
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from genesis.system.base_system import BaseSystem
from genesis.controller.utils import control_osc

_HERE = Path(__file__).parent

class Iiwa14(BaseSystem):
    ARM_XML = _HERE / "kuka_iiwa_14" / "iiwa14_osc.xml"
    ARM_NAME = "iiwa14"
    ARM_BODIES_NAMES = []
    HAND_BASE_NAME = "attachment"

    ACTIVE_JOINT_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7"
    ]

    HOME_QPOS = [
        -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115
    ]

    OSC_GRAVITY_COMPENSATION = True

    def __init__(self, scene: gs.Scene, system_model: mj.MjModel,
                 system_spec: mj.MjSpec,
                 system_name: str, system_xml_path: str,
                 pos: np.ndarray = BaseSystem.ZERO_XYZ,
                 quat: np.ndarray = BaseSystem.IDENTITY_WXYZ,
                 gravity_compensation: float = not OSC_GRAVITY_COMPENSATION,
                 collision: bool = True):
        super().__init__(scene, system_model, system_spec, system_name, system_xml_path,
                         pos, quat,
                         joint_names=Iiwa14.ACTIVE_JOINT_NAMES,
                         q0=Iiwa14.HOME_QPOS,
                         gravity_compensation=gravity_compensation, collision=collision)
        self.plane: Primitive = None

        # 1- EE (hand base)
        self.hand_base = self.system.get_link(Iiwa14.HAND_BASE_NAME)

        # 2- Targets
        self.targets_frame = 0
        # 2.1- EE target
        self.EE_TARGET_CENTER_DEFAULT = np.array([0.5, 0, 0.5])
        self.EE_TARGET_QUAT_DEFAULT = np.array([0, 1, 0, 0])
        self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT = 0.1
        self.ee_target = self.scene.add_entity(
            name=f"ee_target",
            morph=gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.10,
                collision=False
            ),
            surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
        )

    def _config_control(self) -> None:
        self.system.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
            dofs_idx_local=self.dof_ids
        )
        self.system.set_dofs_kv(
            kv=[450, 450, 350, 350, 200, 200, 200],
            dofs_idx_local=self.dof_ids
        )
        self.system.set_dofs_force_range(
            [-87, -87, -87, -87, -12, -12, -12],
            [ 87,  87,  87,  87,  12,  12,  12],
            dofs_idx_local=self.dof_ids
        )

    def _init_targets(self):
        # Init targets (ee_target)
        gink.move_entity_to_frame(self.ee_target,
                                  frame_pos=self.hand_base.get_pos(),
                                  frame_quat=self.EE_TARGET_QUAT_DEFAULT)

    def update_targets(self):
        self.targets_frame += 1
        # Robot's [ee_target]
        delta = self.targets_frame / 360 * np.pi
        target_pos = (self.EE_TARGET_CENTER_DEFAULT +
                      np.array([np.cos(delta), np.sin(delta), 0]) * self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT)
        gink.move_entity_to_frame(self.ee_target, target_pos, self.EE_TARGET_QUAT_DEFAULT)

class Iiwa14OSC:
    ARM_NAME = "iiwa14"

    DT: float = 0.001

    def __init__(self):
        # Genesis scene
        self.scene: gs.Scene = None

        # Model building
        self.arm_spec: mj.MjSpec = None
        self.hand_spec: mj.MjSpec = None

        # Entities
        self.plane: RigidEntity = None
        self.iiwa14: Iiwa14 = None
        self.ball: RigidEntity = None

        # Control loop
        self.rate: RateLimiter = None

    def construct_robot_system_model(self):
        # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
        self.arm_spec = mj.MjSpec.from_file(Iiwa14.ARM_XML.as_posix())
        print(self.arm_spec.modelname)
        Iiwa14.ARM_BODIES_NAMES = [body.name for body in self.arm_spec.bodies]

        # TODO: Remove prev "home" key from arm_spec once MuJoCo releases [rem_key] API
        #self.arm_spec.add_key(name="home", qpos=Iiwa14.HOME_QPOS)
        return self.arm_spec.compile(), self.arm_spec

    def run(self):
        ########################## init ##########################
        gs.init(seed=0, precision="32", backend=gs.cpu, logging_level=None)

        # Rate
        self.rate = RateLimiter(frequency=1/Iiwa14OSC.DT, warn=False)

        ########################## create a scene ################
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=Iiwa14OSC.DT),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, -2, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=200,
            ),
            show_viewer=args.vis,
            show_FPS=True,
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=True,
                enable_collision=True,
                enable_self_collision=False,
                gravity=(0, 0, -9.81),
            ),
        )

        ########################## entities #######################
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane()
        )

        # Robot System
        iiwa14_model, iiwa14_spec = self.construct_robot_system_model()
        # save_model_spec(iiwa14_spec)
        self.iiwa14 = Iiwa14(self.scene, iiwa14_model, iiwa14_spec,
                             system_name=iiwa14_spec.modelname,
                             system_xml_path=Iiwa14.ARM_XML.as_posix())

        ########################## build genesis scene #############
        self.scene.build()

        ########################## init robot system ###################
        self.iiwa14.init()

        ####################### exec ##############################
        while self.scene.viewer.is_alive():
            # 1- Update robot's tasks
            self.iiwa14.update_tasks()

            # 2- Update robot's [ee_target, finger_targets]
            self.iiwa14.update_targets()

            # 3- Calculate next ctrl
            ctrl_mode = gs.CTRL_MODE.FORCE
            ctrl = control_osc(robot=self.iiwa14, ee=self.iiwa14.hand_base, ee_target=self.iiwa14.ee_target,
                               q0=Iiwa14.HOME_QPOS, ctrl_mode=ctrl_mode,
                               gravity_compensation=Iiwa14.OSC_GRAVITY_COMPENSATION,
                               integration_dt=Iiwa14OSC.DT)
            self.iiwa14.configuration.apply_ctrl(entity=self.iiwa14.system, ctrl=ctrl,
                                                 ctrl_mode=ctrl_mode)

            # Visualize at fixed FPS
            self.scene.step()
            self.rate.sleep()
            # End main exec loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    osc = Iiwa14OSC()
    osc.run()
