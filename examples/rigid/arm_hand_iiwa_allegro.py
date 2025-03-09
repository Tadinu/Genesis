import argparse
from pathlib import Path
import os
from typing_extensions import Optional

import numpy as np
import mujoco as mj
from loop_rate_limiters import RateLimiter

import genesis as gs
from genesis.controller import gink
from genesis.options import SimOptions
from genesis.options.morphs import Primitive
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
import trimesh
from trimesh.collision import CollisionManager
from genesis.system.base_system import BaseSystem
from genesis.utils.misc import print_class

_HERE = Path(__file__).parent


class Iiwa14Allegro(BaseSystem):
    ARM_XML = _HERE / "kuka_iiwa_14" / "scene.xml"
    HAND_XML = _HERE / "wonik_allegro" / "left_hand.xml"
    IIWA14_ALLEGRO_XML = _HERE / "kuka_iiwa_14_allegro" / "iiwa14_left_hand.xml"

    ARM_NAME = "iiwa14"
    ARM_BODIES_NAMES = []
    HAND_BASE_NAME = "attachment"
    HAND_FINGERTIP_NAMES = ["rf_tip", "mf_tip", "ff_tip", "th_tip"]
    FINGERTIP_NAMES = []
    FINGERTIP_COLORS = {}
    PALM_NAME = ""
    ATTACH_PREFIX = ""
    ARM_DOFS_NO = 7
    HAND_DOFS_NO = 16

    HOME_QPOS = [
        # iiwa.
        -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115,
        # allegro.
        -0.0694123, 0.0551428, 0.986832, 0.671424,
        -0.186261, -0.0866821, 1.01374, 0.728192,
        -0.218949, -0.0318307, 1.25156, 0.840648,
        1.0593, 0.638801, 0.391599, 0.57284
    ]

    def __init__(self, scene: gs.Scene, system_model: mj.MjModel,
                 system_spec: mj.MjSpec,
                 system_name: str, system_xml_path: str,
                 pos: np.ndarray = BaseSystem.ZERO_XYZ,
                 quat: np.ndarray = BaseSystem.IDENTITY_WXYZ,
                 gravity_compensation: float = 1.,
                 collision: bool = True):
        super().__init__(scene, system_model, system_spec, system_name, system_xml_path,
                         pos, quat, q0=Iiwa14Allegro.HOME_QPOS,
                         gravity_compensation=gravity_compensation, collision=collision)
        self.ee_task: gink.FrameTask = None
        self.posture_task: gink.Task = None
        self.ee_target: RigidEntity = None
        self.finger_tasks: dict[str, gink.RelativeFrameTask] = None
        self.finger_ends: dict[str, RigidLink] = {}
        self.finger_targets: dict[str, RigidEntity] = {}
        self.plane: Primitive = None
        self.palm: RigidLink = None
        self.hand_base: RigidLink = None
        self.T_ee_prev: gink.SE3 = None
        self.N_DOFS = Iiwa14Allegro.ARM_DOFS_NO + Iiwa14Allegro.HAND_DOFS_NO

        # 1- EE (hand base)
        self.hand_base = self.system.get_link(Iiwa14Allegro.HAND_BASE_NAME)

        # 2- Palm
        self.palm = self.system.get_link(Iiwa14Allegro.PALM_NAME)

        # 3- Targets
        self.targets_frame = 0
        # 3.1- EE target
        self.EE_TARGET_CENTER_DEFAULT = np.array([0.5, 0, 0.5])
        self.EE_TARGET_QUAT_DEFAULT = np.array([0, 1, 0, 0])
        self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT = 0.1
        self.ee_target = self.scene.add_entity(
            name=f"{Iiwa14Allegro.ATTACH_PREFIX}ee_target",
            morph=gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.10,
                collision=False
            ),
            surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
        )

        # 3.2- Fingertip targets
        for fingertip in Iiwa14Allegro.FINGERTIP_NAMES:
            finger_target = self.scene.add_entity(
                name=f"{fingertip}_target",
                morph=gs.morphs.Mesh(
                    file="meshes/axis.obj",
                    scale=0.10,
                    collision=False
                ),
                surface=gs.surfaces.Default(color=np.array(Iiwa14Allegro.FINGERTIP_COLORS[fingertip])),
            )
            self.finger_targets[fingertip] = finger_target
            self.finger_ends[fingertip] = self.system.get_link(f"{fingertip}end")

    def _config_control(self) -> None:
        self.system.set_dofs_kp(
            kp=np.full(self.N_DOFS, 100),
        )
        self.system.set_dofs_kv(
            kv=np.full(self.N_DOFS, 100),
        )
        self.system.set_dofs_force_range(
            np.full(self.N_DOFS, -100),
            np.full(self.N_DOFS, 100),
        )

    def _config_tasks(self):
        # EE task
        self.ee_task = gink.FrameTask(
            entity=self.system,
            frame=self.hand_base,
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        # Posture task
        self.posture_task = gink.PostureTask(entity=self.system, model=self.system_model, cost=5e-2)
        self.posture_task.set_target(self.system.get_qpos().cpu().numpy())

        # Finger tasks
        self.finger_tasks = {}
        for fingertip in self.FINGERTIP_NAMES:
            task = gink.RelativeFrameTask(
                entity=self.system,
                frame=self.finger_ends[fingertip],
                base=self.palm,
                position_cost=1.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            self.finger_tasks[fingertip] = task

        self.tasks = [self.ee_task, self.posture_task]
        self.tasks.extend(self.finger_tasks.values())

    def _config_limits(self):
        # Joint limits
        self.limits = [
            gink.ConfigurationLimit(entity=self.system, model=self.system_model),
        ]

        self.collision_pairs = []
        for obs_name in self.OBSTACLE_NAMES:
            self.collision_managers[obs_name] = CollisionManager()
            self.collision_pairs.extend([(self.ARM_NAME, obs_name),
                                         (self.PALM_NAME, obs_name)])

        # Collision managers
        for collision_pair in self.collision_pairs:
            self.collision_managers[collision_pair[0]] = CollisionManager()

        for body in self.system_spec.bodies:
            for i, geom in enumerate(body.geoms):
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom_name = geom.name if geom.name else f"{body.name}_geom{i}"
                mesh = None
                mesh_name = None
                if geom.type == mj.mjtGeom.mjGEOM_MESH:
                    mesh_name = geom.meshname
                    geom_meshpath = None
                    for mesh in self.system_spec.meshes:
                        if mesh.name == mesh_name:
                            geom_meshpath = os.path.join(self.system_spec.modelfiledir, self.system_spec.meshdir,
                                                         mesh.file)
                            break
                    if geom_meshpath:
                        mesh = trimesh.load_mesh(geom_meshpath)
                else:
                    mesh_name = geom_name
                    if geom.type == mj.mjtGeom.mjGEOM_BOX:
                        mesh = trimesh.primitives.Box()
                    elif geom.type == mj.mjtGeom.mjGEOM_SPHERE:
                        mesh = trimesh.primitives.Sphere(radius=geom.size[0])

                    elif geom.type == mj.mjtGeom.mjGEOM_CYLINDER:
                        mesh = trimesh.primitives.Cylinder(radius=geom.size[0], height=geom.size[1])

                    elif geom.type == mj.mjtGeom.mjGEOM_CAPSULE:
                        mesh = trimesh.primitives.Capsule(radius=geom.size[0], height=geom.size[1])

                    elif geom.type == mj.mjtGeom.mjGEOM_PLANE:
                        mesh = trimesh.primitives.Box(extents=geom.size)

                if mesh:
                    for obs_name in self.OBSTACLE_NAMES:
                        collision_mang = self.collision_managers[
                            self.ARM_NAME] if body.name in Iiwa14Allegro.ARM_BODIES_NAMES \
                            else self.collision_managers[self.PALM_NAME] if body.name.startswith(
                            Iiwa14Allegro.ATTACH_PREFIX) \
                            else self.collision_managers[obs_name] if obs_name in self.collision_managers else None
                        if collision_mang:
                            collision_mang.add_object(name=mesh_name,
                                                      mesh=mesh)  # TODO: transform=geom.xmat + geom.xpos

        # Collision avoidance limit
        self.limits.append(gink.CollisionAvoidanceLimit(
            entity=self.system,
            model=self.system_model,
            collision_managers=self.collision_managers,
            collision_pairs=self.collision_pairs,
            minimum_distance_from_collisions=0.1,
            collision_detection_distance=0.2,
        ))

    def update_tasks(self):
        self._update_task_ee()
        self._update_task_fingers()

    def _update_task_ee(self):
        # Update kuka end-effector task, as [target]'s SE3
        T_wt = gink.SE3.from_entity(self.ee_target)
        self.ee_task.set_target(T_wt)

    def _update_task_fingers(self):
        # Update finger-tasks' targets, relative SE3 from [fingertip] to [palm]
        T_ee = gink.SE3()
        for fingertip, task in self.finger_tasks.items():
            finger_target = self.finger_targets[fingertip]
            T_pm = self.configuration.get_transform(finger_target, self.palm)
            task.set_target(T_pm)

            # Move [EE] -> also moving finger_targets
            # Calc [T], delta SE3 from current EE to prev EE (hand_base)
            T_ee = self.configuration.get_transform_frame_to_world(self.hand_base)
            deltaT = T_ee @ self.T_ee_prev.inverse()

            # Calc [T_finger_target], current fingertip-target's SE3
            T_finger_target = self.configuration.get_transform_frame_to_world(finger_target)

            # Calc [T_finger_target_new], new expected fingertip-target mocap-body's SE3,
            # moving them to new poses
            T_finger_target_new = deltaT @ T_finger_target
            gink.move_entity_to_frame(finger_target,
                                      T_finger_target_new.translation(), T_finger_target_new.rotation().wxyz)

        # Save latest [T_ee]
        self.T_ee_prev = T_ee.copy()

    def _init_targets(self):
        # Init targets (ee_target + finger_targets)
        gink.move_entity_to_frame(self.ee_target,
                                  frame_pos=self.hand_base.get_pos(),
                                  frame_quat=self.EE_TARGET_QUAT_DEFAULT)
        for fingertip in self.FINGERTIP_NAMES:
            gink.move_entity_to_entity(self.finger_targets[fingertip], self.finger_ends[fingertip])
        self.T_ee_prev = self.configuration.get_transform_frame_to_world(self.hand_base)

    def update_targets(self):
        self.targets_frame += 1
        # Robot's [ee_target]
        delta = self.targets_frame / 360 * np.pi
        target_pos = (self.EE_TARGET_CENTER_DEFAULT +
                      np.array([np.cos(delta), np.sin(delta), 0]) * self.EE_TARGET_MOVEMENT_RADIUS_DEFAULT)
        gink.move_entity_to_frame(self.ee_target, target_pos, self.EE_TARGET_QUAT_DEFAULT)


class Iiwa14AllegroDiffIK:
    ARM_NAME = "iiwa14"
    HAND_NAME = "allegro"
    BALL_NAME = "ball"
    BALL_SIZE = 0.07

    DT: float = 0.01

    def __init__(self):
        # Genesis scene
        self.scene: gs.Scene = None

        # Model building
        self.arm_spec: mj.MjSpec = None
        self.hand_spec: mj.MjSpec = None
        self.palm_spec: mj.MjSpec = None

        # Entities
        self.plane: RigidEntity = None
        self.iiwa14_allegro: Iiwa14Allegro = None
        self.ball: RigidEntity = None

        # Control loop
        self.rate: RateLimiter = None

        # Solver
        self.solver_name: str = "quadprog"

    def construct_robot_system_model(self):
        # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
        # https://mujoco.readthedocs.io/en/latest/python.html#construction
        self.arm_spec = mj.MjSpec.from_file(Iiwa14Allegro.ARM_XML.as_posix())
        print(self.arm_spec.modelname)
        Iiwa14Allegro.ARM_BODIES_NAMES = [body.name for body in self.arm_spec.bodies]

        self.hand_spec = mj.MjSpec.from_file(Iiwa14Allegro.HAND_XML.as_posix())
        Iiwa14Allegro.HAND_NAME = self.hand_spec.modelname
        Iiwa14Allegro.ATTACH_PREFIX = f"{Iiwa14Allegro.HAND_NAME}/"
        Iiwa14Allegro.PALM_NAME = f"{Iiwa14Allegro.ATTACH_PREFIX}palm"
        self.palm_spec = self.hand_spec.worldbody.find_child("palm")
        self.palm_spec.quat = BaseSystem.IDENTITY_WXYZ
        self.palm_spec.pos = (0, 0, 0.095)

        Iiwa14Allegro.FINGERTIP_NAMES = [f"{Iiwa14Allegro.ATTACH_PREFIX}{ftip}" for ftip in
                                         Iiwa14Allegro.HAND_FINGERTIP_NAMES]
        Iiwa14Allegro.FINGERTIP_COLORS = {
            Iiwa14Allegro.FINGERTIP_NAMES[0]: [0.9, 0, 0, 1],  # Red
            Iiwa14Allegro.FINGERTIP_NAMES[1]: [0, 0.9, 0, 1],  # Green
            Iiwa14Allegro.FINGERTIP_NAMES[2]: [0, 0, 0.9, 1],  # Blue
            Iiwa14Allegro.FINGERTIP_NAMES[3]: [0.9, 0.9, 0.9, 1]  # White
        }

        # Add fingertip-end bodies from sites (since Genesis does not build site info from MJ model)
        for fingertip in Iiwa14Allegro.HAND_FINGERTIP_NAMES:
            fingertip_site = self.hand_spec.site(fingertip)
            self.hand_spec.body(fingertip).add_body(name=f"{fingertip}end", pos=fingertip_site.pos,
                                                    quat=fingertip_site.quat)

        # Attach [hand_spec] to [arm_spec]
        attach_site = self.arm_spec.site("attachment_site")
        attach_site.attach_body(self.hand_spec.worldbody, Iiwa14Allegro.ATTACH_PREFIX)

        # TODO: Remove prev "home" key from arm_spec once MuJoCo releases [rem_key] API
        # self.arm_spec.add_key(name="home", qpos=self.HOME_QPOS)

        return self.arm_spec.compile(), self.arm_spec

    def run(self):
        ########################## init ##########################
        gs.init(seed=0, precision="32", backend=gs.cpu, logging_level=None)

        # Rate
        self.rate = RateLimiter(frequency=1 / Iiwa14AllegroDiffIK.DT, warn=False)

        ########################## create a scene ################
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.rate.dt),
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
                gravity=(0, 0, -0),
            ),
        )

        ########################## entities #######################
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane()
        )

        # Robot System
        iiwa14_allegro_model, iiwa14_allegro_spec = self.construct_robot_system_model()
        # save_model_spec(iiwa14_allegro_spec)
        self.iiwa14_allegro = Iiwa14Allegro(self.scene, iiwa14_allegro_model, iiwa14_allegro_spec,
                                            system_name=iiwa14_allegro_spec.modelname,
                                            system_xml_path=Iiwa14Allegro.IIWA14_ALLEGRO_XML.as_posix())
        self.iiwa14_allegro.OBSTACLE_NAMES = [self.BALL_NAME]

        # Ball
        self.ball = self.scene.add_entity(
            name=Iiwa14AllegroDiffIK.BALL_NAME,
            morph=gs.morphs.Sphere(radius=Iiwa14AllegroDiffIK.BALL_SIZE, pos=(0.5, 0, 0.3), fixed=True, collision=False)
        )

        ########################## build genesis scene #############
        self.scene.build()

        if self.scene.sim.rigid_solver.is_active():
            batch_idx = 0  # only visualize contact for the first scene
            contacts_info = self.scene.sim.rigid_solver.collider.get_contacts()
            if self.scene.sim.rigid_solver.n_envs > 0:
                contacts_info = {key: value[batch_idx] for key, value in contacts_info.items()}

                # Early return if no contact
                n_contacts = len(contacts_info["geom_a"])
                if n_contacts > 0:
                    geoms_aabb = self.scene.sim.rigid_solver.geoms_init_AABB.to_numpy()
                    ga_aabb = geoms_aabb[contacts_info["geom_a"]]
                    gb_aabb = geoms_aabb[contacts_info["geom_b"]]
                    ga_aabb_size = np.linalg.norm(ga_aabb[:, -1] - ga_aabb[:, 0], axis=1)
                    gb_aabb_size = np.linalg.norm(gb_aabb[:, -1] - gb_aabb[:, 0], axis=1)
                    normal_scale = np.minimum(ga_aabb_size, gb_aabb_size)

                    contact_pos = contacts_info["position"] + self.scene.envs_offset[batch_idx]
                    contact_normal_scaled = contacts_info["normal"] * normal_scale[:, None]
                    contact_force = contacts_info["force"]
                    print(contact_pos, contact_normal_scaled, contact_force)

                    for i_c in range(n_contacts):
                        for link_idx, sign in (
                                (contacts_info["link_a"][i_c], -1),
                                (contacts_info["link_b"][i_c], 1),
                        ):
                            print(link_idx, sign)

        ########################## init robot system ###################
        self.iiwa14_allegro.init()

        ####################### exec ##############################
        while self.scene.viewer.is_alive():
            # 1- Update robot's tasks
            self.iiwa14_allegro.update_tasks()

            # 2- Update robot's [ee_target, finger_targets]
            self.iiwa14_allegro.update_targets()

            # 3- Compute velocity and integrate into the next configuration.
            vel = gink.solve_ik(self.iiwa14_allegro.system,
                                self.iiwa14_allegro.configuration, self.iiwa14_allegro.tasks, self.rate.dt,
                                self.solver_name, damping=1e-3,
                                limits=self.iiwa14_allegro.limits)
            # position-control
            self.iiwa14_allegro.configuration.apply_ctrl(entity=self.iiwa14_allegro.system,
                                                         ctrl=self.iiwa14_allegro.configuration.integrate(
                                                             self.iiwa14_allegro.system, vel, self.rate.dt),
                                                         ctrl_mode=gs.CTRL_MODE.POSITION)

            # Visualize at fixed FPS
            self.scene.step()
            self.iiwa14_allegro.step()
            self.rate.sleep()
            # End main exec loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    diffIk = Iiwa14AllegroDiffIK()
    diffIk.run()
