import argparse
from pathlib import Path
import os
from typing_extensions import Optional

import numpy as np
import mujoco as mj
from loop_rate_limiters import RateLimiter

# Genesis
import genesis as gs
from genesis.controller import gink
from genesis.options.morphs import Primitive
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from genesis.ext import trimesh
from trimesh.collision import CollisionManager
from genesis.controller import mjpc
from genesis.controller.mjpc import predictive_sampling

# Iiwa14Allegro
from arm_hand_iiwa_allegro import Iiwa14Allegro

class Iiwa14AllegroMpc:
    N_DOFS = 12
    CUBE_NAME = "cube"
    CUBE_SIZE = 0.06

    DT: float = 0
    ENTITY_SPACING = 1

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
        self.ghost_robot: Iiwa14Allegro = None
        self.cube: RigidEntity = None
        self.ghost_cube: RigidEntity = None
        self.goal_cube: RigidEntity = None

        # Control loop
        self.rate: RateLimiter = None
        self.solver_name: str = "quadprog"

    def construct_robot_system_model(self):
        # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
        self.arm_spec = mj.MjSpec.from_file(Iiwa14Allegro.ARM_XML.as_posix())
        print(self.arm_spec.modelname)
        Iiwa14Allegro.ARM_BODIES_NAMES = [body.name for body in self.arm_spec.bodies]

        self.hand_spec = mj.MjSpec.from_file(Iiwa14Allegro.HAND_XML.as_posix())
        Iiwa14Allegro.HAND_NAME = self.hand_spec.modelname
        Iiwa14Allegro.ATTACH_PREFIX = f"{Iiwa14Allegro.HAND_NAME}/"
        Iiwa14Allegro.PALM_NAME = f"{Iiwa14Allegro.ATTACH_PREFIX}palm"
        self.palm_spec = self.hand_spec.worldbody.find_child("palm")
        self.palm_spec.quat = Iiwa14Allegro.IDENTITY_WXYZ
        self.palm_spec.pos = (0, 0, 0.095)

        Iiwa14Allegro.FINGERTIP_NAMES = [f"{Iiwa14Allegro.ATTACH_PREFIX}{ftip}" for ftip in Iiwa14Allegro.HAND_FINGERTIP_NAMES]
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
        #self.arm_spec.add_key(name="home2", qpos=Iiwa14Allegro.HOME_QPOS)

        return self.arm_spec.compile(), self.arm_spec

    @staticmethod
    def save_model_spec(model_spec: mj.MjSpec, path: Optional[str]=""):
        # NOTE:
        # mj_saveLastXML() only works upon model that was loaded with MjModel.[from_xml() or from_xml_string()]
        # mj_saveModel() only writes to MJCB file
        with open(path if path else f"{os.path.splitext(os.path.basename(__file__))[0]}.xml", "w") as f:
            f.writelines(model_spec.to_xml())

    def init_state(self, entities: list[Iiwa14Allegro | RigidEntity], qpos=None, qvel=None):
        system = entities[0].system
        system.set_dofs_position(qpos)
        system.set_dofs_velocity(qvel)

    def end_state(self, entities: list[RigidEntity]):
        pass

    def apply_ctrl(self, entities: list[Iiwa14Allegro | RigidEntity], ctrl):
        robot = entities[0]
        cube = entities[1]

        # 1. CONTROL
        # 1.1 Control [ee_target]
        # TODO: Only If object is not grasped yet
        cube_pos = cube.get_pos().cpu().numpy()
        gink.move_entity_to_frame(robot.ee_target,
                                  cube_pos + (0,0,0.3),
                                  #np.array([-0.462, 0, 0.887,0])) # ry: 235
                                  np.array([-0.259, 0, 0.966,0])) # ry: 210
        r = 0.5 * self.CUBE_SIZE
        for i, finger_target in enumerate(robot.finger_targets.values()):
            # linear only
            finger_target.control_dofs_position(np.concatenate(
                [cube_pos + (ctrl[3 * i] * np.sin(30 * i) * r, ctrl[3 * i +1] * np.cos(30 * i) * r, ctrl[3 * i+2] * 0.005),
                Iiwa14Allegro.ZERO_XYZ]))

        # 2. DIFF-IK MOVEMENT
        # 2.1 Update [ee_target + finger_targets]'s tasks
        robot.update_tasks()

        # 2.2 Compute velocity and integrate into the next configuration.
        vel = gink.solve_ik(robot.system,
                            robot.configuration, robot.tasks, self.rate.dt, self.solver_name, 1e-3, limits=robot.limits)
        robot.configuration.apply_ctrl(entity=robot.system,
                                       ctrl=vel, ctrl_type=gs.CTRL_MODE.VELOCITY)

    @staticmethod
    def distance(pos_error) -> float:
        # L22 norm
        p = 0.02
        q = 2.0
        c = np.dot(pos_error, pos_error)
        a = c ** (0.5 * q) + p ** q
        s = a ** (1 / q)
        return (s - p)

    def reward(self, model: mj.MjModel, entities: list[Iiwa14Allegro | RigidEntity] = None) -> float:
        robot = entities[0]
        cube = entities[1]
        # cube position - palm position
        cube_pos = cube.get_pos().cpu().numpy()
        hand_cube_pos_error = cube_pos - robot.palm.get_pos().cpu().numpy()
        r0 = -self.distance(hand_cube_pos_error)

        # cube position - goal cube position
        cube_goal_pos_error = cube_pos - self.goal_cube.get_pos().cpu().numpy()
        r1 = -self.distance(cube_goal_pos_error)

        # cube orientation - goal orientation
        goal_orientation = self.goal_cube.get_quat().cpu().numpy()
        cube_orientation = cube.get_quat().cpu().numpy()
        subquat = np.zeros(3)
        mj.mju_subQuat(subquat, goal_orientation, cube_orientation)
        r2 = -0.5 * np.dot(subquat, subquat)

        # cube linear velocity
        linvel = cube.get_vel().cpu().numpy()
        r3 = -0.5 * np.dot(linvel, linvel)

        # actuator
        effort = robot.system.get_dofs_force().cpu().numpy()
        r4 = -0.5 * np.dot(effort, effort)

        # grasp
        graspdiff = robot.system.get_qpos().cpu().numpy()[7:] - robot.HOME_QPOS[7:]
        r5 = -0.5 * np.dot(graspdiff, graspdiff)

        # joint velocity
        jntvel = robot.system.get_dofs_velocity().cpu().numpy()[6:]
        r6 = -0.5 * np.dot(jntvel, jntvel)

        return 20.0 * r0 + 5.0 * r1 + 5.0 * r2 + 10.0 * r3 + 0.1 * r4 + 2.5 * r5 + 1.0e-4 * r6

    def reset(self, model: mj.MjModel, entities: list[Iiwa14Allegro | RigidEntity] = None) -> None:
        pass

    def run(self):
        ########################## init ##########################
        gs.init(seed=0, precision="32", backend=gs.gpu, logging_level=None)

        # Rate
        self.rate = RateLimiter(frequency=100.0, warn=False)
        self.DT = self.rate.dt
        FPS = 1.0 / self.DT

        ########################## create a scene ################
        self.scene = gs.Scene(
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

        # Robot
        iiwa14_allegro_model, iiwa14_allegro_spec = self.construct_robot_system_model()
        #save_model_spec(iiwa14_allegro_spec)
        self.iiwa14_allegro = Iiwa14Allegro(self.scene, iiwa14_allegro_model, iiwa14_allegro_spec,
                                            system_name=iiwa14_allegro_spec.modelname,
                                            system_xml_path=Iiwa14Allegro.IIWA14_ALLEGRO_XML.as_posix())

        self.ghost_robot = Iiwa14Allegro(self.scene, iiwa14_allegro_model, iiwa14_allegro_spec,
                                         system_name=iiwa14_allegro_spec.modelname,
                                         system_xml_path=Iiwa14Allegro.IIWA14_ALLEGRO_XML.as_posix(),
                                         pos=np.array([0, self.ENTITY_SPACING, 0]))

        # Cube
        self.cube = self.scene.add_entity(
            name=self.CUBE_NAME,
            morph=gs.morphs.Box(
                pos=(0.75, 0, 0.2),
                size=np.full(3, self.CUBE_SIZE), fixed=False, collision=True),
        )

        self.ghost_cube = self.scene.add_entity(
            name=self.CUBE_NAME,
            morph=gs.morphs.Box(
                pos=(0.75, self.ENTITY_SPACING, 0.2),
                size=np.full(3, self.CUBE_SIZE), fixed=False, collision=True),
        )

        self.goal_cube = self.scene.add_entity(
            name="goal_cube",
            morph=gs.morphs.Box(
                pos=(0.5, 0.5 * self.ENTITY_SPACING, 0.4),
                size=np.full(3, self.CUBE_SIZE),
                euler=(30, 40, 0),
                fixed=True,
            ),
        )

        ########################## build genesis scene #############
        self.scene.build()

        if self.scene.sim.rigid_solver.is_active():
            batch_idx = 0  # only visualize contact for the first scene
            for i_con in range(self.scene.sim.rigid_solver.collider.n_contacts[batch_idx]):
                contact_data = self.scene.sim.rigid_solver.collider.contact_data[i_con, batch_idx]
                print(contact_data)
                contact_pos = np.array(contact_data.pos) + self.scene.envs_offset[batch_idx]
                #contact_data.force

        ########################## config tasks ###################
        self.iiwa14_allegro.init()
        self.ghost_robot.init()

        ########################## controller #########################
        horizon = 0.25
        splinestep = 0.05
        planstep = FPS
        nimprove = 10
        nsample = 10
        noise_scale = 0.1
        interp = "zero"
        planner = mjpc.predictive_sampling.Planner(
            self.scene,
            [self.ghost_robot, self.ghost_cube, self.goal_cube],
            iiwa14_allegro_model,
            self.reward,
            self.reset,
            self.apply_ctrl,
            self.init_state,
            self.end_state,
            Iiwa14AllegroMpc.N_DOFS,
            np.full((Iiwa14AllegroMpc.N_DOFS, 2), [1, -1]),
            horizon,
            splinestep,
            planstep,
            nsample,
            noise_scale,
            nimprove,
            interp=interp,
        )

        ####################### exec ##############################
        VERBOSE = False
        while self.scene.viewer.is_alive():
            # PREDICTIVE SAMPLING
            # improve policy
            planner.improve_policy(
                self.iiwa14_allegro.system.get_dofs_position().cpu().numpy(),
                self.iiwa14_allegro.system.get_dofs_velocity().cpu().numpy(),
                self.rate.next_tick, self.rate.dt)

            # set ctrl to action from policy
            ctrl_val = planner.ctrl_from_policy(self.rate.next_tick)
            self.apply_ctrl([self.iiwa14_allegro, self.cube], ctrl_val)

            # reward
            reward_val = self.reward(iiwa14_allegro_model, [self.iiwa14_allegro, self.cube])

            if VERBOSE:
                print("time  : ", self.rate.next_tick)
                print(" qpos  : ", self.iiwa14_allegro.system.get_qpos().cpu().numpy())
                print(" qvel  : ", self.iiwa14_allegro.system.get_dofs_velocity().cpu().numpy())
                print(" act   : ", self.iiwa14_allegro.system.get_dofs_force().cpu().numpy())
                print(" action: ", ctrl_val)
                print(" reward: ", reward_val)

            # Visualize at fixed FPS
            self.scene.step()
            self.rate.sleep()
            # End main exec loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    exp = Iiwa14AllegroMpc()
    exp.run()
