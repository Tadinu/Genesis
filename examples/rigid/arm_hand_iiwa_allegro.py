import argparse
from pathlib import Path
import os
from typing_extensions import Optional

import numpy as np
import mujoco as mj
from loop_rate_limiters import RateLimiter

import genesis as gs
from genesis.planner import mink
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "kuka_iiwa_14" / "scene.xml"
_HAND_XML = _HERE / "wonik_allegro" / "left_hand.xml"
_IIWA14_ALLEGRO_XML = _HERE / "kuka_iiwa_14_allegro" / "iiwa14_left_hand.xml"

ATTACH_PREFIX = "allegro_left/"
HAND_BASE = "hand_base"
PALM = f"{ATTACH_PREFIX}palm"
BALL = "ball"

hand_fingertip_names = ["rf_tip", "mf_tip", "ff_tip", "th_tip"]
fingertip_names = [f"{ATTACH_PREFIX}{ftip}" for ftip in hand_fingertip_names]
finger_colors = {
    fingertip_names[0]: [0.9, 0, 0, 1], # Red
    fingertip_names[1]: [0, 0.9, 0, 1], # Green
    fingertip_names[2]: [0, 0, 0.9, 1], # Blue
    fingertip_names[3]: [0.9, 0.9, 0.9, 1] # White
}

# fmt: off
HOME_QPOS = [
    # iiwa.
    -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115,
    # allegro.
    -0.0694123, 0.0551428, 0.986832, 0.671424,
    -0.186261, -0.0866821, 1.01374, 0.728192,
    -0.218949, -0.0318307, 1.25156, 0.840648,
    1.0593, 0.638801, 0.391599, 0.57284
]
# fmt: on

ARM_DOF = 7
PALM_DOF = 16
def construct_robot():
    # https://github.com/google-deepmind/mujoco/blob/main/python/mjspec.ipynb
    arm_spec = mj.MjSpec.from_file(_ARM_XML.as_posix())

    hand_spec = mj.MjSpec.from_file(_HAND_XML.as_posix())
    palm = hand_spec.worldbody.find_child("palm")
    palm.quat = (1, 0, 0, 0)
    palm.pos = (0, 0, 0.095)

    # Add fingertip-end bodies from sites (since Genesis does not build site info from MJ model)
    for fingertip in hand_fingertip_names:
        fingertip_site = hand_spec.find_site(fingertip)
        hand_spec.find_body(fingertip).add_body(name=f"{fingertip}end", pos=fingertip_site.pos,
                                                quat=fingertip_site.quat)

    # Attach [hand_spec] to [arm_spec]
    attach_site = arm_spec.find_site("attachment_site")
    attach_site.attach(hand_spec, ATTACH_PREFIX)

    # TODO: Remove prev "home" key from arm_spec once MuJoCo releases [rem_key] API
    arm_spec.add_key(name="home2", qpos=HOME_QPOS)

    return arm_spec.compile(), arm_spec

def save_model_spec(model_spec: mj.MjSpec, path: Optional[str]=""):
    # NOTE:
    # mj_saveLastXML() only works upon model that was loaded with MjModel.[from_xml() or from_xml_string()]
    # mj_saveModel() only writes to MJCB file
    with open(path if path else f"{os.path.splitext(os.path.basename(__file__))[0]}.xml", "w") as f:
        f.writelines(model_spec.to_xml())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="debug")

    ########################## create a scene ##########################
    scene = gs.Scene(
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
            gravity=(0, 0, -0),
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # Robot
    robot_model, robot_spec = construct_robot()
    robot_morph = gs.morphs.MuJoCoMorph(model=robot_model, file=_IIWA14_ALLEGRO_XML.as_posix())
    #save_model_spec(robot_spec)
    robot = scene.add_entity(robot_morph)
    robot.name = robot_spec.modelname

    # Arm's EE (hand base)
    hand_base = robot.get_link(HAND_BASE)
    ee_target = scene.add_entity(
        name=f"{ATTACH_PREFIX}ee_target",
        morph=gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.10,
            collision=False
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    # Palm
    palm = robot.get_link(PALM)

    # Fingertip targets
    finger_ends: dict[str, RigidLink] = {}
    finger_targets: dict[str, RigidEntity] = {}
    for fingertip in fingertip_names:
        finger_target = scene.add_entity(
            name=f"{fingertip}_target",
            morph=gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.10,
                collision=False
            ),
            surface=gs.surfaces.Default(color=np.array(finger_colors[fingertip])),
        )
        finger_targets[fingertip] = finger_target
        finger_ends[fingertip] = robot.get_link(f"{fingertip}end")

    # Ball
    ball = scene.add_entity(
        name=BALL,
        morph=gs.morphs.Sphere(radius=.07, collision=False),
    )

    ########################## build genesis scene ##########################
    scene.build()

    if scene.sim.rigid_solver.is_active():
        batch_idx = 0  # only visualize contact for the first scene
        for i_con in range(scene.sim.rigid_solver.collider.n_contacts[batch_idx]):
            contact_data = scene.sim.rigid_solver.collider.contact_data[i_con, batch_idx]
            print(contact_data)
            contact_pos = np.array(contact_data.pos) + scene.envs_offset[batch_idx]
            #contact_data.force
    ########################## config tasks ###################
    robot.set_dofs_kp(
        kp=np.full(robot_model.nq, 100),
    )
    robot.set_dofs_kv(
        kv=np.full(robot_model.nq, 100),
    )
    robot.set_dofs_force_range(
        np.full(robot_model.nq, -100),
        np.full(robot_model.nq, 100),
    )
    robot.set_qpos(HOME_QPOS)
    target_quat = hand_base.get_quat().cpu().numpy()
    center = np.array([0.5, 0, 0.5])
    r = 0.1

    # Robot kinematic config
    configuration = mink.Configuration(robot_model)

    # EE task
    end_effector_task = mink.FrameTask(
        entity=robot,
        frame=hand_base,
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # Posture task
    posture_task = mink.PostureTask(entity=robot, model=robot_model, cost=5e-2)
    posture_task.set_target(robot.get_qpos().cpu().numpy()[:robot_model.nq])

    # Finger tasks
    finger_tasks = {}
    for fingertip in fingertip_names:
        task = mink.RelativeFrameTask(
            entity=robot,
            frame=finger_ends[fingertip],
            base=palm,
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        finger_tasks[fingertip] = task

    tasks = [end_effector_task, posture_task]
    tasks.extend(finger_tasks.values())

    # Joint limits
    limits = [
        mink.ConfigurationLimit(entity=robot, model=robot_model),
    ]

    # IK settings
    solver = "quadprog"

    # Init the targets (ee_target + finger_targets)
    mink.move_entity_to_entity(ee_target, hand_base)
    for fingertip in fingertip_names:
        mink.move_entity_to_entity(finger_targets[fingertip], finger_ends[fingertip])
    T_ee_prev = configuration.get_transform_frame_to_world(hand_base)

    # Init ball
    mink.move_entity_to_entity(ball, robot.get_link(BALL))

    # Start exec loop
    rate = RateLimiter(frequency=100.0, warn=False)
    i = 0
    T_ee = None
    while scene.viewer.is_alive():
        # Update kuka end-effector task, as [target]'s SE3
        T_wt = mink.SE3.from_entity(ee_target)
        end_effector_task.set_target(T_wt)

        # Update finger tasks' targets, relative SE3 from [fingertip] to [palm]
        for fingertip, task in finger_tasks.items():
            finger_target = finger_targets[fingertip]
            T_pm = configuration.get_transform(finger_target, palm)
            task.set_target(T_pm)

            # Move [EE] -> also moving finger_targets
            # Calc [T], delta SE3 from current EE to prev EE (hand_base)
            T_ee = configuration.get_transform_frame_to_world(hand_base)
            dT = T_ee @ T_ee_prev.inverse()

            # Calc [T_finger_target], current fingertip-target's SE3
            T_finger_target = configuration.get_transform_frame_to_world(finger_target)

            # Calc [T_finger_target_new], new expected fingertip-target mocap-body's SE3,
            # moving them to new poses
            T_finger_target_new = dT @ T_finger_target
            mink.move_entity_to_frame(finger_target,
                                      T_finger_target_new.translation(), T_finger_target_new.rotation().wxyz)

        # Compute velocity and integrate into the next configuration.
        vel = mink.solve_ik(robot,
                            configuration, tasks, rate.dt, solver, 1e-3, limits=limits
                            )
        configuration.apply_ctrl(entity=robot, velocity=vel, dt=rate.dt, ctrl_type=gs.CTRL_MODE.VELOCITY)

        # Move [target] to new pose
        i+=1
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
        mink.move_entity_to_frame(ee_target, target_pos, target_quat)

        # Save latest [T_eef]
        T_ee_prev = T_ee.copy()

        # Visualize at fixed FPS
        rate.sleep()
        scene.step()
        # End main exec loop

if __name__ == "__main__":
    main()
