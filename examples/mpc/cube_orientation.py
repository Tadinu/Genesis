import argparse
import numpy as np
from pathlib import Path

import mujoco as mj

from loop_rate_limiters import RateLimiter

import genesis as gs
from genesis.controller import mink
from genesis.engine.entities.rigid_entity import RigidEntity, RigidLink
from genesis.controller import mjpc
from genesis.controller.mjpc import predictive_sampling

_HERE = Path(__file__).parent
_EXAMPLES_DIR = Path(_HERE).parent

HAND_PREFIX = "rh_"
HAND_FINGERTIP_NAMES = ["rfdistal", "mfdistal", "ffdistal", "lfdistal", "thdistal"]
FINGERTIP_NAMES = [f"{HAND_PREFIX}{ftip}" for ftip in HAND_FINGERTIP_NAMES]
FINGERTIP_COLORS: dict[str, np.array] = {
    FINGERTIP_NAMES[0]: (0.9, 0, 0, 1),  # Red
    FINGERTIP_NAMES[1]: (0, 0.9, 0, 1),  # Green
    FINGERTIP_NAMES[2]: (0, 0, 0.9, 1),  # Blue
    FINGERTIP_NAMES[3]: (0.9, 0.9, 0.9, 1),  # White
    FINGERTIP_NAMES[4]: (0.5, 0.5, 0.5, 1)
}

CUBE_SIZE = 0.05
CUBE_MIN_Z = 0.026
CUBE_INIT_POS = np.array([0.32, 0, 0.4])
GHOST_CUBE_INIT_POS = np.array([0.32, 0.35, 0.4])

def apply_ctrl(entities, ctrl):
    entities[0].control_dofs_position(ctrl)
    # entity.control_dofs_position(np.random.normal(scale=0.1, size=entity.n_dofs))

def init_state(entities: list[RigidEntity], qpos, qvel):
    main_entity = entities[0]
    main_entity.set_qpos(qpos)
    main_entity.set_dofs_velocity(qvel)

def end_state(entities: list[RigidEntity]):
    apply_ctrl(entities, np.zeros(entities[0].n_dofs))

def reward(model: mj.MjModel, entities: list[RigidEntity]) -> float:
    hand = entities[0]
    cube = entities[1]
    goal_cube = entities[2]

    # cube position - palm position (L22 norm)
    palm = hand.get_link("rh_palm")
    pos_error = cube.get_pos().cpu().numpy() - palm.get_pos().cpu().numpy()

    p = 0.02
    q = 2.0
    c = np.dot(pos_error, pos_error)
    a = c ** (0.5 * q) + p**q
    s = a ** (1 / q)
    r0 = -(s - p)

    # cube orientation - goal orientation
    goal_orientation = goal_cube.get_quat().cpu().numpy()
    cube_orientation = cube.get_quat().cpu().numpy()
    subquat = np.zeros(3)
    mj.mju_subQuat(subquat, goal_orientation, cube_orientation)
    r1 = -0.5 * np.dot(subquat, subquat)

    # cube linear velocity
    linvel = cube.get_vel().cpu().numpy()
    r2 = -0.5 * np.dot(linvel, linvel)

    # actuator
    effort = hand.get_dofs_force().cpu().numpy()
    r3 = -0.5 * np.dot(effort, effort)

    # grasp
    graspdiff = hand.get_qpos().cpu().numpy()[7:] - model.key_qpos[0][7:]
    r4 = -0.5 * np.dot(graspdiff, graspdiff)

    # joint velocity
    jntvel = hand.get_dofs_velocity().cpu().numpy()[6:]
    r5 = -0.5 * np.dot(jntvel, jntvel)

    return 20.0 * r0 + 5.0 * r1 + 10.0 * r2 + 0.1 * r3 + 2.5 * r4 + 1.0e-4 * r5

def reset(model: mj.MjModel, entities: list[RigidEntity]) -> None:
    #hand = entities[0]
    cube = entities[1]
    #goal_cube = entities[2]
    cube_pos = cube.get_pos().cpu().numpy()
    if (cube_pos[2] < CUBE_MIN_Z) or (cube_pos[0] < 0.25):
        cube.set_pos(pos=GHOST_CUBE_INIT_POS)

def construct_main_model():
    # model
    main_model_path = f"{_EXAMPLES_DIR}/rigid/shadow_reorient/right_hand.xml"
    main_spec = mj.MjSpec.from_file(main_model_path)
    return main_spec.compile(), main_spec, main_model_path

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
            gravity=(0, 0, -9.81),
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # model
    main_model, main_spec, main_model_path = construct_main_model()
    main_morph = gs.morphs.MuJoCoMorph(model=main_model, file=main_model_path,
                                       pos=(0, 0, 0.2))
    # save_model_spec(main_spec)
    hand = scene.add_entity(name=main_spec.modelname,
                            morph=main_morph, material=gs.materials.Rigid(gravity_compensation=0.5))

    ghost_morph = gs.morphs.MuJoCoMorph(model=main_model, file=main_model_path,
                                        pos=(0, 0.35, 0.2))
    ghost_hand = scene.add_entity(name="GhostShadowHand",
                                  morph=ghost_morph, material=gs.materials.Rigid(gravity_compensation=0.5))

    # Fingertip targets
    finger_ends: dict[str, RigidLink] = {}
    finger_targets: dict[str, RigidEntity] = {}
    for fingertip in FINGERTIP_NAMES:
        finger_target = scene.add_entity(
            name=f"{fingertip}_target",
            morph=gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.10,
                collision=False
            ),
            surface=gs.surfaces.Default(color=FINGERTIP_COLORS[fingertip]),
        )
        finger_targets[fingertip] = finger_target
        finger_ends[fingertip] = hand.get_link(fingertip)

    # cube
    MAT_FRICTIONLESS_RIGID = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
    cube = scene.add_entity(
        name="cube",
        material=MAT_FRICTIONLESS_RIGID,
        morph=gs.morphs.Box(
            pos=(0.35, 0, 0.4),
            size=np.full(3, CUBE_SIZE),
            fixed=False,
            collision=True
        ),
    )

    # cube
    ghost_cube = scene.add_entity(
        name="ghost_cube",
        material=MAT_FRICTIONLESS_RIGID,
        morph=gs.morphs.Box(
            pos=(0.35, 0.35, 0.4),
            size=np.full(3, CUBE_SIZE),
            fixed=False,
            collision=True
        ),
    )

    # goal cube
    goal_cube = scene.add_entity(
        name="goal_cube",
        morph=gs.morphs.Box(
            pos=(0.5, 0.5, 0.4),
            size=np.full(3, CUBE_SIZE),
            euler=(30, 40, 0),
            fixed=True,
        ),
    )

    ########################## build genesis scene ###############
    #scene.build(n_envs=10, env_spacing=(1.0, 1.0))
    scene.build()

    # rate
    rate = RateLimiter(frequency=100.0, warn=False)
    FPS = 1.0 / rate.dt

    ########################## controller ###########################
    horizon = 0.25
    splinestep = 0.05
    planstep = FPS
    nimprove = 10
    nsample = 10
    noise_scale = 0.1
    interp = "zero"
    planner = mjpc.predictive_sampling.Planner(
        scene,
        [ghost_hand, ghost_cube, goal_cube],
        main_model,
        reward,
        reset,
        apply_ctrl,
        init_state,
        end_state,
        ghost_hand.n_dofs,
        ghost_hand.get_dofs_limit_numpy(),
        horizon,
        splinestep,
        planstep,
        nsample,
        noise_scale,
        nimprove,
        interp=interp,
    )

    # history
    qpos = [hand.get_qpos()]
    qvel = [hand.get_dofs_velocity()]
    act = [hand.get_dofs_force()]
    ctrl = []
    rewards = []

    ########################## exec #############################
    # Init the targets (finger_targets)
    for fingertip in FINGERTIP_NAMES:
        mink.move_entity_to_entity(finger_targets[fingertip], finger_ends[fingertip])

    # verbose
    VERBOSE = False
    while scene.viewer.is_alive():
        ## Predictive sampling
        # improve policy
        planner.improve_policy(
            hand.get_qpos(), hand.get_dofs_velocity(),
            rate.next_tick, rate.dt
        )

        # set ctrl to action from policy
        apply_ctrl([hand], planner.ctrl_from_policy(rate.next_tick))
        mink.move_entity_to_frame(cube, frame_pos=CUBE_INIT_POS)

        # reward
        rewards.append(reward(main_model, [hand, cube, goal_cube]))

        if VERBOSE:
            print("time  : ", rate.next_tick)
            print(" qpos  : ", hand.get_qpos())
            print(" qvel  : ", hand.get_dofs_velocity())
            print(" act   : ", hand.get_dofs_force())
            print(" action: ", hand.get_dofs_control_force())
            print(" reward: ", rewards[-1])

        # step
        scene.step()

        # history
        qpos.append(hand.get_qpos())
        qvel.append(hand.get_dofs_velocity())
        act.append(hand.get_dofs_force())
        ctrl.append(hand.get_dofs_position())

        # Visualize at fixed FPS
        rate.sleep()
        # End main exec loop

if __name__ == "__main__":
    main()
