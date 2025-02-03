import argparse
import mujoco as mj
import numpy as np
from pathlib import Path

# mjpc
from loop_rate_limiters import RateLimiter
from mink.mjpc import predictive_sampling

# genesis
import genesis as gs
from genesis.controller import mink
from genesis.engine.entities.rigid_entity import RigidEntity
from genesis.controller import mjpc
from genesis.controller.mjpc import predictive_sampling

_HERE = Path(__file__).parent
PARTICLE_CTRL_MODE=gs.CTRL_MODE.POSITION

INIT_Z = 0.03
N_DOFS = 2
i = 0
center = (0,0,INIT_Z)
R = 0.3
target_quat = np.array([1.0, 0.0, 0.0, 0.0])

def apply_ctrl(entities, ctrl):
    particle = entities[0]
    assert ctrl.shape == (N_DOFS,)
    if PARTICLE_CTRL_MODE == gs.CTRL_MODE.POSITION:
        ctrl = (np.concatenate([ctrl, ([INIT_Z]), np.full(5 - N_DOFS, 0)]))
        particle.control_dofs_position(ctrl)
    elif PARTICLE_CTRL_MODE == gs.CTRL_MODE.VELOCITY:
        ctrl = (np.concatenate([ctrl, ([0]), np.full(5 - N_DOFS, 0)]))
        particle.control_dofs_velocity(ctrl)

def init_state(entities: list[RigidEntity], qpos, qvel):
    main_entity = entities[0]
    assert qpos.shape == (N_DOFS,)
    main_entity.set_pos(np.concatenate([qpos,([INIT_Z])]))
    main_entity.set_dofs_velocity(qvel)

def end_state(entities: list[RigidEntity]):
    #apply_ctrl(entities, np.zeros(2))
    pass

# reward
def reward(model: mj.MjModel, entities: list[RigidEntity]) -> float:
    # position
    particle = entities[0]
    goal = entities[1]
    goal = goal.get_pos().cpu().numpy()[:2]
    pos_error = particle.get_pos().cpu().numpy()[:2] - goal
    r0 = -np.dot(pos_error, pos_error)

    # velocity
    linvel = particle.get_vel().cpu().numpy()[:2]
    r1 = -np.dot(linvel, linvel)

    # effort
    #ctrl = particle.get_qpos().cpu().numpy()[:2]
    #r2 = -np.dot(ctrl, ctrl)

    return 5.0 * r0 + 0.1 * r1 #+ 0.1 * r2

def reset(model: mj.MjModel, entities: list[RigidEntity]) -> None:
    global i
    target = entities[1]
    # move target
    i += 1
    target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * R
    mink.move_entity_to_frame(target, target_pos, target_quat)

def construct_main_model():
    # model
    main_spec = mj.MjSpec.from_string("""
        <mujoco model="Particle">
        <worldbody>
            <body name="pointmass"/>
        </worldbody>
        </mujoco>
    """)
    return main_spec.compile(), main_spec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", backend=gs.cpu, logging_level="debug")

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
            enable_collision=False,
            gravity=(0, 0, -0),
        ),
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # model
    main_model, main_spec = construct_main_model()

    # particle
    particle = scene.add_entity(
        name="particle",
        #material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.03,
            collision=True
        ),
        surface=gs.surfaces.Smooth(color=(1.0, 0.0, 1.0, 0.5))
    )

    ghost_particle = scene.add_entity(
        name="ghost_particle",
        #material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.03,
            collision=True
        ),
        surface=gs.surfaces.Smooth(color=(0.0, 1.0, 1.0, 0.5))
    )

    # target
    target = scene.add_entity(
        name="target",
        morph=gs.morphs.Mesh(
            file="meshes/sphere.obj",
            scale=0.03,
            collision=False
        ),
        surface=gs.surfaces.Smooth(color=(0.0, 1.0, 0.0, 0.5))
    )

    ########################## build genesis scene ###############
    scene.build()

    # Start exec loop
    rate = RateLimiter(frequency=100.0, warn=False)

    ########################## controller ###########################
    FPS = 1.0 / rate.dt

    horizon = 0.25
    splinestep = 0.05
    planstep = FPS
    nimprove = 10
    nsample = 10
    noise_scale = 0.1
    interp = "cubic"
    planner = mjpc.predictive_sampling.Planner(
        scene,
        [ghost_particle, target],
        main_model,
        reward,
        reset,
        apply_ctrl,
        init_state,
        end_state,
        N_DOFS,
        np.full((N_DOFS, 2), [R, -R]),
        horizon,
        splinestep,
        planstep,
        nsample,
        noise_scale,
        nimprove,
        interp=interp
    )

    # history
    qpos = [particle.get_qpos()]
    qvel = [particle.get_dofs_velocity()]
    act = [particle.get_dofs_force()]
    ctrl = []
    rewards = []

    ########################## exec #############################
    # Init particle, target
    mink.move_entity_to_frame(particle, np.array([0.0, 0.25, INIT_Z]))
    mink.move_entity_to_frame(ghost_particle, np.array([0.25, 0.25, INIT_Z]))
    mink.move_entity_to_frame(target, np.array([0.25, 0.0, INIT_Z]))

    # verbose
    VERBOSE = False

    rate = RateLimiter(frequency=100.0, warn=False)
    from time import perf_counter, sleep
    while scene.viewer.is_alive():
        ## predictive sampling

        # improve policy
        # Use [particle]'s qpos & qvel as seed to train [ghost_particle], which is registered to [controller] earlier
        planner.improve_policy(
            particle.get_qpos()[:2], particle.get_dofs_velocity(),
            rate.next_tick, rate.dt
        )

        # get action from policy
        ctrl_val = planner.ctrl_from_policy(rate.next_tick)
        apply_ctrl([particle], ctrl_val)

        # reward
        rewards.append(reward(main_model, [particle, target]))

        if VERBOSE:
            print("time  : ", rate.next_tick)
            print(" qpos  : ", particle.get_qpos())
            print(" qvel  : ", particle.get_dofs_velocity())
            print(" act   : ", particle.get_dofs_force())
            print(" ctrl  : ", ctrl_val)
            print(" reward: ", rewards[-1])


        # step
        scene.step()

        # history
        qpos.append(particle.get_qpos())
        qvel.append(particle.get_dofs_velocity())
        act.append(particle.get_dofs_force())
        ctrl.append(ctrl_val)

        # Visualize at fixed FPS
        rate.sleep()
        # End main exec loop

    if VERBOSE:
        target_pos = target.get_pos().cpu().numpy()
        print("\nfinal qpos: ", qpos[-1])
        print("goal state : ", target_pos[0, 0:2])
        print("state error: ", np.linalg.norm(qpos[-1][0:2] - target_pos[0, 0:2]))

if __name__ == "__main__":
    main()