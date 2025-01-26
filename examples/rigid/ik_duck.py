import argparse

import numpy as np

import genesis as gs
from genesis import quat_to_xyz, xyz_to_quat


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(precision="32", logging_level="info")
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=200,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_collision=False,
            gravity=(0, 0, -0),
        ),
        show_viewer=args.vis,
    )

    ########################## entities ##########################

    robot = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.06,
            pos=(3.5, -1.5, 0.7),
            requires_jac_and_IK=True,
        ),
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()
    kinematics_mode = False
    if not kinematics_mode:
        # set positional gains
        robot.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000]),
            dofs_idx_local=np.arange(6),
        )
        # set velocity gains
        robot.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200]),
            dofs_idx_local=np.arange(6),
        )
        # set force range for safety
        robot.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -87, -87]),
            upper=np.array([87, 87, 87, 87, 87, 87]),
            dofs_idx_local=np.arange(6),
        )

    #target_quat = np.array([0, 1, 0, 0])  # pointing downwards
    center = np.array([0.4, -0.2, 0.25])
    r = 0.1
    ee_link = robot.links[0]

    for i in range(0, 2000):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
        target_quat = xyz_to_quat(np.array([i / 360 * np.pi] * 3), rpy=True, degrees=True)

        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            # return_error=True,
            # rot_mask=[False, False, True], # for demo purpose: only care about direction of z-axis
        )

        # Note that this IK example is only for visualizing the solved q, so here we do not call scene.step(), but only update the state and the visualizer
        # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
        if kinematics_mode:
            robot.set_qpos(q)
        else:
            assert robot.n_dofs == 6
            ctrl = np.empty(6)
            ctrl[:3] = q[:3].cpu().numpy()
            ctrl[3:] = quat_to_xyz(q[3:].cpu().numpy())
            robot.control_dofs_position(ctrl, np.arange(robot.n_dofs))
            scene.step()
        scene.visualizer.update()


if __name__ == "__main__":
    main()
