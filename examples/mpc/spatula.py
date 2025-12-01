import argparse
import os

import numpy as np

import genesis as gs
from genesis import quat_to_xyz, xyz_to_quat, euler_to_R, R_to_quat, quat_to_R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(precision="32", logging_level="info", backend=gs.gpu)

    # load poses
    spatula_poses = load_poses(
        "/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/GENESIS/Genesis/genesis/assets/urdf/cheezit/pose_cheezit_only.json")

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, 15, 6),
            camera_lookat=(0, 4, 1),
            camera_fov=35,
            max_FPS=120,
        ),
        # If only want to enable the first env visibility
        # vis_options=gs.options.VisOptions(
        #    rendered_envs_idx=[0],
        # ),
        show_viewer=args.vis,
    )

    cam = scene.add_camera(
        res=(1280, 960),
        pos=(0, 15, 6),
        lookat=(0, 4, 1),
        fov=35,
        GUI=not args.vis,
        spp=128,
    )

    plane = scene.add_entity(gs.morphs.Plane())
    spatula = scene.add_entity(
        gs.morphs.MJCF(file="xml/spatula/cookie_spatula/cookie_spatula.xml", scale=10,
                       requires_jac_and_IK=True),
        vis_mode="collision",
        visualize_contact=True,
    )

    pancake = scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=0.2,
            height=0.1,
            pos=(0.5, 0., 0.3),
        )
    )

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=1.5,
            collision=False,
            # visualization=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 0)),
        material=gs.materials.Rigid(gravity_compensation=1)
    )

    scene.build(n_envs=5, env_spacing=(3.0, 3.0))
    ########################## Scene - end ##########################

    # Spatula dofs
    assert spatula.n_dofs == 6
    spatula_dofs_idx = spatula.base_joint.dofs_idx
    spatula.set_dofs_kv(
        np.ones(spatula.n_dofs) * 50.0,
        spatula_dofs_idx,
    )
    # pos = spatula.get_dofs_position()

    kinematics_mode = False
    if not kinematics_mode:
        # set positional gains
        spatula.set_dofs_kp(
            kp=np.array([10000] * spatula.n_dofs),
            dofs_idx_local=np.arange(spatula.n_dofs),
        )
        # set velocity gains
        spatula.set_dofs_kv(
            kv=np.array([2000] * spatula.n_dofs),
            dofs_idx_local=np.arange(spatula.n_dofs),
        )
        if False:
            # set force range for safety
            spatula.set_dofs_force_range(
                lower=np.array([-87, -87, -87, -87, -87, -87]),
                upper=np.array([87, 87, 87, 87, 87, 87]),
                dofs_idx_local=np.arange(6),
            )

    # target_quat = np.array([0, 1, 0, 0])  # pointing downwards
    ee_link = spatula.links[0]

    poses_num = len(spatula_poses)
    horizon = 1000  # 280 if "PYTEST_VERSION" not in os.environ else 5
    cam.start_recording()
    for i in range(0, horizon):
        # Pose label
        label = f"Pose {i}/{poses_num}"
        if scene.viewer:
            scene.viewer.set_message(label)
        else:
            print(label)

        # spatula
        if i < poses_num:
            # print(spatula_poses[i])
            pose = spatula_poses[i]
            pose = np.tile(pose, (scene.n_envs, 1))
            offset = np.random.uniform(-0.5, 0.5, (scene.n_envs, 3))
            target_entity.set_qpos(pose)
            if kinematics_mode:
                spatula.set_qpos(pose)
            else:
                if True:
                    q = pose
                    ctrl = np.empty((scene.n_envs, 6))
                    for _ in range(scene.n_envs):
                        ctrl[_, :3] = q[_, :3] + offset[_, :3]
                        ctrl[_, 3:] = quat_to_xyz(q[_, 3:])
                else:
                    target_pos = pose[:3]
                    target_quat = pose[3:]
                    q = spatula.inverse_kinematics(
                        link=ee_link,
                        pos=target_pos,
                        quat=target_quat,
                        # return_error=True,
                        # rot_mask=[False, False, True], # for demo purpose: only care about direction of z-axis
                    )
                    ctrl = np.empty(6)
                    ctrl[:3] = q[:3].cpu().numpy()
                    ctrl[3:] = quat_to_xyz(q[3:].cpu().numpy())
                spatula.control_dofs_position(ctrl, np.arange(spatula.n_dofs))

        # progress simulation
        scene.step()
        cam.render()

    # Save video
    if not args.vis:
        cam.stop_recording(save_to_filename='spatula.mp4', fps=20)


def load_poses(file_path):
    import json

    with open(file_path, 'r') as f:
        data = json.load(f)
    rot_correction = np.array([-1, 0, 0, 0, 0, -1, 0, -1, 0]).reshape(3, 3)

    poses = []
    for i, pose in data.items():
        obj_rot = np.asarray(pose["R"]).reshape(3, 3)
        ori = R_to_quat(rot_correction @ obj_rot)
        trans = 0.01 * np.asarray(pose["t"])
        trans = np.dot(rot_correction, trans)
        trans += np.array([0, 20, 5.5])
        # print(i,  trans, ori)
        poses.append(np.concatenate([trans, ori]))
    return poses


if __name__ == "__main__":
    main()
