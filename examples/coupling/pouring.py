import argparse
import os
import json

from genesis.ext.LuisaRender.src.ext.assimp.port.PyAssimp.scripts.transformations import euler_from_matrix

import numpy as np

import genesis as gs
from genesis import quat_to_xyz, xyz_to_quat, euler_to_quat, R_to_quat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="sph", choices=("sph", "mpm"))
    parser.add_argument("--recon", action="store_true", default=False)
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.gpu, precision="32", logging_level="info")

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -1.5, 0.0),
            upper_bound=(1.0, 1.5, 4.0),
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=0.02,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.5, 6.5, 3.2),
            camera_lookat=(0.5, 1.5, 1.5),
            camera_fov=35,
            max_FPS=120,
        ),
        vis_options=gs.options.VisOptions(
            rendered_envs_idx=[0],
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(gs.morphs.Plane())
    cheezit_box = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="urdf/cheezit/cheezit_box_open_thick.ply",
            # file="urdf/cheezit/hollow_box.stl",
            # file="urdf/cheezit/cheezit_hollow_box.stl",
            pos=(0.5, 0.1, 1.6),
            # scale=(0.01, 0.03, 0.03),
            scale=0.01,
            euler=(0, 0, 0),
            requires_jac_and_IK=True,
            # fixed=True,
            # convexify=True,
            # collision=True,
            # coacd_options=gs.options.CoacdOptions(
            #    resolution=2000,
            #    mcts_iterations=150,
            #    extrude_margin=0.01,
            # ),
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="urdf/cheezit/cheezit_box_open_thick.png",
            ),
        ),
        material=gs.materials.Rigid(gravity_compensation=1)
    )
    bowl = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="urdf/cheezit/bowl.ply",
            pos=(0.5, 0.5, 0.5),
            scale=0.01,
            euler=(0, 0, 0),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="urdf/cheezit/bowl.png",
            ),
        ),
        material=gs.materials.Rigid(gravity_compensation=1)
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

    water = scene.add_emitter(
        material=gs.materials.PBD.Liquid(
            sampler="regular", rho=1000.0,
            density_relaxation=0.2,
            viscosity_relaxation=0.1,
        ),
        max_particles=1000000,
        surface=gs.surfaces.Glass(
            color=(0.7, 0.85, 1.0, 0.7),
            vis_mode="recon" if args.recon else "particle",
        ),
    )
    scene.build()
    ########################## Scene - end ##########################

    # Box dofs
    cheezit_box_dofs_idx = cheezit_box.base_joint.dofs_idx
    cheezit_box.set_dofs_kv(
        np.array([1, 1, 1, 1, 1, 1]) * 50.0,
        cheezit_box_dofs_idx,
    )
    # pos = cheezit_box.get_dofs_position()

    # load poses
    cheezit_box_poses = load_poses(
        "/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/GENESIS/Genesis/genesis/assets/urdf/cheezit/pose_cheezit_only.json")

    kinematics_mode = True
    if not kinematics_mode:
        # set positional gains
        cheezit_box.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000]),
            dofs_idx_local=np.arange(6),
        )
        # set velocity gains
        cheezit_box.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200]),
            dofs_idx_local=np.arange(6),
        )
        # set force range for safety
        cheezit_box.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -87, -87]),
            upper=np.array([87, 87, 87, 87, 87, 87]),
            dofs_idx_local=np.arange(6),
        )

    # target_quat = np.array([0, 1, 0, 0])  # pointing downwards
    ee_link = cheezit_box.links[0]

    poses_num = len(cheezit_box_poses)
    horizon = 2000 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        if False:
            water.emit(
                pos=(0.5, 0.25, 3.5),
                direction=(0.0, 0, -1.0),
                speed=5.0,
                droplet_shape="circle",
                droplet_size=0.3,
            )

        if i >= poses_num:
            scene.step()
        else:
            # print(cheezit_box_poses[i])
            pose = cheezit_box_poses[i]
            target_entity.set_qpos(pose)
            if kinematics_mode:
                cheezit_box.set_qpos(pose)
            else:
                target_pos = pose[:3]
                target_quat = pose[3:]
                q = cheezit_box.inverse_kinematics(
                    link=ee_link,
                    pos=target_pos,
                    quat=target_quat,
                    # return_error=True,
                    # rot_mask=[False, False, True], # for demo purpose: only care about direction of z-axis
                )
                assert cheezit_box.n_dofs == 6
                ctrl = np.empty(6)
                ctrl[:3] = q[:3].cpu().numpy()
                ctrl[3:] = quat_to_xyz(q[3:].cpu().numpy())
                cheezit_box.control_dofs_position(ctrl, np.arange(cheezit_box.n_dofs))
                scene.step()
            scene.visualizer.update()


def load_poses(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    poses = []
    for i, pose in data.items():
        ori = R_to_quat(np.asarray(pose["R"]).reshape(3, 3))
        if True:
            ori = np.asarray([ori[0], -ori[1], -ori[2], -ori[3]])
        trans = 0.001 * np.asarray(pose["t"])
        # print(i,  trans, ori)
        poses.append(np.concatenate([trans, ori]))
    return poses


if __name__ == "__main__":
    main()
