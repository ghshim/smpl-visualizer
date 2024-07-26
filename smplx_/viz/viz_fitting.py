'''
References:
    https://github.com/vchoutas/smplx/blob/1265df7ba545e8b00f72e7c557c766e15c71632f/examples/demo.py
    https://github.com/mohamedhassanmus/prox/blob/master/prox/viz/viz_fitting.py
'''
import argparse
import os
import pickle
import json
import time

import numpy as np
import smplx
import torch

from smplx_.utils.constants import JOINT_NAMES, SKELETON, JOINT_HIERARHCY

def viz_motion(motion, corners):
    poses, translation, forward_direction, global_velocity, rotational_velocity = motion
    num_frames = len(poses)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.cla()
    
    def update(frame):
        ax.clear()
        
        # Calculate the current translation and forward direction based on velocities
        current_translation = translation[frame]
        
        # if frame == 0:
        #     forward_direction = np.array([1, 0, 0])
        # else:
        #     # current_translation = translation[frame] #+ global_velocity[frame - 1]
        #     current_forward_direction = forward_direction[frame - 1]
        #     # current_forward_direction = rotate_vector(forward_direction[frame - 1], 
        #     #                                           rotational_velocity[frame - 1], 
        #     #                                           np.array([0, 1, 0]))  # Assuming Y-axis rotation
        current_pose = poses[frame,:,:]
        
        # if frame < num_frames:
        #     current_pose = poses[frame,:,:]
        # else:
        #     current_pose = poses[-1,:,:]
        
        # Translate the current pose to ensure it's attached to the ground
        # min_x = np.min(gt_current_pose[:, 0])
        # min_z = np.min(gt_current_pose[:, 2])
        # ground_translation = np.array([-min_x, 0, -min_z])
        
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        
        # Update scatter plot
        # scatter = ax.scatter(current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])
        
        # Update the human body position
        human_body = current_pose + current_translation
        ax.scatter(human_body[:, 0], human_body[:, 1], human_body[:, 2], c='r')
        
        # Draw the forward direction vector
        # ax.quiver(current_translation[0], current_translation[1], current_translation[2], 
        #           current_forward_direction[0], current_forward_direction[1], current_forward_direction[2], color='g')
        
        # Draw connections between joints (assuming some SKELETON structure, adjust as needed)
        for joint_start, joint_end in SKELETON:
            ax.plot([human_body[joint_start, 0], human_body[joint_end, 0]],
                    [human_body[joint_start, 1], human_body[joint_end, 1]],
                    [human_body[joint_start, 2], human_body[joint_end, 2]], 'k-')
        
        
        # Plot bounding boxes
        for i in range(len(corners)):
            # Define the 12 lines connecting the corners of the bounding box
            edges = [
                [corners[i][j] for j in [0, 1, 3, 2, 0]],
                [corners[i][j] for j in [4, 5, 7, 6, 4]],
                [corners[i][j] for j in [0, 4]],
                [corners[i][j] for j in [1, 5]],
                [corners[i][j] for j in [2, 6]],
                [corners[i][j] for j in [3, 7]]
            ]
            # poly3d = [[bbox_corners[vert] for vert in face] for face in [
            #     [0, 1, 5, 4], [7, 6, 2, 3], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]]
            # ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
            
            for edge in edges:
                ax.plot([point[0] for point in edge], [point[1] for point in edge], [point[2] for point in edge], 'g-')
        
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100)
    
    return ani

def viz(fitting_list,               
        model_dir,                  # smplx model dir
        model_type='smplx',
        gender='neutral',
        ext='npz',                  # model file extension (npz or pkl)
        num_betas=10,          
        num_pca_comps=6,     
        num_expression_coeffs=10,
        use_face_contour=False,
        plot_joints=True,      
        plotting_module='open3d',
        xform_path=None,
        fps=33):
    
    '''load transformation matrix'''
    if xform_path:
        with open(xform_path, 'r') as f:
            trans = np.array(json.load(f)) # transformation matrix

    '''load plotting module'''
    if plotting_module == 'pyrender':
        import pyrender
        import trimesh  

        # TODO : implement OffScreen renderer -> test each osmesa, egl in server
       
        # live scene viewer
        scene = pyrender.Scene()
        v = pyrender.Viewer(scene,
                        use_raymond_lighting=True,
                        cull_faces=False,
                        run_in_thread=True) # run the viewer in a separate thread so that you can update the scene while the viewer is running
        

    elif plotting_module == 'open3d':
        import open3d as o3d
        # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        body = o3d.geometry.TriangleMesh()
        if plot_joints:
            joints_pcl = o3d.geometry.PointCloud()
            vis.add_geometry(joints_pcl)
        

    elif plotting_module == 'matplotlib':
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.animation import FuncAnimation, FFMpegWriter
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-2, 2]); ax.set_ylim([-2, 2]); ax.set_zlim([-2, 2])
        ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_zlabel('Y')
        face_color = (1.0, 1.0, 0.9); edge_color = (0, 0, 0)

        def update(mesh, joints, draw_skel=False):
            ax.clear()

            mesh.set_edgecolor(edge_color)
            mesh.set_facecolor(face_color)
            ax.add_collection3d(mesh)
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r')
            if plot_joints:
                ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
            
            if draw_skel:
                # Draw connections between joints (assuming some SKELETON structure, adjust as needed)
                for joint_start, joint_end in SKELETON:
                    ax.plot([joints[joint_start, 0], joints[joint_end, 0]],
                            [joints[joint_start, 1], joints[joint_end, 1]],
                            [joints[joint_start, 2], joints[joint_end, 2]], 'k-')
        

    else:
        raise ValueError('Unknown plotting_module: {}'.format(plotting_module))
    
    '''create smplx model'''
    body_model = smplx.create(model_dir, 
                              model_type='smplx',
                              gender=gender, 
                              ext=ext, 
                              num_betas=num_betas,
                              num_pca_comps=num_pca_comps)
    
    '''visualize the output of smplx'''
    for i, fitting_file in enumerate(fitting_list):
        frame_name = fitting_file.split('/')[-2]  # Change this code suitable format for printing frame (or data) name
        print(frame_name)

        # open smplx fitting file
        with open(fitting_file, 'rb') as f:
            param = pickle.load(f) # smplx parameters
            torch_param = {}

        param['left_hand_pose'] = param['left_hand_pose'][:, :num_pca_comps]   
        param['right_hand_pose'] = param['right_hand_pose'][:, :num_pca_comps]
        
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])
        # print(torch_param)
                    
        # input fitting parameters to smplx model
        # https://github.com/vchoutas/smplx/blob/1265df7ba545e8b00f72e7c557c766e15c71632f/smplx/body_models.py#L1122
        output = body_model(return_verts=True, **torch_param)
        
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()
        
        # print('Vertices shape =', vertices.shape)
        # print('Joints shape =', joints.shape)
    
        if plotting_module == 'pyrender':
            # lock viewer
            v.render_lock.acquire()
            
            vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
            tri_mesh = trimesh.Trimesh(vertices, body_model.faces, vertex_colors=vertex_colors)

            if xform_path: mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False, poses=trans)
            else: mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)

            scene.add(mesh, name=frame_name)

            if plot_joints:
                if xform_path: joints = (trans[:3,:3] @ joints.T).T  + trans[:3, 3]

                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                tfs = np.tile(np.eye(4), (len(joints), 1, 1))
                tfs[:, :3, 3] = joints
                joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                
                scene.add(joints_pcl, name=f'joints_{frame_name}')

            # remove previous mesh and joints
            for node in scene.get_nodes():
                if node.name is None:
                    continue    
                elif frame_name not in node.name:
                    scene.remove_node(node)
                    # print("Removed node name:", node.name)
            
            # release viwer
            v.render_lock.release()

            # time.sleep(1/fps)
            
        elif plotting_module == 'open3d':
            # Open3D follows the conventional coordinate system is x right, y down, z forward, 
            # but OpenGL for rendering has different convention. 
            # Thus, it is needed to convert the sign of y, z.
            vertices = vertices @ np.diag([1., -1., -1.])
            
            body.vertices = o3d.utility.Vector3dVector(vertices)
            body.triangles = o3d.utility.Vector3iVector(body_model.faces)
            body.vertex_normals = o3d.utility.Vector3dVector([])
            body.triangle_normals = o3d.utility.Vector3dVector([])
            body.compute_vertex_normals()

            if xform_path:
                body.transform(trans)
            
            if i == 0: vis.add_geometry(body)
            else: vis.update_geometry(body)

            if plot_joints:
                joints = joints @ np.diag([1., -1., -1.])
                joints_pcl.points = o3d.utility.Vector3dVector(joints)
                joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
                vis.update_geometry(joints_pcl)
            
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1/fps)
            # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
            
        elif plotting_module == 'matplotlib':
            mesh = Poly3DCollection(vertices[body_model.faces], alpha=0.1)
            update(mesh, joints, draw_skel=False)
            plt.show()
    
    if plotting_module == 'trimesh':
        pass

    elif plotting_module == 'open3d':
        vis.destroy_window()

    elif plotting_module == 'matplotlib':
        pass

def main(args):
    fitting_dir = args.fitting_dir
    model_dir = args.model_dir
    plotting_module = args.plotting_module
    xform_path = args.xform_path

    '''Load your fitting data'''
    # fitting_files = os.listdir(fitting_dir)
    # fitting_file_list = sorted([os.path.join(fitting_dir, file) for file in fitting_files])
    # viz(fitting_file_list, model_dir, plotting_module=plotting_module, xform_path=xform_path)

    prox_dir = '/home/gahyeon/Desktop/data/prox'
    fitting_dir = os.path.join(prox_dir, 'fittings', 'MPH1Library_00034_01', 'results')
    xform_path = os.path.join(prox_dir, 'cam2world', 'MPH1Library.json')
    fitting_files = os.listdir(fitting_dir)
    fitting_file_list = sorted([os.path.join(fitting_dir, file, '000.pkl') for file in fitting_files])

    viz(fitting_file_list, model_dir, plotting_module=plotting_module, num_pca_comps=12, xform_path=xform_path, plot_joints=True)
    

if __name__ == '__main__':
    '''
    Usage: 
        python ./smplx/viz/viz_fitting.py --fitting_dir YOUR_DATA_PATH --model_dir SMPLX_MODEL_PATH --plotting_module open3d

    Example:
        python ./smplx_/viz/viz_fitting.py -f YOUR_DATA_PATH -m ./models/body_models -p open3d
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fitting_dir', required=True, type=str, help='The directory containing SMPL-X fitting data')
    parser.add_argument('-m', '--model_dir', required=True, type=str, default='./body_models', help='The SMPL-X model directory.')
    parser.add_argument('-p', '--plotting_module', type=str, default='matplotlib', help='Renderer for visualizing SMPL-X. Choose pyrender, open3d, or matplotlib')
    parser.add_argument('-x', '--xform_path', type=str, help='The file path containing transformation matrix (cam2world, world2cam, and so on.)')
    args = parser.parse_args()

    main(args)