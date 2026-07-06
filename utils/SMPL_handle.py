import numpy as np
import os
import torch
import smplx
import h5py
import utils.rotation_conversions as geometry  
from utils.prior import MaxMixturePrior


""""
Visualizattion code is borrowed from  AttT2M, https://github.com/ZcyMonkey/AttT2M/tree/main
"""

def guess_init_3d(config, model_joints, 
                  j3d, 
                  joints_category="SMPL"):
    """Initialize the camera translation via torso esttimation, by using the 4 main joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four (body part，身體主體的4個點)
    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    
    if joints_category=="SMPL":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category=="AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints] 
    else:
        print("NO SUCH JOINTS CATEGORY!") 

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(dim=1)
    init_t = sum_init_t / 4.0
    return init_t

def camera_fitting_loss_3d(config, model_joints, camera_t, camera_t_est,
                           j3d, joints_category="SMPL", depth_loss_weight=100.0):
    """
    Loss function for camera optimization.
    """
    model_joints = model_joints + camera_t
    # # get the indexed four
    # op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
    # op_joints_ind = [config.JOINT_MAP[joint] for joint in op_joints]
    #
    # j3d_error_loss = (j3d[:, op_joints_ind] -
    #                          model_joints[:, op_joints_ind]) ** 2

    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    
    if joints_category=="SMPL":
        select_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category=="AMASS":
        select_joints_ind = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    j3d_error_loss = (j3d[:, select_joints_ind] -
                      model_joints[:, gt_joints_ind]) ** 2

    # Loss that penalizes deviation from depth estimate
    depth_loss = (depth_loss_weight**2) *  (camera_t - camera_t_est)**2

    total_loss = j3d_error_loss +  depth_loss
    return total_loss.sum()

# Guassian
def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(
        pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2

def body_fitting_loss_3d(body_pose, preserve_pose,
                         betas, model_joints, camera_translation,
                         j3d, pose_prior,
                         joints3d_conf,
                         sigma=100, pose_prior_weight=4.78*1.5,
                         shape_prior_weight=5.0, angle_prior_weight=15.2,
                         joint_loss_weight=500.0,
                         pose_preserve_weight=0.0,
                         use_collision=False,
                         model_vertices=None, model_faces=None,
                         search_tree=None,  pen_distance=None,  filter_faces=None,
                         collision_loss_weight=1000
                         ):
    """
    Loss function for body fitting
    """
    batch_size = body_pose.shape[0]

    #joint3d_loss = (joint_loss_weight ** 2) * gmof((model_joints + camera_translation) - j3d, sigma).sum(dim=-1)
    
    joint3d_error = gmof((model_joints + camera_translation) - j3d, sigma)
    
    joint3d_loss_part = (joints3d_conf ** 2) * joint3d_error.sum(dim=-1)
    joint3d_loss = ((joint_loss_weight ** 2) * joint3d_loss_part).sum(dim=-1)
    
    # Pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)
    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)
    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    collision_loss = 0.0
    # Calculate the loss due to interpenetration
    if use_collision:
        triangles = torch.index_select(
            model_vertices, 1,
            model_faces).view(batch_size, -1, 3, 3)

        with torch.no_grad():
            collision_idxs = search_tree(triangles)

        # Remove unwanted collisions
        if filter_faces is not None:
            collision_idxs = filter_faces(collision_idxs)

        if collision_idxs.ge(0).sum().item() > 0:
            collision_loss = torch.sum(collision_loss_weight * pen_distance(triangles, collision_idxs))
    
    pose_preserve_loss = (pose_preserve_weight ** 2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

    total_loss = joint3d_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + collision_loss + pose_preserve_loss

    return total_loss.sum()

class SMPLify3D:
     def __init__(self,
                 config, 
                 smplxmodel,
                 step_size=1e-2,
                 batch_size=1,
                 num_iters=100,
                 use_collision=False,
                 use_lbfgs=True,
                 joints_category="SMPL",
                 device=torch.device('cuda:0'),
                 ):

        # Store options
        self.config =config
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # --- choose optimizer
        self.use_lbfgs = use_lbfgs
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=config.GMM_MODEL_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # collision part
        #self.use_collision = use_collision
        #if self.use_collision:
        #    self.part_segm_fn = config.Part_Seg_DIR
        
        # reLoad SMPL-X model
        self.smpl = smplxmodel

        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # select joint joint_category
        self.joints_category = joints_category
        
        if joints_category=="SMPL":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx 
        elif joints_category=="AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        else:
            self.smpl_index = None 
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

     def __call__(self, init_pose, init_betas, init_cam_t, j3d, conf_3d=1.0):
        # 根據第 0 幀 (initial pose) 去對齊後續的動作，並透過optimize來優化模型體型以配合骨架來產生[seq, 24, 3]的motion tensor
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints [seq, 24, 3]
            conf_3d: confidence for 3d joints [24]
			seq_ind: index of the sequence
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape [seq, 24, 3]
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas)
        model_joints = smpl_output.joints

        # Estimate the pose trnslation by a simple torso estimation (初始身軀位移估計)
        init_cam_t = guess_init_3d(self.config, model_joints, j3d, self.joints_category).unsqueeze(1).detach()
        camera_translation = init_cam_t.clone()
        preserve_pose = init_pose[:, 3:].detach().clone()

        #Step 1: Optimize camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=self.num_iters,
                                                 lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(10):
                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    # print('model_joints', model_joints.shape)
                    # print('camera_translation', camera_translation.shape)
                    # print('init_cam_t', init_cam_t.shape)
                    # print('j3d', j3d.shape)
                    loss = camera_fitting_loss_3d(self.config, model_joints, camera_translation,
                                                  init_cam_t, j3d, self.joints_category)
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(20):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints

                loss = camera_fitting_loss_3d(self.config,  model_joints[:, self.smpl_index], camera_translation,
                                              init_cam_t,  j3d[:, self.corr_index], self.joints_category)
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        #Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        body_opt_params = [body_pose, betas, global_orient, camera_translation]

        # For loss Calculation, Default is None
        search_tree = None
        pen_distance = None
        filter_faces = None

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.num_iters,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas)
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices

                    loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                                j3d[:, self.corr_index], self.pose_prior,
                                                joints3d_conf=conf_3d,
                                                joint_loss_weight=600.0,
                                                pose_preserve_weight=5.0,
                                                model_vertices=model_vertices, model_faces=self.model_faces,
                                                search_tree=search_tree, pen_distance=pen_distance, filter_faces=filter_faces)
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                            j3d[:, self.corr_index], self.pose_prior,
                                            joints3d_conf=conf_3d,
                                            joint_loss_weight=600.0,
                                            model_vertices=model_vertices, model_faces=self.model_faces,
                                            search_tree=search_tree,  pen_distance=pen_distance,  filter_faces=filter_faces)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

         # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            model_vertices = smpl_output.vertices

            final_loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                              j3d[:, self.corr_index], self.pose_prior,
                                              joints3d_conf=conf_3d,
                                              joint_loss_weight=600.0,
                                              use_collision=False, model_vertices=model_vertices, model_faces=self.model_faces,
                                              search_tree=search_tree,  pen_distance=pen_distance,  filter_faces=filter_faces)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()  #[seq, 24, 3]
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, final_loss



class joints2smpl:
    def __init__(self, config, num_frames, device, cuda=True):
        self.device = device
        # self.device = torch.device("cpu")
        self.batch_size = num_frames
        self.num_joints = 22  # SMPL Standard
        self.joint_category = "AMASS"
        self.num_smplify_iters = int(getattr(config, "smpl_fit_iters", 150))
        self.fix_foot = False
        #print(config.SMPL_MODEL_DIR)
        smplmodel = smplx.create(config.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl",
                                 batch_size=self.batch_size).to(self.device)

        # ## --- load the mean pose as original ----
        smpl_mean_file = config.SMPL_MEAN_FILE

        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel, config=config,
                            batch_size=self.batch_size,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            device=self.device)


    def __call__(self, input_joints):
        pred_pose = torch.zeros(self.batch_size, 72).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)

        # run the whole seqs
        num_seqs = input_joints.shape[0]


        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.as_tensor(input_joints, dtype=torch.float32, device=self.device) # [B, L, 24, 3]

        #init pose setup (according to pre-defined file)
        pred_betas = self.init_mean_shape
        pred_pose = self.init_mean_pose   # []
        pred_cam_t = self.cam_trans_zero
        # raise the optimization weight on feet (增加腳部貼地權種正確性避免滑動)
        confidence_input = torch.ones(self.num_joints)
        confidence_input[7] = 1.5
        confidence_input[8] = 1.5
        confidence_input[10] = 1.5
        confidence_input[11] = 1.5

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = self.smplify(
            pred_pose.detach(), # init pose data
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d.detach(),  # motion sequence
            conf_3d=confidence_input.to(self.device),
        )

        thetas = new_opt_pose.reshape(self.batch_size, 24, 3)
        thetas = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(thetas))  # [B=seq, 24, 6]
        root_loc = keypoints_3d[:, 0].detach().clone()  # [bs, 3]
        root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-1).unsqueeze(1)  # [seq, 1, 6]
        thetas = torch.cat([thetas, root_loc], dim=1).unsqueeze(0).permute(0, 2, 3, 1)  # [1, 25, 6, seq]

        return thetas.clone().detach(), {'pose': new_opt_joints[0, :24].flatten().clone().detach(), 'betas': new_opt_betas.clone().detach(), 'cam': new_opt_cam_t.clone().detach()}
