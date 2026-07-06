import torch 
import matplotlib.pyplot as plt
from trimesh import Trimesh
import numpy as np
import os, argparse, math
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import matplotlib
from ultralytics import YOLO
import ultralytics.utils.ops as ops_module
from os.path import join as pjoin
import cv2

import imageio
import pyrender
from pyrender.constants import RenderFlags

from SnapMogen_model.vq.rvq_model import HRVQVAE
from model import VAMotion
from dataset.PR_VIPE_end import read_input, pr_vipe_infer
from configs.load_config import load_config
from utils.SMPL_handle import joints2smpl
from utils.rotation2xyz import Rotation2xyz
from utils.motion_process_bvh import recover_pos_from_rot, recover_pos_from_ric, contact_joint_names
from scipy.ndimage import gaussian_filter1d
from utils.skeleton import Skeleton
from utils import bvh_io
from utils.utils import plot_3d_motion
from data_preprocess.keypoint_normalize import normalize

# 關閉出界joint的Clip
original_clip_coords = ops_module.clip_coords
def no_clip_coords(coords, shape):
    return coords   # 什麼都不做，保留原始值
ops_module.clip_coords = no_clip_coords


model = YOLO('yolov8m-pose.pt')

#python -m master_project.utils.visualization --config /utils/visual_config.yaml  --PrVipe False

def visualize(name, motion, device, config, pred = True):
    frames, njoints, nfeats = motion.shape # [Seq_len, 24, 3]
    motion_np = motion.detach().cpu().numpy()
    MINS = motion_np.min(axis=0).min(axis=0)
    MAXS = motion_np.max(axis=0).max(axis=0)

    print("Motion Max:", motion.max().item())
    print("Motion Min:", motion.min().item())

    height_offset = MINS[1]  # floor height
    motion[:, :, 1] -= height_offset
    trajec = motion[:, 0, [0, 2]] # pelvis pos (withoutt height)

    j2s = joints2smpl(config, num_frames=frames, device=device, cuda=True)
    rot2xyz = Rotation2xyz(config, device=device)
    faces = rot2xyz.smpl_model.faces

    # SMPLify
    # 最一開始進來的SMPL似乎不是用XYZ 而是用 rotation matrix表示 => 我們需要想辦法在把motion套上去後將其轉回xyz
    motion = motion / 50.0
    motion_tensor, opt_dict = j2s(motion)  # [nframes, njoints, 3] -> motion_tensor([1, 25, 6, seq]), opt_dict: {pose ([seq, 24, 3]), beta (體型), camera}

    vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                    pose_rep='rot6d', translation=True, glob=True,
                                    jointstype='vertices', betas = (opt_dict["betas"] / 4),  # beta could be our optimized tensor or Default (None) 
                                    vertstrans=True)
    frames = vertices.shape[3] # [1, nb_frames, 3, nb_joints]
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0] # return [x_min, y_min, z_min] 
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0] # return [x_max, y_max, z_max]

    out_list = []
    #add margin
    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    vid = []
    # mesh color setting
    bg_color = [1, 1, 1, 0.8]
    base_color = [(245/ 255.0,222/ 255.0,179/ 255.0,0.1),
                  (255/ 255.0,215/ 255.0,0/ 255.0,0.3),
                  (237/ 255.0,145/ 255.0,33/ 255.0,0.5),
                  (255/ 255.0,128/ 255.0,0/ 255.0,0.7)]

    n = 0
    for i in range(frames):
        if i % 1 ==0:
            # Use TTrimesh to build the mesh model by atttained vertices
            mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)
            #base_color_var = base_color[n % 4]
            base_color_var = [255/ 255.0,(145+n*0.8)/ 255.0,(33+n*0.5)/ 255.0,0.9]
            n += 1

            ## OPAQUE rendering without alpha
            ## BLEND rendering consider alpha
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.5,
                alphaMode='BLEND',
                baseColorFactor=base_color_var
            )


            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
            sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]
            scene.add(mesh)

            #Camera & light shading settting
            camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

            light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

            c = np.pi / 2

            light_pose = np.eye(4)
            light_pose[:3, 3] = [0, -1, 1]
            scene.add(light, pose=light_pose.copy())

            light_pose[:3, 3] = [0, 1, 1]
            scene.add(light, pose=light_pose.copy())

            light_pose[:3, 3] = [1, 1, 2]
            scene.add(light, pose=light_pose.copy())


            c = -np.pi / 6

            # Fix: track current frame center instead of global sequence center
            frame_verts = vertices[0, :, :, i]  # [nb_verts, 3]
            cx = frame_verts[:, 0].mean().cpu().numpy()
            cz_min = frame_verts[:, 2].min().cpu().numpy()
            cam_z = max(4, float(cz_min) + (1.5 - MINS[1].cpu().numpy()) * 2, float((maxx - minx).cpu().numpy()))

            scene.add(camera, pose=[[ 1, 0, 0, cx],
                                    [ 0, np.cos(c), -np.sin(c), 1.5],
                                    [ 0, np.sin(c), np.cos(c), cam_z],
                                    [ 0, 0, 0, 1]
                                    ])

            # render scene
            r = pyrender.OffscreenRenderer(960, 960)

            color, _ = r.render(scene, flags=RenderFlags.RGBA)
            # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

            vid.append(color)

            r.delete()

    outdir = config.Output_DIR
    out = np.stack(vid, axis=0)
    if pred:
        if not os.path.exists(outdir + name):
            os.makedirs(outdir + name)
        for k in range(int(len(out)/3)):  #color 是每3個channel 算一幀
            imageio.imwrite(outdir + name+'/'+str(k*3)+'_pred.png', np.squeeze(out[k*3]))
        imageio.mimsave(outdir + name+'/'+'pred.gif', out, fps=20)
    else:
        imageio.imsave(outdir + name+'_gt.png', out)

#根據motion prediction output (ID?) 透過BVH骨架轉換回jointt position (XYZ)
def forward_kinematic_func(data, skeleton, mean, std):
    # inverse Normalization
    if isinstance(data, np.ndarray):
        motions = data * std[:data.shape[-1]] + mean[:data.shape[-1]]
    elif isinstance(data, torch.Tensor):
        motions =  data * torch.from_numpy(std[:data.shape[-1]]).float().to(
            data.device
        ) + torch.from_numpy(mean[:data.shape[-1]]).float().to(data.device)
    else:
        raise TypeError("Expected data to be either np.ndarray or torch.Tensor")
    # calculate the xyz pos from predicted motion ID
    global_pos = recover_pos_from_rot(motions, 
                                      joints_num=cfg.data.joint_num, 
                                      skeleton=skeleton)
    return global_pos

def video_process(input_type, video_pth, debug_video=False):
    if input_type == "video":
        frame_buffer = []
        frame_size   = 7
        data         = []
 
        cap    = cv2.VideoCapture(video_pth)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
 
        if debug_video:
            CANVAS_SCALE = 3
            canvas_w = width  * CANVAS_SCALE
            canvas_h = height * CANVAS_SCALE
            pad_x    = (canvas_w - width)  // 2
            pad_y    = (canvas_h - height) // 2
            fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = "./visualize/" + video_pth.split("/")[-1] + '_debug.mp4'
            out      = cv2.VideoWriter(out_path, fourcc, fps, (canvas_w, canvas_h))
            SKEL     = [(0,1),(0,2),(1,2),(1,3),(3,5),(2,4),(4,6),
                        (1,7),(2,8),(7,8),(7,9),(9,11),(8,10),(10,12)]
 
        end = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if len(frame_buffer) <= 0:
                    break
                while len(frame_buffer) < frame_size:
                    frame_buffer.append(frame_buffer[-1])
                end = 1
            else:
                frame_buffer.append(frame)
 
            if len(frame_buffer) < frame_size:
                continue
 
            results = model.predict(frame_buffer, save=False, conf=0.7)
            window  = []
 
            for result in results:
                try:
                    keypoints = result.keypoints.xyn.cpu().numpy()  # [person, 17, 2]
 
                    kp = np.concatenate(([keypoints[0][0]], keypoints[0][5:]))  # [13, 2]
 
                    kp_norm = normalize(kp, True, height=height, width=width)
                    window.append(kp_norm)
 
                    # ── Debug 繪圖（僅在 debug_video=True 時執行）───────────
                    if debug_video:
                        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        canvas[pad_y:pad_y+height, pad_x:pad_x+width] = result.plot(conf=False)
                        cv2.rectangle(canvas,
                                      (pad_x, pad_y),
                                      (pad_x+width-1, pad_y+height-1),
                                      (128,128,128), 2)
                        pts = [(int(kp[j, 0]*width) + pad_x,
                                int(kp[j, 1]*height) + pad_y) for j in range(13)]
                        for a, b in SKEL:
                            cv2.line(canvas, pts[a], pts[b], (0, 200, 0), 2)
                        for xp, yp in pts:
                            cv2.circle(canvas, (xp, yp), 6, (0, 255, 0), -1)
                        out.write(canvas)
 
                except Exception as e:
                    # except 補全改用前一窗口最後一幀 
                    if len(data) > 0 and len(data[-1]) > 0:
                        last_valid = np.array(data[-1][-1])   # last frame in previous window
                        window.append(last_valid.tolist())
                    else:
                        window.append(np.zeros([13, 2]).tolist())
                    print(f"[video_process] detection failed: {e}")
 
                    if debug_video:
                        canvas_blank = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        out.write(canvas_blank)
            data.append(window)
            frame_buffer = []
            if end == 1:
                break
        cap.release()
        if debug_video:
            out.release()
 
        data = np.array(data)   # (windows, 7, 13, 2)
 
    elif input_type == "joints":
        data = np.array(read_input(video_pth))
 
    return data


     
if __name__ == "__main__": 
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default="./utils/visual_config.yaml",
                        help='config file for visualiztion setting')
    parser.add_argument('--vqcfg',
                        type=str,
                        default="./configs/residual_vqvae.yaml",
                        help='config file for VQ model')
    parser.add_argument('--trans_cfg',
                        type=str,
                        default="./configs/train_vamotion.yaml",
                        help='config file for VA mottion training')
    parser.add_argument('--PrVipe',
                        type=bool,
                        default=True,
                        help='Whether using Pr-vipe embedding')
    args = parser.parse_args()
    config = load_config(args.config)
    # config loading
    vq_cfg = load_config(args.vqcfg)
    cfg = load_config(args.trans_cfg)
    cfg.vq = vq_cfg.quantizer
    dataset = cfg.data.name
    device = torch.device(cfg.exp.device)

    # 骨架姿勢平均值 這和skeleton都是為了將motion還原成骨架用的參考
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    Mean = np.load(pjoin(meta_dir, 'mean.npy'))
    Std = np.load(pjoin(meta_dir, 'std.npy'))

    # Skeleton setup (SnapMogen only)
    if dataset == 'snapmogen':
        template_anim = bvh_io.load(pjoin(cfg.data.root_dir, 'renamed_bvhs', 'm_ep2_00086.bvh'))
        skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)
        anim_joints_dict = {template_anim.names[i]: i for i in range(len(template_anim.names))}
        foot_contact_ids = [anim_joints_dict[n] for n in contact_joint_names if n in anim_joints_dict]
    else:
        skeleton = None
        foot_contact_ids = []

    # set up testing caption & video input
    data_num = 10  # depend on testing
    #caption setup
    caption = []
    caption.append("The person is peeking into a room, appearing to realize they've forgotten something, and is about to walk away from the door.")
    caption.append("")
    caption.append("The guy is walking from right to left,and left to right over again.")
    caption.append("The person is peeking into a room, appearing to realize they've forgotten something, and is about to walk away from the door.")
    caption.append("The person is seated on a couch and interacting with their smartphone, manipulating the phone with the hands.") #
    caption.append("The person is excitely doing a high jump to touch a high ceiling, quickly scurrying and then doing a high jump and raise one arm to touch ceiling.")
    caption.append("The person is excitely doing a high jump to touch a high ceiling, quickly scurrying and then doing a high jump and raise one arm to touch ceiling.")
    caption.append("The person is performing a motion that emulates the 'crane kick' pose from the movie 'The Karate Kid.' They jump into the air and kick one of his leg on air, them fall and taking a pose and shout.") #
    caption.append("A person bends forward and picks something from the floor, and then walk straight leaving away.")
    caption.append("A man prepare to catch a baseball, shift their weight, move the glove hand towards and catch the incoming ball")
    #caption.append("A man is running on the bank, and then jumping into the pool in a swimming pose.")
    #video settup
    video_joints = {}
    video_joints["0"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Enter_a_Room_Missing_Something_clip1.mp4")
    video_joints["1"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Enter_a_Room_Missing_Something_clip1.mp4")
    video_joints["2"] = np.zeros((1, 7, 15, 2)) #video_joints("video", "")
    video_joints["3"] = np.zeros((1, 7, 15, 2)) 
    video_joints["4"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Text_KEVINBPARRY_clip1.mp4")
    video_joints["5"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Jump_+_Sit_+_Fall_Low_Ceiling_Touch_clip2.mp4")
    video_joints["6"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Jump_+_Sit_+_Fall_Low_Ceiling_Touch_clip1.mp4")
    video_joints["7"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Jump_+_Sit_+_Fall_Karate_Kid_clip1.mp4")
    video_joints["8"] = video_process("video", "/home/imlab/momask_data/eval_video/Take_something.mp4")
    video_joints["9"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Catch_Baseball_Glove_clip1.mp4")
    #video_joints["10"] = video_process("video", "/home/imlab/momask_data/eval_video/Ways_to_Jump_In_+_Swim_+_Get_Out_of_a_Pool_Lifeguard_clip1.mp4")

    T_drop = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    V_drop = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

    video_input = []
    v_lens = []
    joint_input=[]
    j_lens=[]
    #Pr_vipe extracttion
    max_vipe_len  = (cfg.data.max_motion_length // 7) + 2
    max_joint_len = cfg.data.max_motion_length

    if args.PrVipe:
        video_dict = pr_vipe_infer(config.vipe_checkpt, video_joints)
        for data in video_dict.values():
            v_len, dim = data[0].shape
            if v_len < max_vipe_len:
                mean = np.concatenate([data[0], np.zeros((max_vipe_len - v_len, dim))], axis=0)
                mean = torch.as_tensor(mean, device=device, dtype=torch.float32).unsqueeze(0)
                std = np.concatenate([data[1], np.zeros((max_vipe_len - v_len, dim))], axis=0)
                std = torch.as_tensor(std, device=device, dtype=torch.float32).unsqueeze(0)
            video_input.append((mean, std))
            v_lens.append(v_len)
        v_lens = torch.as_tensor(v_lens, device=device, dtype=torch.float32).unsqueeze(1)

    for data in video_joints.values():
        data = data.reshape(-1, 30)
        j_len = len(data)
        if j_len < max_joint_len:
            data = np.concatenate(
                [data, np.zeros((max_joint_len - j_len, data.shape[1]))], axis=0)
        else:
            data = data[:max_joint_len]
        joint_input.append(data)
        j_lens.append(j_len)
    joint_input = torch.as_tensor(joint_input, device=device, dtype=torch.float32).unsqueeze(1)
    j_lens = torch.as_tensor(j_lens, device=device, dtype=torch.float32).unsqueeze(1)

    # generate predivtion motion
    vq_model = HRVQVAE(vq_cfg,
            vq_cfg.data.dim_pose,
            vq_cfg.model.down_t,
            vq_cfg.model.stride_t,
            vq_cfg.model.width,
            vq_cfg.model.depth,
            vq_cfg.model.dilation_growth_rate,
            vq_cfg.model.vq_act,
            vq_cfg.model.use_attn,
            vq_cfg.model.vq_norm)
    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model',cfg.vq_ckpt),
                            map_location=device, weights_only=True) # To future Ina， 正式training時要記得調整
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key]) # type: ignore

    trans = VAMotion(
       cfg = cfg,
       device = device,
       full_length= cfg.data.max_motion_length // cfg.data.unit_length,
       max_video_len = (cfg.data.max_motion_length // 7) + 2,  # The rermainder parts occupy one extra sequence frame at the head and tail respectively, hence +2. ex: (320 // 7) + 2 = 47
       video_joint_num= None
    )
    checkpoint = torch.load(pjoin(cfg.exp.root_ckpt_dir, cfg.data.name, 'VA_motion', 'model', 'current_best.tar'), map_location=device, weights_only=True)
    trans.load_state_dict(checkpoint["training_model"])

    # tran all data into GPU
    vq_model.to(device)
    trans.to(device)

    for i in range(data_num):
        m_length = torch.randint(7, 10, (1,), device=device) * 30
        print(video_input[i][0].shape, joint_input[i].shape)
        mids = trans.generate(video_input[i], caption[i], m_length // 4, v_lens[i], joint_input[i], j_lens[i], T_drop[i], V_drop[i], 16, 4, 1.0, temperature=1)
        pred_motions = vq_model.forward_decoder(mids, m_length.clone())
        print(f"prdiction shape: {pred_motions.shape}")

        if dataset == "snapmogen":
            # turn predict motion into [L, joints (24), 3]
            #visualize 
            global_pos = forward_kinematic_func(pred_motions, skeleton, Mean, Std).squeeze(dim=0)
            # Fix: smooth to reduce jitter from cumulative velocity integration errors
            global_pos_np = global_pos.detach().cpu().numpy()
            global_pos_np = gaussian_filter1d(global_pos_np, sigma=1.0, axis=0)

            # Fix: foot contact snapping to prevent floor skating
            if foot_contact_ids:
                foot_height_thresh = 5.0  # units same as BVH data
                foot_vel_thresh = 0.11
                for fid in foot_contact_ids:
                    foot_y = global_pos_np[:, fid, 1]
                    foot_vel = np.sqrt(np.sum(np.diff(global_pos_np[:, fid, :], axis=0)**2, axis=-1))
                    contact = (foot_y[:-1] < foot_height_thresh) & (foot_vel < foot_vel_thresh)
                    for t in range(len(contact)):
                        if contact[t]:
                            global_pos_np[t, fid, 1] = max(0.0, global_pos_np[t, fid, 1])

            global_pos = torch.from_numpy(global_pos_np).to(global_pos.device)
            print(f"global pos shape: {global_pos.shape},m_len:{m_length}")
            # bvh 骨架joint format
            #kinematic_chain = [[0, 1, 2, 3, 4, 5, 6],
            #    [3, 7, 8, 9, 10],
            #    [3, 11, 12, 13, 14],
            #    [0, 15, 16, 17, 18, 19],
            #   [15, 20, 21, 22, 23]]
            kinematic_chain = [[0, 1, 2, 3, 4, 6],
                [3, 7, 8, 9, 10],
                [3, 11, 12, 13, 14],
                [0, 16, 17, 18, 19],
                [0, 20, 21, 22, 23]]
            bvh_to_smpl_map = [0, 16, 20, 1, 17, 21, 2, 18, 22, 3, 19, 23, 4, 7, 11, 6, 8, 12, 9, 13, 10, 14]
            global_pos_smpl = global_pos[:, bvh_to_smpl_map, :]

        elif dataset == "humanml3d":
            motions = pred_motions * torch.from_numpy(Std[:pred_motions.shape[-1]]).float().to(device) \
                      + torch.from_numpy(Mean[:pred_motions.shape[-1]]).float().to(device)
            global_pos = recover_pos_from_ric(motions, joints_num=cfg.data.joint_num - 1, hml3d=True).squeeze(0)  # [T, 22, 3]
            global_pos_np = global_pos.detach().cpu().numpy()
            global_pos_np = gaussian_filter1d(global_pos_np, sigma=1.2, axis=0)
            global_pos = torch.from_numpy(global_pos_np).to(global_pos.device)
            # Sanity check on ground truth data
            import os
            gt_path = os.path.join(cfg.data.feat_dir, '000000.npy')
            if os.path.exists(gt_path):
                gt_raw = np.load(gt_path)[:1]  # first frame
                gt_mean = np.load(os.path.join(cfg.data.root_dir, 'meta_data', 'mean.npy'))
                gt_std  = np.load(os.path.join(cfg.data.root_dir, 'meta_data', 'std.npy'))
                gt_feat = torch.from_numpy((gt_raw - gt_mean) * 0 + gt_raw).float().to(device)  # un-normalized GT
                gt_pos = recover_pos_from_ric(gt_feat, joints_num=cfg.data.joint_num - 1, hml3d=True)
                # gt_pos may be [1,22,3] or [22,3] depending on input shape
                if gt_pos.dim() == 3:
                    gt_pos = gt_pos[0]   # [22, 3]
            global_pos_smpl = global_pos
            # Standard SMPL 22-joint chain (joint 0 = pelvis/root)
            kinematic_chain = [
                [0, 1, 4, 7, 10],      # pelvis-lhip-lknee-lankle-lfoot
                [0, 2, 5, 8, 11],       # pelvis-rhip-rknee-rankle-rfoot
                [0, 3, 6, 9, 12, 15],   # pelvis-spine1-spine2-spine3-neck-head
                [9, 13, 16, 18, 20],    # spine3-lcol-lsho-lelbow-lwrist
                [9, 14, 17, 19, 21],    # spine3-rcol-rsho-relbow-rwrist
            ]
            
        hml_radius = 4 if dataset == "humanml3d" else 100
        plot_3d_motion(f"./visualize/skeleton_{i}.mp4", kinematic_chain,
                       global_pos[:m_length].detach().cpu().numpy(),
                       title=caption[i], fps=30, radius=hml_radius)
        #visualize(f"visual_{i}", global_pos_smpl[:m_length], torch.device(cfg.exp.device), config)

