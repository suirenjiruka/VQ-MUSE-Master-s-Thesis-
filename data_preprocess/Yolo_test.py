from ultralytics import YOLO
import ultralytics.utils.ops as ops_module
import os
import cv2
import numpy
import json
import torch
import pandas as pd
import numpy as np


model = YOLO('yolov8m-pose.pt')
dir_path = "./mogen_video"
#index_pth= "C:\\Users\\imlab\\Desktop\\T2M\\HumanML3D\\index.csv"
dir = os.listdir(dir_path)
frame_buffer = []
frame_size = 7
count = 0

original_clip_coords = ops_module.clip_coords
def no_clip_coords(coords, shape):
    return coords   # 什麼都不做，保留原始值
ops_module.clip_coords = no_clip_coords

#index_file = pd.read_csv(index_pth)
idx_count = 0

for item in dir:
    if item.lower().endswith(".mp4"):
        #print(dir_path + "/" + item)
        output = []
        cap = cv2.VideoCapture(dir_path +"/" + item)

        #畫面設定
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps = cap.get(cv2.CAP_PROP_FPS)

        #輸出設定
        os.makedirs("./eval_joints", exist_ok=True)
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #out = cv2.VideoWriter("./eval_joints/" + item[:-4] + '_output.mp4', fourcc, fps, (width, height))
        end = 0
        while cap.isOpened():
            ret, frame = cap.read()
            #print(idx_count)
            #print(f"frame:{frame}")
            if not ret:
                if len(frame_buffer) <= 0:
                    break
                while len(frame_buffer) < frame_size:
                    frame_buffer.append(frame_buffer[-1])
                end = 1
            else:
                frame_buffer.append(frame)
                
            if(len(frame_buffer) < frame_size):
                  continue

            results = model.predict(frame_buffer, save=False, conf=0.7)
            window = []
            for result in results:
                try:
                    keypoints = result.keypoints.xyn.cpu().numpy() #(frame, xyz)
                    keypoints = np.concatenate(([keypoints[0][0]], keypoints[0][5:]))
                    keypoints = keypoints.tolist()
                    #print(f"keypoint : {keypoints}")
                    window.append(keypoints)
                    #annotated_frame = result.plot()
                    #out.write(annotated_frame)
                except:
                    window.append(np.zeros([13,2]).tolist())
                    print("error")
                    #out.write(frame)
            output.append(window)
            frame_buffer = []
            if(end == 1):
                break
        # while(item != index_file.loc[count]['new_name'].replace(".npy", ".mp4")):
        #           print(item, index_file.loc[count]['new_name'].replace(".npy", ".mp4"))
        #           count += 1
        #output = output[index_file.loc[count]['start_frame']:index_file.loc[count]['end_frame']]
        with open("./eval_joints/" + item[:-4] +".json", "w") as f:
            json.dump(output, f, indent=2)
        # output_m = output.copy()
        # for frame in output_m:
        #     for joint in frame:
        #         joint[0] = 1 -joint[0]
        
        # with open("./test_video\\M" + item[:-4] +".json", "w") as f:
        #                 json.dump(output_m, f, indent=2)
        count+=1
        cap.release()
        #out.release()