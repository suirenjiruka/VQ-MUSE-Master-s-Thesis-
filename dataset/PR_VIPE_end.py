import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import json
import numpy as np
import gc

from tqdm import tqdm
from poem.core import common
from poem.core import input_generator
from poem.core import keypoint_profiles
from poem.core import keypoint_utils
from poem.core import models
from poem.core import pipeline_utils
tf.disable_v2_behavior()

frame_length = 7
key_point_num = 13
image_sizes = [720, 1280]


def read_input(json_path):
    # read 1 file per time, repeat beyond the function call for all data
    with open(json_path, "r")as jfile:
        data = json.load(jfile)
    return data

#deprecated call
def read_data_by_dir(dir_path):
    files = os.listdir(dir_path)
    output_list = {}
    for file in files:
       path = os.path.join(dir_path, file)
       if os.path.isfile(path):
        data = read_input(path)
        output_list[file.split(".")[0]] = data
    return output_list


def pr_vipe_infer(check_point_path, data_2D: dict):  #infer from pr-vipe, return a list of Gaussian distribution (Mean, Std deviation)
    
    keypoint_profile_2d = (keypoint_profiles.create_keypoint_profile_or_die("LEGACY_2DH36M13"))

    data_len_index = {}
    keypoints_2d = np.zeros((0, frame_length, key_point_num, 2))
    cord_2d = np.zeros((0, frame_length, 3))
    for mid, data in tqdm(data_2D.items()):    #data = (data_chunk num, frames(7), keypoints (13 + 2), 2)
        cord = data[:, :, -2:, :]
        # calculate position cordinate (pelvis position to represent the overall cordinate)
        cord = cord.reshape([len(cord), frame_length, -1])[:, :, :-1]  #[d, frame, 2, 2] -> [d, frame, 3 (x, y, z)]
        data = data[:, :, :-2, :]
        data_len_index[mid]= len(data) # len = d
        keypoints_2d = np.concatenate((keypoints_2d, data), axis = 0) #全部拼起來 => [total data num, frame(7), 13, 2]
        cord_2d = np.concatenate((cord_2d, cord), axis = 0)  #全部拼起來 => [total data num, total frame(7), 3]
  
    g = tf.Graph()
    with g.as_default():

        keypoints_2d =  tf.constant(keypoints_2d, dtype=tf.float32)
        keypoints_2d = tf.reshape(keypoints_2d, [-1, key_point_num, 2])
        total_num = int(keypoints_2d.shape[0]) #all frames in dataset
        keypoints_2d = keypoint_utils.denormalize_points_by_image_size(keypoints_2d, image_sizes=tf.constant([image_sizes] * total_num))
        keypoints_2d = tf.reshape(keypoints_2d, [-1, frame_length, key_point_num, 2])  # for 2d key points input
        keypoint_masks_2d = tf.ones([key_point_num, total_num], dtype=tf.float32)

        #print(keypoints_2d.shape, keypoint_masks_2d.shape)

        #裁切輸入inference的input format
        model_inputs, _ = input_generator.create_model_input( 
            keypoints_2d,
            keypoint_masks_2d=keypoint_masks_2d,
            keypoints_3d=None,
            model_input_keypoint_type=common.MODEL_INPUT_KEYPOINT_TYPE_2D_INPUT,
            model_input_keypoint_mask_type= 'NO_USE', #FLAGS.model_input_keypoint_mask_type
            keypoint_profile_2d = keypoint_profile_2d,
            # Fix seed for determinism.
            seed=1)

        #import embedder
        embedder_fn = models.get_embedder(
            base_model_type="TEMPORAL_SIMPLE",
            embedding_type='GAUSSIAN',
            num_embedding_components=1,
            embedding_size=16,    #output dimension size (13 keypoints + 3 transformation factor (xyz))
            num_embedding_samples=20,
            is_training=False,
            num_fc_blocks=2,
            num_fcs_per_block=2,
            num_hidden_nodes=1024,
            num_bottleneck_nodes=0,
            weight_max_norm=0.0)
        
        outputs, _ = embedder_fn(model_inputs)

        variables_to_restore = (pipeline_utils.get_moving_average_variables_to_restore())
        saver = tf.train.Saver(variables_to_restore)

        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(), saver=saver)
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=scaffold,
            master='',
            checkpoint_filename_with_path=check_point_path)
        output_data = {}

        with tf.train.MonitoredSession(session_creator=session_creator, hooks=None) as sess:
            outputs_result = sess.run(outputs)

        mean = outputs_result[common.KEY_EMBEDDING_MEANS]
        stddev = outputs_result[common.KEY_EMBEDDING_STDDEVS]
        cord_mean = np.mean(cord_2d, axis = 1).astype(np.float32) #[total num, 7(frame), 3] -> [total num, 3]
        cord_stddev = np.std(cord_2d, axis = 1).astype(np.float32)

        assert len(cord_mean) == len(cord_stddev)
        assert len(cord_mean) == len(mean)
        assert len(cord_stddev) == len(stddev)

        start = 0
        for mid, length in tqdm(data_len_index.items()):
            out = [mean[start: start + length].reshape([length,-1]), stddev[start: start + length].reshape([length,-1])] #[length, 16]
            out[0] = np.concatenate((out[0], cord_mean[start: start + length]), axis = 1)  # concatenate mean of pelvis cordinate (center point)
            out[1] = np.concatenate((out[1], cord_stddev[start: start + length]), axis =1) # and stddev to represent the translation of pose
            #print(output_data[mid][0].shape)
            output_data[mid] = (out[0], out[1])
            start += length
    del sess, keypoints_2d, cord_2d, outputs_result, mean, stddev
    gc.collect()
    return output_data #dict[]: tuple([length(47), dim(19)], [length(47), dim(19)]) (mean, stddev)

if __name__ == '__main__':
    output_list = read_data_by_dir("/mnt/c/Users/USER/Desktop/Tzu-Hsuan/normal_joints")
    out_data = pr_vipe_infer("/mnt/c/Users/USER/Desktop/Tzu-Hsuan/master_project/models/model.ckpt-02059146", output_list)
    print(out_data)