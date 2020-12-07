import os
import numpy as np
from PIL import Image
import cv2
calib_images_path = './calib_images'
calib_batch_size = 47

def get_calib_data(iter):
    """
    Function provides calibration images to the quantizer from the training set
    """

    frames = os.listdir(calib_images_path)
    # np.random.shuffle(frames)
    num_frames = len(frames)
    print("number of calibration images : ", num_frames)

    out_train_x_normalized = np.zeros((calib_batch_size, 200, 200, 3))

    frame_indices = list(range(iter*calib_batch_size, calib_batch_size + (iter * calib_batch_size)))
    print(frame_indices)
    for i, frame in enumerate(frame_indices):
        f_path = calib_images_path + '/' + frames[frame]
        print(f_path)
        im = cv2.imread(f_path) #Image.open(f_path)
        file = cv2.resize(im,(200,200))# np.array(im)
        # file = file[0: 200, 0: 200, :]
        out_train_x_normalized[i] = np.expand_dims(file, axis=0)  # depth channel
        out_train_x_normalized /= np.max(out_train_x_normalized)  # normalize
    
    return {"conv2d_1_input": out_train_x_normalized}
    

if __name__ == "__main__":
    # keras_convert(keras_json=None, tf_ckpt=None, keras_hdf5='/home/sambit/Xilinx_Works/Vitis-AI-Tutorials-DenseNetX_DPUv2/files/from_docker_tf/trained_unet_1ch_input.h5')
    for i in range(10):
        get_calib_data(i)
