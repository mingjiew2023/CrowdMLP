import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2
import random

def load_data(img_path, args, train=True):

    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
    img = Image.open(img_path).convert('RGB')

    if train:
        img_path_temp = img_path.replace('images_crop256', 'images')
        img = Image.open(img_path_temp).convert('RGB')

        if args['dataset'] == 'UCF_QNRF'or img_path.split('/')[-4]=='UCF_QNRF_raw':
            mat = io.loadmat(
                img_path.replace('.jpg', '_ann.mat').replace('images', 'gt_density_map').replace('IMG_', 'GT_IMG_'))
            Gt_data = mat["annPoints"]
        else:
            mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'gt_density_map').replace('IMG_', 'GT_IMG_'))
            Gt_data = mat["image_info"][0][0][0][0][0]


        img_data = np.asarray(img).copy()

        rate_1 = 1
        rate_2 = 1

        if img_data.shape[1] >= img_data.shape[0]:  # 后面的大
            rate_1 = 1024.0  / img_data.shape[1]
            rate_2 = 768.0 / img_data.shape[0]
            Img_data = cv2.resize(img_data, (0, 0), fx=rate_1, fy=rate_2)
            Gt_data[:, 0] = Gt_data[:, 0] * rate_1
            Gt_data[:, 1] = Gt_data[:, 1] * rate_2

        elif img_data.shape[0] > img_data.shape[1]:  # 前面的大
            rate_1 = 1024.0 / img_data.shape[0]
            rate_2 = 768.0 / img_data.shape[1]
            Img_data = cv2.resize(img_data, (0, 0), fx=rate_2, fy=rate_1)
            Gt_data[:, 0] = Gt_data[:, 0] * rate_2
            Gt_data[:, 1] = Gt_data[:, 1] * rate_1

        print(img_path)

        kpoint = np.zeros((Img_data.shape[1], Img_data.shape[0]))

        for i in range(0, len(Gt_data)):
            if int(Gt_data[i][1]) < Img_data.shape[0] and int(Gt_data[i][0]) < Img_data.shape[1]:
                kpoint[int(Gt_data[i][0]), int(Gt_data[i][1])] = 1

        gt_count = kpoint
        img = Img_data   #Image.fromarray(Img_data)

    else:
        #img_data = np.asarray(img).copy()
        #rate_1 = (256) / img_data.shape[1]
        #rate_2 = (256) / img_data.shape[0]
        #Img_data = cv2.resize(img_data, (0, 0),fx=rate_1, fy=rate_2)
        #img = Img_data

        while True:
            try:
                gt_file = h5py.File(gt_path)
                gt_count = np.asarray(gt_file['gt_count'])
                break  # Success!
            except OSError:
                print("load error:", img_path)
                cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    gt_count = gt_count.copy()

    return img, gt_count
