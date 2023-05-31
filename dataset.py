import torch
from torch.utils.data import Dataset
import os
import random
from image import *
import numpy as np
import numbers
from torchvision import datasets, transforms
import torch.nn.functional as F

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)

def random_rotate(x):
    h, w = x.size
    x = np.asarray(x).copy()
    angle = np.random.randint(-20,20)
    center = (w/2,h/2)
    M = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(x,M,(w,h))

def random_affine(x):
    h,w = x.size
    x = np.asarray(x).copy()
    p1 =  np.float32([[0,0],[w-1,0],[0,h-1],[h-1,w-1]])
    p2 =  np.float32([[0,w*0.3],[h*0.8,w*0.2],[h*0.15,w*0.7],[h*0.8,w*0.8]])
    M = cv2.getPerspectiveTransform(p1,p2)
    dst = cv2.warpPerspective(x,M,(w,h))
    return dst


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'


        fname = self.lines[index]['fname']
        img = self.lines[index]['img']
        gt_count = self.lines[index]['gt_count']

        if self.train:
            mask_size = 256
            if True:
                image = img
                w, h = image.shape[:2]
                cxmin, cxmax = 0, h - mask_size
                cymin, cymax = 0, w - mask_size

                if cymin == cymax:
                    cy = 0
                else:
                    cy = np.random.randint(cymin, cymax)

                if cxmin == cxmax:
                    cx = 0
                else:
                    cx = np.random.randint(cxmin, cxmax)

                xmin =  cx
                ymin =  cy
                xmax = xmin + mask_size
                ymax = ymin + mask_size

                if len(image.shape) == 3:
                    img = image[ymin:ymax, xmin:xmax, :]
                else:
                    img = image[ymin:ymax, xmin:xmax]

                gt_count =  gt_count[xmin:xmax,ymin:ymax]

                gt_count =  gt_count.sum()

                img = Image.fromarray(img.astype('uint8'))

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if self.train == True:
            if random.random() > 0.5:
                img = random_rotate(img)
                img = Image.fromarray(img.astype('uint8'))
            if random.random() > 0.5:
                img = random_light(img)
                img = Image.fromarray(img.astype('uint8'))


        gt_count = gt_count.copy()

        img = img.copy()

        if self.train == True:
            if self.transform is not None:
                img = self.transform(img)

            if False:
                with open("./num_distribution.txt", 'a+') as abc:  # 写入numpy.ndarray数据
                    np.savetxt(abc, gt_count.reshape(1),fmt="%.1f", delimiter=",")
            return fname, img, gt_count

        else:
            if self.transform is not None:
                img = self.transform(img)

            width, height = img.shape[2], img.shape[1]

            m = int(width / 256)
            n = int(height /  256)
            for i in range(0, m):
                for j in range(0, n):

                    if i == 0 and j == 0:
                        img_return = img[:, j *  256:  256 * (j + 1), i *  256:(i + 1) *  256].cuda().unsqueeze(0)
                    else:
                        crop_img = img[:, j *  256:  256 * (j + 1), i *  256:(i + 1) *  256].cuda().unsqueeze(0)

                        img_return = torch.cat([img_return, crop_img], 0).cuda()
            return fname, img_return, gt_count
