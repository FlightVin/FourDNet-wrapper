from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import os 
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        # try:
        img = Image.open(img_path).convert('RGB')
        got_img = True
        # except IOError:
        #     print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
        #     pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid,img_path.split('/')[-1]




class RGBD_Dataset(Dataset):
    def __init__(self, dataset, transform=None, depth_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.depth_transform = depth_transform
        # self.depth_mean = 2.74432
        # self.depth_std = 1.37260
        # self.max_depth = 8.0
        # self.depth_mean = 3.45718
        # self.depth_std = 1.96257
        self.max_depth = 9.0
        self.min_depth = 1.0 
        self.depth_mean = np.array([0.485, 0.456, 0.406])
        self.depth_std = np.array([0.229, 0.224, 0.225]) 


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img_name = img_path.split("/")[-1]
        depth_path = osp.join(*(img_path.split("/")[:-1]), f"{img_name.split('.')[0]}.npy")
        # print(f"depth_path = {depth_path}") 
        assert os.path.exists(depth_path) 
        # print(f"the depth path exists!")
        # depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        # depth = cv2.resize(depth, (128, 256))
        # depth = np.clip(depth, 0.0, self.max_depth)
        # depth = depth / self.max_depth
        depth = np.load(depth_path)
        # print(f"loaded the depth successfully!")
        # depth = cv2.resize(depth, (128, 256))

        """for VGG"""
        depth = cv2.resize(depth, (224, 224))
        depth = np.repeat(depth[:, :, None], 3, axis=-1)
        # print(f"depth.shape = {depth.shape}")
        depth = np.clip(depth, self.min_depth, self.max_depth)
        depth = depth / (self.max_depth - self.min_depth)
        depth = depth - self.depth_mean[None, None, :]
        depth = depth / self.depth_std[None, None, :]
        # assert img_path.find(".npy") == -1
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.depth_transform is not None: 
            depth = self.depth_transform(depth)
        
        # print(f"img.shape = {img.shape}")
        # print(f"depth.shape = {depth.shape}")

        return img, depth, pid, camid, trackid,img_path.split('/')[-1]
