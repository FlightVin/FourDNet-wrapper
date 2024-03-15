import os
import torch
from config import cfg
import argparse
# from datasets import make_dataloader
from model import make_model
# from processor import do_inference
# from utils.logger import setup_logger
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np
import cv2
import os
import shutil
import os.path as osp
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from dataclasses import dataclass
import tyro

NUM_CLASSES = 69
NUM_VIEWS = 5 

@dataclass
class LocalArgs:
    """
    Class to hold local configuration arguments.
    """
    reid_config_file: str = "config.yml"
    reid_model: str = "/scratch/vineeth.bhat/FourDNet/procthor_final.pth"
    reid_num_classes: int = 69
    test_folder: str = "/scratch/vineeth.bhat/FourDNet/data/procthor_final/val"
    reid_model_pretrain_path: str = "/scratch/vineeth.bhat/FourDNet/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"

def load_reid_model(largs):
    cfg.merge_from_file(largs.reid_config_file)
    cfg.MODEL.DEVICE_ID = "0"
    cfg.TEST.WEIGHT = largs.reid_model
    cfg.MODEL.PRETRAIN_PATH = largs.reid_model_pretrain_path
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, num_class=largs.reid_num_classes, camera_num=1, view_num=1, gpu0=0, gpu1=0, target_gpu=0)
    model.load_param(cfg.TEST.WEIGHT)

    return model

def get_reid_emb(model, rgb_path, depth_path):
    val_transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    # reading RGB image and applying transforms
    rgb = Image.open(rgb_path)
    rgb = val_transforms(rgb)

    # reading depth image and applying transforms
    depth = np.load(depth_path)
    depth = cv2.resize(depth, (128, 256))
    depth = np.repeat(depth[None, :, :], 3, axis=0)
    depth = np.clip(depth, 0.0, 10.0) 
    depth = depth / (10.0)
    depth = depth - 0.5 
    depth = depth / 0.5 
    depth = torch.tensor(depth)

    with torch.no_grad():
        k = model(rgb.unsqueeze(0), depth.unsqueeze(0))

    return k

def get_reid_emb_new(model, rgb, depth):
    val_transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    # reading RGB image and applying transforms
    # rgb = Image.open(rgb_path)
    rgb = val_transforms(rgb)

    # reading depth image and applying transforms
    # depth = np.load(depth_path)
    depth = cv2.resize(depth, (128, 256))
    depth = np.repeat(depth[None, :, :], 3, axis=0)
    depth = np.clip(depth, 0.0, 10.0) 
    depth = depth / (10.0)
    depth = depth - 0.5 
    depth = depth / 0.5 
    depth = torch.tensor(depth)

    with torch.no_grad():
        k = model(rgb.unsqueeze(0), depth.unsqueeze(0))

    return k


if __name__ == "__main__":
    largs = tyro.cli(LocalArgs, description=__doc__)
    print(largs)

    model = load_reid_model(largs)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_path = largs.test_folder
    test_classes = []
    for classname in os.listdir(test_path):
        test_classes.append(classname)


    test_images = [os.listdir(os.path.join(test_path, c)) for c in test_classes]
    
    model_inputs = []
    for class_idx, classname in enumerate(test_classes):
        model_inputs.append([])
        for img_idx, img in enumerate(test_images[class_idx]):
            if img.find(".npy") != -1:
                # if it is a depth image, we do not take it here, it would be taken along with it's rgb counterpart
                continue

            img_name = test_images[class_idx][img_idx]
            rgb_path = osp.join(test_path, classname, img)
            depth_path = osp.join(test_path, classname, img.split(".")[0] + ".npy")

            model_inputs[-1].append((rgb_path, depth_path))


    assert len(model_inputs) == NUM_CLASSES
    for i in range(NUM_CLASSES):
        assert len(model_inputs[i]) == NUM_VIEWS

    model.eval()
    w = []
    with torch.no_grad():
        with tqdm(NUM_CLASSES * NUM_VIEWS) as bar:
            for ctg in model_inputs:
                r = []
                for rgb, depth in ctg:
                    k = get_reid_emb(model, rgb, depth)
                    r.append(k)
                    bar.update(1)
                w.append(torch.stack(r))
    w = torch.stack(w)

    print(f"We have {NUM_VIEWS} views for {NUM_CLASSES} classes.")

    print(f"w.shape before reshaping = {w.shape}")
    w = w.reshape((-1, w.shape[-1]))

    scores = torch.zeros((w.shape[0], w.shape[0])).cpu().numpy()
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            scores[i][j] = w[i] @ w[j] / (torch.norm(w[i]) * torch.norm(w[j]))

    print(f"w.shape = {w.shape}")
    print(f"scores.shape = {scores.shape}")

    plt.figure(figsize=(15, 15))
    plt.imshow(scores, cmap="hot")
    plt.colorbar()

    num_instances = NUM_VIEWS

    x_axis_titles = [
        f"{test_classes[i//num_instances]}"
        for i in range(
            num_instances // 2, num_instances // 2 + len(scores), num_instances
        )
    ]
    y_axis_titles = [
        f"{test_classes[i//num_instances]}"
        for i in range(
            num_instances // 2, num_instances // 2 + len(scores), num_instances
        )
    ]

    plt.xticks(
        range(num_instances // 2, num_instances // 2 + len(scores), num_instances),
        x_axis_titles,
        fontsize=6,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(num_instances // 2, num_instances // 2 + len(scores), num_instances),
        y_axis_titles,
        fontsize=6,
        va="center",
    )

    for i in range(1, len(scores)):
        if i % num_instances == 0:
            plt.axvline(x=i - 0.5, color="blue", linestyle="-", linewidth=0.5)
            plt.axhline(y=i - 0.5, color="blue", linestyle="-", linewidth=0.5)

    # Show the heatmap
    plt.title("FourDNet")
    plt.savefig("heatmap.jpg")

    print("Saved plot")
