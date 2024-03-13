import os
import torch
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
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


NUM_CLASSES = 70
NUM_INSTANCES = 5 

if __name__ == "__main__":
    cfg.merge_from_file("config.yml")
    cfg.MODEL.DEVICE_ID = "0"
    cfg.TEST.WEIGHT = "./procthor_final.pth"
    cfg.freeze()
    print(cfg)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, num_class=NUM_CLASSES, camera_num=1, view_num=1, gpu0=0, gpu1=0, target_gpu=0)
    model.load_param(cfg.TEST.WEIGHT)

    val_transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    test_path = "./data/procthor_final/val"
    test_classes = []
    for classname in os.listdir(test_path):
        test_classes.append(osp.join(test_path, classname))


    test_images = [os.listdir(os.path.join(test_path, c)) for c in test_classes]
    for class_idx, classname in enumerate(test_classes):
        for img_idx, img in test_images:
            img_name = test_images[class_idx][img_idx]
            rgb_path = osp.join(test_path, classname, img)
            depth_path = osp.join(test_path, classname, img.split(".")[0] + ".npy")

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

            test_images[class_idx][img_idx] = (rgb, depth)


    model.eval()
    w = []
    with torch.no_grad():
        with tqdm(total=len(test_images) * len(test_images[0])) as bar:
            for row in test_images:
                r = []
                for i in row:
                    # im = test_transforms(i.convert("RGB"))
                    im = val_transforms(i)
                    with torch.no_grad():
                        k = model(im.unsqueeze(0))
                    r.append(k)
                    bar.update(1)
                w.append(torch.stack(r))
    w = torch.stack(w)
    w = w.reshape((-1, w.shape[-1]))

    scores = torch.zeros((w.shape[0], w.shape[0])).cpu().numpy()
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            scores[i][j] = w[i] @ w[j] / (torch.norm(w[i]) * torch.norm(w[j]))

    print(f"w.shape = {w.shape}")
    plt.figure(figsize=(15, 15))
    plt.imshow(scores, cmap="hot")
    plt.colorbar()

    num_instances = NUM_INSTANCES

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
    plt.title("Trans-ReID")
    plt.savefig("heatmap.jpg")
