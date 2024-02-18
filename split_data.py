import os 
import os.path as osp 
import shutil 
from tqdm import tqdm

ROOT_DIR = "datav1"
TRAIN = 32
VAL = 8 
TEST = 8
NUM_INSTANCES = TRAIN + VAL + TEST 


NEW_DIR = f"{ROOT_DIR}_{TRAIN}_{VAL}_{TEST}"
os.mkdir(NEW_DIR)
os.mkdir(osp.join(NEW_DIR, "train"))
os.mkdir(osp.join(NEW_DIR, "val"))
os.mkdir(osp.join(NEW_DIR, "test"))
for class in tqdm(os.listdir(f"{ROOT_DIR}")):
    for i in range(TRAIN):
        shutil.copy(osp.join(ROOT_DIR, class, f"{i}.jpg"), osp.join(NEW_DIR, "train", f"{i}.jpg"))
        shutil.copy(osp.join(ROOT_DIR, class, f"{i}_d.jpg"), osp.join(NEW_DIR, "train", f"{i}_d.jpg"))
        
    for i in range(VAL):
        shutil.copy(osp.join(ROOT_DIR, class, f"{TRAIN + i}.jpg"), osp.join(NEW_DIR, "val", f"{i}.jpg"))
        shutil.copy(osp.join(ROOT_DIR, class, f"{TRAIN + i}_d.jpg"), osp.join(NEW_DIR, "val", f"{i}_d.jpg"))
    for i in range(TEST):
        shutil.copy(osp.join(ROOT_DIR, class, f"{TRAIN + VAL + i}.jpg"), osp.join(NEW_DIR, "test", f"{i}.jpg"))
        shutil.copy(osp.join(ROOT_DIR, class, f"{TRAIN + VAL + i}_d.jpg"), osp.join(NEW_DIR, "test", f"{i}_d.jpg"))
