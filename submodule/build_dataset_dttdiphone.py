import json
import os
import re
import shutil

import cv2
import numpy as np
import numpy.ma as ma
import yaml
from tqdm import tqdm
from PIL import Image


################################### Utils #########################################
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

######################## dataset utils #############################
Borderlist = [-1] + list(range(40, 1960, 40))

def get_discrete_width_bbox(label, border_list, img_w, img_h):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt
    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt
    return rmin, rmax, cmin, cmax
    
def binary_search(sorted_list, target):
    l = 0
    r = len(sorted_list)-1
    while l!=r:
        mid = (l+r)>>1
        if sorted_list[mid] > target:
            r = mid
        elif sorted_list[mid] < target:
            l = mid + 1
        else:
            return mid
    return l
        

################################### DTTD #########################################
def getDTTDBBox(entry):
    label = np.array(Image.open(entry+"_label.png")) 
    with open(entry+"_meta.json", "r") as f:
        # dict_keys(['objects', 'object_poses', 'intrinsic', 'distortion'])
        objs = np.array(json.load(f)['objects']).flatten().astype(np.int32)
        if len(objs) == 0:
            print(entry)
    H, W = label.shape
    # get object's bounding box
    bboxs = []
    for obj_idx in objs:
        mask_label = ma.getmaskarray(ma.masked_equal(label, obj_idx))
        rmin, rmax, cmin, cmax = get_discrete_width_bbox(mask_label, Borderlist, W, H)
        bboxs.append((obj_idx,
                      (cmin+cmax)/(2*W),
                      (rmin+rmax)/(2*H), 
                      (cmax-cmin)/W,
                      (rmax-rmin)/H))
    return bboxs


def saveBBox(fpath, bbox, class_map):
    with open(fpath, 'w') as f:
        for bbox_line in bbox:
            f.write(f"{class_map[bbox_line[0]]} {bbox_line[1]} {bbox_line[2]} {bbox_line[3]} {bbox_line[4]}\n")
        

def transDTTDDataset(
        target_root,
        target_img_path,
        target_label_path,
        root="../estimation/dataset/dttd_iphone/DTTD_IPhone_Dataset/root/data", 
        config="../estimation/dataset/dttd_iphone/dataset_config", 
        real_only=True
    ):
    # load config lists
    train_list = test_list = class_list = None
    with open(os.path.join(config, "train_data_list.txt"), "r") as f:
        train_list = f.read().splitlines()
        if real_only:
            train_list = [ent for ent in train_list if not ent.startswith("data_syn")]
    with open(os.path.join(config, "test_data_list.txt"), "r") as f:
        test_list = f.read().splitlines()

    import pandas as pd
    class_idx = pd.read_csv(os.path.join(config, "objectids.csv"), index_col=None).to_dict()['id']
    class_map = {}
    for i in range(len(class_idx)):
        class_map[class_idx[i]] = i
    # load data and save
    count = 0
    
    # train
    target_train_list = []
    for entry in tqdm(train_list):
        bbox = getDTTDBBox(os.path.join(root, entry))
        shutil.copy(os.path.join(root, entry)+"_color.jpg", os.path.join(target_img_path, '%06d.png' % count))
        saveBBox(os.path.join(target_label_path, '%06d.txt' % count), bbox, class_map)
        target_train_list.append(os.path.join(target_img_path, '%06d.png' % count)+"\n")
        count += 1
        
    # test
    target_test_list = []
    for entry in tqdm(test_list):
        bbox = getDTTDBBox(os.path.join(root, entry))
        shutil.copy(os.path.join(root, entry)+"_color.jpg", os.path.join(target_img_path, '%06d.png' % count))
        saveBBox(os.path.join(target_label_path, '%06d.txt' % count), bbox, class_map)
        target_test_list.append(os.path.join(target_img_path, '%06d.png' % count)+"\n")
        count += 1
        
    # create yaml file
    with open(os.path.join(target_root, "train_data_list.txt"), 'w') as f:
        f.writelines(target_train_list)
    with open(os.path.join(target_root, "test_data_list.txt"), 'w') as f:
        f.writelines(target_test_list)

    dataset_config = {
        "path": target_root,
        "train": "train_data_list.txt",
        "val": "test_data_list.txt",
        "test": "test_data_list.txt",
        "names": pd.read_csv(os.path.join(config, "objectids.csv"), index_col=None).to_dict()['name']
    }
    with open(os.path.join(target_root, 'config.yml'), 'w') as outfile:
        yaml.dump(dataset_config, outfile, default_flow_style=False)
        
################################### Main #########################################

def BuildDataset(root="./dataset"):
    create_dir(root)
    img_path = create_dir(os.path.join(root, "images"))
    label_path = create_dir(os.path.join(root, "labels"))
    transDTTDDataset(root, img_path, label_path)
        
if __name__=='__main__':
    BuildDataset(root="/ssd/Dataset/Detection/DTTD_IPhone_Detection_Dataset")
    
