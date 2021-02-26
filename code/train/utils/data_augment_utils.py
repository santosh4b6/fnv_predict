import os
import json
import copy
import cv2
from pathlib import Path

#import general utils functions
from . import general_utils as gu
#import project related utils
from . import project_utils as pu

train_config_path = os.path.join(Path(os.path.dirname(os.path.realpath(__file__))).parent, 'train_config.ini')

import logging
logger = logging.getLogger('fnv_train')
LOGI = logger.info
LOGD = logger.debug
LOGE = logger.error
import logging.config
log_conf = pu.get_logconf_path(train_config_path)
logging.config.fileConfig(log_conf)

#Load sku data augmentation file
def get_aug_sku_annots(meta_dir, sku_id, aug_annot_json):

    # Create augmented annotation json file based on the original sku annotation json file
    if(not os.path.exists(aug_annot_json)):
        # sku annotation json file
        annot = json.load(open(os.path.join(meta_dir, "fnv_" + sku_id + ".json")))
        aug_annot = copy.deepcopy(annot)
        aug_annot['_via_img_metadata'] = {}
        gu.save_as_json(aug_annot_json, aug_annot)
    else:
        aug_annot = gu.load_json(aug_annot_json)

    return aug_annot

#Update augmented image annotation in sku annotation data
def update_aug_img_annotation(aug_sku_annot, aug_image_path, aug_image_name, img_annot):
    image_key = aug_image_name+str(os.stat(aug_image_path).st_size)
    aug_sku_annot['_via_img_metadata'][image_key] = img_annot
    return aug_sku_annot

#Generate augmented image based on flip mode and corresponding annotation
def get_image_aug_flip(image_name, dataset_dir, sku_id, aug_data_dir, img_annot, aug_sku_annot, mode):
    image_path = os.path.join(dataset_dir, sku_id, image_name)

    # Read original image
    img = cv2.imread(image_path)

    for tag in ['hf', 'vf', 'bf']:
        #Create augmented image path (folders if does not exist) and name
        aug_image_name = image_name[:-4]+'_'+tag+'.jpg'
        aug_image_dir =  os.path.join(aug_data_dir, sku_id)
        gu.create_folder(aug_image_dir)
        aug_image_dir = os.path.join(aug_image_dir, mode)
        gu.create_folder(aug_image_dir)
        aug_image_path = os.path.join(aug_image_dir, aug_image_name)

        #create a copy from original image
        img_annot_flip = copy.deepcopy(img_annot)
        img_annot_flip['filename'] = aug_image_name

        tot_regions = len(img_annot_flip['regions'])

        #Horizontal flip
        if(tag == 'hf'):
            img_flip = cv2.flip(img, 1)
            for ind in range(tot_regions):
                X = img_annot_flip['regions'][ind]['shape_attributes']['all_points_x']
                X = [img.shape[1] - val - 1 for val in X]
                img_annot_flip['regions'][ind]['shape_attributes']['all_points_x'] = X
        #Vertical flip
        elif(tag == 'vf'):
            img_flip = cv2.flip(img, 0)
            for ind in range(tot_regions):
                Y = img_annot_flip['regions'][ind]['shape_attributes']['all_points_y']
                Y = [img.shape[0] - val - 1 for val in Y]
                img_annot_flip['regions'][ind]['shape_attributes']['all_points_y'] = Y
        #Horizontal and Vertical flip
        elif(tag == 'bf'):
            img_flip = cv2.flip(img, -1)
            for ind in range(tot_regions):
                X = img_annot_flip['regions'][ind]['shape_attributes']['all_points_x']
                X = [img.shape[1] - val - 1 for val in X]
                img_annot_flip['regions'][ind]['shape_attributes']['all_points_x'] = X
                Y = img_annot_flip['regions'][ind]['shape_attributes']['all_points_y']
                Y = [img.shape[0] - val - 1 for val in Y]
                img_annot_flip['regions'][ind]['shape_attributes']['all_points_y'] = Y
        #Unsupported flip mode
        else:
            LOGE("[ERROR] Error in flip mode")

        #Save augmented image
        cv2.imwrite(aug_image_path, img_flip)
        aug_sku_annot = update_aug_img_annotation(aug_sku_annot, aug_image_path, aug_image_name, img_annot_flip)

    return aug_sku_annot


