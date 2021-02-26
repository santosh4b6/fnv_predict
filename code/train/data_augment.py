import os

import warnings
warnings.filterwarnings('ignore')

#import general utils functions
from utils import general_utils as gu
#import project related utils
from utils import project_utils as pu
#import data augment related utils
from utils import data_augment_utils as dau

train_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train_config.ini')
print(train_config_path)

import logging
logger = logging.getLogger('fnv_train')
LOGI = logger.info
LOGD = logger.debug
LOGE = logger.error
import logging.config
log_conf = pu.get_logconf_path(train_config_path)
logging.config.fileConfig(log_conf)


class DataAugment():
    def __init__(self, train_config_path):
        self.train_config_path = train_config_path
        train_config = pu.get_configParser(train_config_path)

        project_folder = train_config['local_variables']['project_folder']
        self.dataset_dir = os.path.join(project_folder, train_config['project_folders']['data'])
        self.meta_dir = os.path.join(project_folder, train_config['project_folders']['meta_data'])

        #Data Augment base folders
        self.aug_data_dir = os.path.join(project_folder, train_config['project_folders']['aug_data'])
        self.aug_meta_data_dir = os.path.join(project_folder, train_config['project_folders']['aug_meta_data'])

        # Split data json file
        self.split_data_filepath = os.path.join(self.meta_dir, train_config['files']['split_data_file'])

    def augment_image(self, image_name, sku_id, mode):

        #SKU augmented images annotations json file
        aug_annot_json = os.path.join(self.aug_meta_data_dir, "fnv_" + sku_id + "_da_"+mode+".json")

        #Create augment annot dict if not present in augment meta data dir
        aug_sku_annot = dau.get_aug_sku_annots(self.meta_dir, sku_id, aug_annot_json)

        #get sku annotation data
        annotations = pu.get_annot_sku(sku_id, self.meta_dir)

        #Get image corresponding annotations data
        img_annot = pu.get_image_annotation(image_name, annotations)

        #Horizontal flip, Vertical flip and Flip Horziontal and vertical both
        aug_sku_annot = dau.get_image_aug_flip(image_name, self.dataset_dir, sku_id, self.aug_data_dir, img_annot, aug_sku_annot, mode)

        #Save augmented annotation data
        gu.save_as_json(aug_annot_json, aug_sku_annot)


