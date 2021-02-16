import numpy as np
import random
import os

import warnings
warnings.filterwarnings('ignore')

#import general utils functions
import utils.general_utils as gu
#import project related utils
import utils.project_utils as pu

train_config_path = 'train_config.ini'

import logging
logger = logging.getLogger('fnv_train')
LOGI = logger.info
LOGD = logger.debug
LOGE = logger.error
import logging.config
log_conf = pu.get_logconf_path(train_config_path)
logging.config.fileConfig(log_conf)

class SplitData():
    def __init__(self, train_config_path):

        self.train_config_path = train_config_path
        train_config = pu.get_configParser(train_config_path)

        # Split train, val, test ratios
        self.train_ratio = 0.9
        self.val_ratio = 0.06
        self.test_ratio = 1.0 - (self.train_ratio + self.val_ratio)

        project_folder = train_config['local_variables']['project_folder']
        self.dataset_dir = os.path.join(project_folder, train_config['project_folders']['data'])
        self.meta_dir = os.path.join(project_folder, train_config['project_folders']['meta_data'])

        # Split data json file
        self.split_data_filepath = os.path.join(self.meta_dir, train_config['files']['split_data_file'])

    # Split data specific to sku
    def sku_data_split(self, sku_id):
        LOGI('\n')
        LOGI('Splitting data for sku id %s' % sku_id)

        annotations = pu.get_annot_sku(sku_id, self.meta_dir)

        annotations = list(annotations['_via_img_metadata'].values())
        annot_imgs = [a['filename'] for a in annotations if a['regions']]

        tot_imgs = len(annot_imgs)

        # Split data based on train, val, test ratios
        train_len = int(self.train_ratio * tot_imgs)
        val_len = int(np.round(self.val_ratio * tot_imgs))
        test_len = tot_imgs - (train_len + val_len)
        LOGI("train_len = %d val_len = %d test_len = %d" % (train_len, val_len, test_len))

        # shuffle images list and random split in unique way for all the runs
        annot_imgs.sort()
        random.seed(6)
        random.shuffle(annot_imgs)

        sku_data = {}
        sku_data['train'] = annot_imgs[:train_len]
        sku_data['val'] = annot_imgs[train_len:(train_len + val_len)]
        sku_data['test'] = annot_imgs[(train_len + val_len):]
        LOGI("Splitted images count train_len = %d val_len = %d test_len = %d" % (
        len(sku_data['train']), len(sku_data['val']), len(sku_data['test'])))

        return sku_data

    #Split data and save into json file
    def data_split(self):
        # Get individal unique sku ids
        sku_ids = pu.get_sku_ids(self.train_config_path)
        LOGI('Unique sku_ids')
        LOGI(sku_ids)
        data = {}

        for sku_id in sku_ids:
            data[sku_id] = self.sku_data_split(sku_id)

        # Save splitted data in json file
        gu.save_as_json(self.split_data_filepath, data)

if __name__ == "__main__":

    #Split data into train, val and test for all the skus
    split_data = SplitData(train_config_path)
    split_data.data_split()