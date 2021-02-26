import os
import progressbar
import warnings
warnings.filterwarnings('ignore')

#import general utils functions
import utils.general_utils as gu
#import project related utils
import utils.project_utils as pu
#import data augment
import data_augment as da

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


class GenerateAugmentedData():
    def __init__(self, train_config_path):
        self.train_config_path = train_config_path
        train_config = pu.get_configParser(train_config_path)

        project_folder = train_config['local_variables']['project_folder']
        self.dataset_dir = os.path.join(project_folder, train_config['project_folders']['data'])
        self.meta_dir = os.path.join(project_folder, train_config['project_folders']['meta_data'])

        # Data Augment base folders
        self.aug_data_dir = os.path.join(project_folder, train_config['project_folders']['aug_data'])
        self.aug_meta_data_dir = os.path.join(project_folder, train_config['project_folders']['aug_meta_data'])

        # Split data json file
        self.split_data_filepath = os.path.join(self.meta_dir, train_config['files']['split_data_file'])

        #Create data augment object
        self.data_augment = da.DataAugment(train_config_path)


    #Generating augmented data corresponding to sku and image
    def generate_sku_mode_augment_data(self, sku_id, mode, split_data):
        sku_annots = pu.get_annot_sku(sku_id, self.meta_dir)
        sku_mode_images = []

        #Get the images which have regions
        sku_annots = sku_annots['_via_img_metadata']
        sku_annots = list(sku_annots.values())
        sku_annots = [a for a in sku_annots if a['regions']]
        for a in sku_annots:
            img_name = a['filename']
            if (img_name in split_data[sku_id][mode]):
                sku_mode_images.append(img_name)

        tot_imgs = len(sku_mode_images)
        LOGI("Total images present for sku %s in mode %s is %d"%(sku_id, mode, tot_imgs))

        LOGI("Generating augmented data for %s sku and %s mode"%(sku_id, mode))
        #Augmenting each image
        bar = progressbar.ProgressBar(max_value=tot_imgs)
        img_no = 1
        for image in sku_mode_images:
            bar.update(img_no)
            self.data_augment.augment_image(image, sku_id, mode)
            img_no+=1


    def run(self):

        #Get sku ids
        sku_ids = pu.get_sku_ids(self.train_config_path)

        #modes
        modes = ['train', 'val', 'test']

        #Load random split data of sku
        split_data = gu.load_json(self.split_data_filepath)

        #Augment data for each mode and sku
        for sku_id in sku_ids:
            for mode in modes:
                self.generate_sku_mode_augment_data(sku_id, mode, split_data)