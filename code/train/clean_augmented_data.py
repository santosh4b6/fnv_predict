import os

import warnings
warnings.filterwarnings('ignore')

#import general utils functions
import utils.general_utils as gu
#import project related utils
import utils.project_utils as pu

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


class CleanAugmentedData():
    def __init__(self, train_config_path):
        self.train_config_path = train_config_path
        train_config = pu.get_configParser(train_config_path)

        project_folder = train_config['local_variables']['project_folder']

        #Data Augment base folders
        self.aug_data_dir = os.path.join(project_folder, train_config['project_folders']['aug_data'])
        self.aug_meta_data_dir = os.path.join(project_folder, train_config['project_folders']['aug_meta_data'])


    def run(self):

        #Delete augmented images
        gu.clean_dir(self.aug_data_dir)

        #Delete augmented data annotation
        gu.clean_dir(self.aug_meta_data_dir)
