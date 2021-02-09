

import os
import configparser
config = configparser.ConfigParser()
train_config = config.read('train_config.ini')

base_folder = config['local_variables']['project_folder']
log_conf = os.path.join(base_folder, config['project_folders']['train'], config['files']['log_file'])

import logging
logger = logging.getLogger('fnv_train')
LOGI = logger.info
LOGD = logger.debug
LOGE = logger.error
import logging.config
logging.config.fileConfig(log_conf)


import ntpath
import glob
import warnings
warnings.filterwarnings('ignore')

#Cleanup local database files which are not present in dropbox folder
class CleanupDatabase:

    def __init__(self):
        self.dropbox_base_folder = config['local_variables']['dropbox_folder']
        self.project_folder = config['local_variables']['project_folder']
        self.dropbox_data = os.path.join(self.dropbox_base_folder, config['project_folders']['data'])
        self.project_data = os.path.join(self.project_folder, config['project_folders']['data'])

    def process(self):

        fnv_folders = os.listdir(self.dropbox_data)
        LOGI('Total fnv folders present in dropbox are %d'%(len(fnv_folders)))
        LOGI('fnv_folders are %s', str(fnv_folders))

        for folder in fnv_folders:
            project_fnv_images_list = glob.glob(os.path.join(self.project_data, folder, '*.jpg'))
            del_imgs_cnt = 0
            for image_path in project_fnv_images_list:
                image_name = ntpath.basename(image_path)
                dropbox_image_path = os.path.join(self.dropbox_data, folder, image_name)
                if(not os.path.exists(dropbox_image_path)):
                    os.remove(image_path)
                    del_imgs_cnt+=1
            LOGI("Total deleted images are %d for fnv %s" % (del_imgs_cnt, folder))


if __name__ == "__main__":

    clean_database = CleanupDatabase()
    clean_database.process()