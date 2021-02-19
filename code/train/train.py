import os
import ntpath

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

#import split data class
from split_data import SplitData

#Import from inference package
from fnv_inference import model as modellib
from fnv_inference.fnv import FNVDataset, FNVConfig


class Train():

    def __init__(self, train_config_path):
        self.train_config_path = train_config_path
        train_config = pu.get_configParser(train_config_path)

        project_folder = train_config['local_variables']['project_folder']
        self.dataset_dir = os.path.join(project_folder, train_config['project_folders']['data'])
        self.meta_dir = os.path.join(project_folder, train_config['project_folders']['meta_data'])
        self.model_dir = os.path.join(project_folder, train_config['project_folders']['models'])
        self.deployment_model_path = os.path.join(project_folder, train_config['project_folders']['deployment'], train_config['model_files']['fnv_model'])

        # Split data json file
        self.split_data_filepath = os.path.join(self.meta_dir, train_config['files']['split_data_file'])

    def load_data(self, mode):
        dataset = FNVDataset()

        #load split data
        split_data = gu.load_json(self.split_data_filepath)

        #Get sku product ids
        sku_product_ids = pu.get_sku_ids(self.train_config_path)

        # dataset_train.load_balloon(args.dataset, "train")
        dataset.load_csku(self.dataset_dir, self.meta_dir, sku_product_ids, split_data, mode)
        dataset.prepare()

        LOGI('\n')
        LOGI("%s Samples Details"%mode)
        LOGI("Image Count: {}".format(len(dataset.image_ids)))
        LOGI("Class Count: {}".format(dataset.num_classes))
        for i, info in enumerate(dataset.class_info):
            LOGI("{:3}. {:50}".format(i, info['name']))

        return dataset

    def data_preparation(self):
        # Split data into train, val and test for all the skus
        split_data = SplitData(train_config_path)
        split_data.data_split()

        #data augmentation TBD

        # Training dataset.
        dataset_train = self.load_data("train")

        # Validation dataset
        dataset_val = self.load_data("val")

        return dataset_train, dataset_val

    def copy_model_deployment_folder(self, trained_model_path, deploy_model_path):
        LOGI("Copying updated trained model in deployment folder")
        gu.copy_file(trained_model_path, deploy_model_path)

    def run(self):

        #Split annotated data and data preparation
        # dataset_train, dataset_val = self.data_preparation()

        # train configuration setup
        config = FNVConfig()
        config.NAME = "fnv"
        # config.display()

        #Get pretrained model and fnv model paths
        pretrained_model_path = pu.get_previous_model_path(self.train_config_path)

        if(pretrained_model_path == -1):
            LOGE("[ERROR] Unable to find pretrained model")
            return
        else:
            LOGI("%s is previous latest model file"%(ntpath.basename(pretrained_model_path)))

        # model = modellib.MaskRCNN(mode="training", config=config,
        #                           model_dir=self.model_dir)
        #
        # # Exclude the last layers because they require a matching
        # # number of classes
        # model.load_weights(pretrained_model_path, by_name=True, exclude=[
        #     "mrcnn_class_logits", "mrcnn_bbox_fc",
        #     "mrcnn_bbox", "mrcnn_mask"])
        #
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=3000,
        #             layers='all')

        #Copy trained modelfile to deployment file
        trained_model_path = pu.get_fnvmodel_path(self.train_config_path)
        self.copy_model_deployment_folder(trained_model_path, self.deployment_model_path)


if __name__ == "__main__":

    train = Train(train_config_path)
    train.run()