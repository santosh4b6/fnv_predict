import logging
import os
import json
import configparser

logger = logging.getLogger('fnv_utils')
LOGI = logger.info
LOGD = logger.debug
LOGE = logger.error


#Loads configuration file and parse it
def get_configParser(config_filepath):
    config = configparser.ConfigParser()
    config.read(config_filepath)
    return config

#get train log configuration path
def get_logconf_path(config_filepath):
    config = get_configParser(config_filepath)
    log_conf = os.path.join(config['local_variables']['project_folder'], config['project_folders']['train'],
                            config['files']['log_file'])
    return log_conf

#Get unique sku ids of this project for training inference etc.,
def get_sku_ids(config_filepath):
    train_config = get_configParser(config_filepath)
    project_folder = train_config['local_variables']['project_folder']
    dataset_dir = os.path.join(project_folder, train_config['project_folders']['data'])
    sku_ids = os.listdir(dataset_dir)
    return sku_ids

#Load annotations data of particular sku
def get_annot_sku(sku_id, meta_dir):
    annotations = json.load(open(os.path.join(meta_dir, "fnv_" + sku_id + ".json")))
    return annotations

#Get pretrained weight model file
def get_pretrainedmodel_path(config_filepath):
    config = get_configParser(config_filepath)
    model_path = os.path.join(config['local_variables']['project_folder'], config['project_folders']['models'],
                            config['model_files']['pre_trained'])
    return model_path


#Get fnv weight model file
def get_fnvmodel_path(config_filepath):
    config = get_configParser(config_filepath)
    model_path = os.path.join(config['local_variables']['project_folder'], config['project_folders']['models'],
                            config['model_files']['fnv_model'])
    return model_path

#Get model file paths
def get_model_paths(config_filepath):
    pretrained_model_path = get_pretrainedmodel_path(config_filepath)
    fnv_model_path = get_fnvmodel_path(config_filepath)
    return pretrained_model_path,fnv_model_path