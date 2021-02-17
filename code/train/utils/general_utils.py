import logging
import os
import json
import shutil

logger = logging.getLogger('fnv_utils')
LOGI = logger.info
LOGD = logger.debug
LOGE = logger.error

#save data to json file
def save_as_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

#load json file
def load_json(filepath):
    if (os.path.exists(filepath)):
        with open(filepath, 'r') as infile:
            data = json.load(infile)
    else:
        data = {}
        LOGE('%s file does not exists'%filepath)
    return data

def copy_file(trained_model_path, deploy_model_path):
    shutil.copyfile(trained_model_path, deploy_model_path)








