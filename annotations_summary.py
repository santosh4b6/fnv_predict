import json
import os
import glob
import numpy as np



def get_sku_annotation_details(sku_tag):

    print('%s summarry'%sku_tag)
    curpath = os.getcwd()
    annot_path = os.path.join(curpath, 'annotation')
    data_path = os.path.join(curpath, 'database')

    json_in = "fnv_" + sku_tag + ".json"

    labels_pending_list = []

    if (os.path.exists(os.path.join(annot_path, json_in))):
        annotations_in = json.load(open(os.path.join(annot_path, json_in)))

        image_annotations = annotations_in['_via_img_metadata']
        tot_annotations = 0
        for image in image_annotations.keys():

            regions = image_annotations[image]['regions']
            for region in regions:
                if ('Fruits and Vegetables' not in region['region_attributes']):
                    labels_pending_list.append(image_annotations[image]['filename'])
            if (len(regions) > 0):
                tot_annotations += 1

        # print("Total annotated images %d" % tot_annotations)
    else:
        tot_annotations = 0

    data_path = os.path.join(data_path, sku_tag)

    tot_images = len(glob.glob(os.path.join(data_path, '*.jpg')))
    pend_annotations = tot_images - tot_annotations

    print("Total annotated images are %d/%d" % (tot_annotations, tot_images))
    print('Approximate work time %d days' % np.ceil(pend_annotations / 100))

    if (len(labels_pending_list) == 0):
        print('All annotated images are labelled')
    else:
        print('Labels pending for annotated images list is')
        print(labels_pending_list)

    print('\n')
	
if __name__ == "__main__":

    # sku_tag = 'tomato-hybrid'
    # sku_tag = input('Enter csku tag: ')
    sku_tag = 'tomato-hybrid'
    get_sku_annotation_details(sku_tag)

    sku_tag = 'ladies_finger'
    get_sku_annotation_details(sku_tag)

    sku_tag = 'tapioca'
    get_sku_annotation_details(sku_tag)
