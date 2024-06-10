import os
from tqdm import tqdm
from vcoco import vsrl_utils as vu
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib
import argparse

def draw_bbox(plt, ax, rois, fill=False, linewidth=2, edgecolor=[1.0, 0.0, 0.0], **kwargs):
    for i in range(rois.shape[0]):
        roi = rois.astype(int)
        ax.add_patch(plt.Rectangle((roi[0], roi[1]),
            roi[2] - roi[0], roi[3] - roi[1],
            fill=False, linewidth=linewidth, edgecolor=edgecolor, **kwargs))

def crop_bbox(plt, original_image, bbox, dir, filename):
    bbox = bbox.astype(int)
    img_shape = original_image.shape
    crop = original_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :] if len(img_shape) == 3 else original_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if crop.size == 0:
        print(f'{filename[:-4]} doesn\'t exsit')
        return
    plt.imshow(crop)
    plt.axis('off')
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + filename, bbox_inches='tight', pad_inches=0)
    #plt.show()
    
def subplot(plt, Y, X, sz_y = 10, sz_x = 10):
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X)
    return fig, axes

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def crop_image(dataset_type):
    print(f'\n ===== Processing {dataset_type}ing data ===== ')

    # Load COCO annotations for V-COCO images
    coco = vu.load_coco()

    # Load the VCOCO annotations
    vcoco_all = vu.load_vcoco(f'vcoco_{dataset_type}')
    for x in vcoco_all:
        x = vu.attach_gt_boxes(x, coco)

    # Filter the actions that have only one person and one object
    actions_id = []
    for i, action in enumerate(vcoco_all):
        if len(action['role_name']) == 2:
            actions_id.append([i, action['action_name']])
    print("\nActions with only one person and one object:")
    print(*actions_id, sep='\n')

    error_img = []

    # Calculate the number of positive samples for each action
    print("\nNumber of positive samples for each action")
    for i, action in enumerate(actions_id):
        action_id = action[0]
        action_name = action[1]
        vcoco = vcoco_all[action_id]
        positive_index = np.where(vcoco['label'] == 1)[0]
        print(f'| {action_name}: {len(positive_index)}')

    for i, action in enumerate(actions_id):
        action_id = action[0]
        action_name = action[1]

        # Load specific action
        # need_action = ['hold']
        # if action_name not in need_action:
        #     continue
        
        print(f'\n▌  Downloading action: {action_id} {action_name}')
        vcoco = vcoco_all[action_id]
        positive_index = np.where(vcoco['label'] == 1)[0]

        for j, id in enumerate(tqdm(positive_index)):

            if os.path.exists(f'data/{dataset_type}/{action_name}/{vcoco["image_id"][id][0]}/'):
                continue

            role_bbox = vcoco['role_bbox'][id,:]*1.
            role_bbox = role_bbox.reshape((-1,4))
            if np.isnan(role_bbox[1,0]): # Skip if there is no object
                continue
            # Load the image from the URL
            coco_image = coco.loadImgs(ids=[vcoco['image_id'][id][0]])[0]
            coco_url = coco_image['coco_url'] # flickr_url
            flickr_url = coco_image['flickr_url']

            #print(f'Downloading {coco_image["file_name"]}...')
            try:
                im = np.array(Image.open(urllib.request.urlopen(flickr_url, timeout=10)))
            except:
                try:
                    im = np.array(Image.open(urllib.request.urlopen(coco_url, timeout=20)))
                except:
                    print(f'{bcolors.WARNING}Warning: {action_name}/{coco_image["file_name"]}{bcolors.ENDC}')
                    error_img.append(f'{action_name}/{coco_image["file_name"]}')
                    continue
            
            sy = 4.0; sx = float(im.shape[1]) / float(im.shape[0]) * sy
            fig, ax = subplot(plt, 1, 1, sy, sx); ax.set_axis_off()
            ax.imshow(im)

            # Processing the bounding boxes
            person_box = vcoco['bbox'][[id],:][0]
            object_box = role_bbox[[1],:][0]
            union_box = [
                min(person_box[0], object_box[0]),
                min(person_box[1], object_box[1]),
                max(person_box[2], object_box[2]),
                max(person_box[3], object_box[3])
            ]
            intersection_box = [
                max(person_box[0], object_box[0]),
                max(person_box[1], object_box[1]),
                min(person_box[2], object_box[2]),
                min(person_box[3], object_box[3])
            ]
            # draw_bbox(plt, ax, person_box, edgecolor='r')
            # draw_bbox(plt, ax, object_box, edgecolor='lime')
            # draw_bbox(plt, ax, np.array(union_box), edgecolor='b')
            # draw_bbox(plt, ax, np.array(intersection_box), edgecolor='y')

            # Save the image and the bounding boxes
            dir = f'data/{dataset_type}/{action_name}/{vcoco["image_id"][id][0]}/'
            crop_bbox(plt, im, person_box, dir, 'agent.jpg')
            crop_bbox(plt, im, object_box, dir, 'object.jpg')
            crop_bbox(plt, im, np.array(union_box), dir, 'union.jpg')
            crop_bbox(plt, im, np.array(intersection_box), dir, 'intersection.jpg')

            # draw_bbox(plt, ax, vcoco['bbox'][[id],:], edgecolor='r')
            # for j in range(1, len(vcoco['role_name'])):
            #     if not np.isnan(role_bbox[j,0]):
            #         crop_bbox(plt, im, role_bbox[[j],:][0], dir, 'object.jpg')
            #         draw_bbox(plt, ax, role_bbox[[j],:], edgecolor='lime')
            #plt.show()
            plt.close()
    print('\nImages that cannot be downloaded:', *error_img, sep='\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type',
                      action='store',
                      dest='dataset_type', type=str,
                      metavar="type",
                      choices=['test', 'train', 'trainval', 'val'],
                      nargs=1,
                      help='The type of the dataset. Allowed values are: test, train, trainval, val.',
                      required=True)
    args = parser.parse_args()

    crop_image(args.dataset_type[0])