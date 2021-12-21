import pandas as pd
import numpy as np
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import shutil


def get_data(path):
    with open(path) as d:
        djson = json.load(d)

    for d in djson['images']:
        d['id'] = str(d['id'])
    df_images = pd.DataFrame().from_dict(djson['images'])

    for d in djson['annotations']:
        d['image_id'] = str(d['image_id'])
    df_annotations = pd.DataFrame().from_dict(djson['annotations'])

    image_id_list = list(df_annotations['_image_id'].unique())
    df_images = df_images.loc[df_images['_image_id'].isin(image_id_list)]
    df_images.index = range(len(df_images))

    data = dict()
    df = pd.DataFrame().from_dict(djson['categories'])
    data['categories'] = df['name'].unique().tolist()
    old_ids = df['id'].unique()
    new_ids = range(len(old_ids))
    id_dict = dict()
    for e, old in enumerate(old_ids):
        id_dict[old] = new_ids[e]

    for i in range(len(df_images)):
        annot = df_annotations.loc[df_annotations['_image_id'] == df_images['_image_id'][i]]
        data_dict = dict()
        data_dict['bbox'] = annot['bbox'].tolist()
        data_dict['label'] = annot['category_id'].tolist()
        data_dict['label'] = [id_dict[cat_id] for cat_id in data_dict['label']]
        data[df_images['file_name'][i]] = data_dict
    data['images'] = df_images['file_name'].tolist()

    return data


def convert_bboxes_to_yolo(data_dict, path):
    for img_name in data_dict['images']:
        img_width, img_height = Image.open(f'data/images/{img_name}').size
        annotations_buffer = []
        for i in range(len(data_dict[img_name]['bbox'])):
            width = data_dict[img_name]['bbox'][i][2]
            height = data_dict[img_name]['bbox'][i][3]
            x_center = data_dict[img_name]['bbox'][i][0] + (data_dict[img_name]['bbox'][i][2] / 2)
            y_center = data_dict[img_name]['bbox'][i][1] + (data_dict[img_name]['bbox'][i][3] / 2)

            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height
            print(f'{x_center} {y_center} {width} {height}')

            annotations_buffer.append(f'{data_dict[img_name]["label"][i] - 1} {x_center} {y_center} {width} {height}')
        file_name = f'labels/{path}/{img_name.split(".")[0]}.txt'
        with open(file_name, 'w') as f:
            f.write('\n'.join(annotations_buffer))
        shutil.copy(f'data/images/{img_name}', f'images/{path}/{img_name}')


def plot_bounding_box(img, annotation_list):
    annotations_arr = np.array(annotation_list)
    w, h = img.size

    print(f'shape:{annotations_arr.shape} ann:{annotations_arr}')

    plotted_image = ImageDraw.Draw(img)

    transformed_annotations = np.copy(annotations_arr)
    transformed_annotations[:, [1, 3]] = annotations_arr[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations_arr[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (transformed_annotations[:, 3] / 2)
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (transformed_annotations[:, 4] / 2)
    transformed_annotations[:, 3] = transformed_annotations[:, 1] + transformed_annotations[:, 3]
    transformed_annotations[:, 4] = transformed_annotations[:, 2] + transformed_annotations[:, 4]

    for ann in transformed_annotations:
        print(f'trans:{ann}')
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)))

    plt.imshow(np.array(img))
    plt.show()


def create_yaml(categories):
    pathes = '\ntrain: images/train\nval: images/val\ntest: images/test'
    nc = f'nc: {len(categories)}'
    with open('data/config.yaml', 'w') as f:
        f.write(f'{pathes}\n\n{nc}\n\nnames: {categories}')


if __name__ == '__main__':
    data_train = get_data('data/annotations/instances_train.json')
    data_test = get_data('data/annotations/instances_test.json')
    data_val = get_data('data/annotations/instances_validation.json')

    create_yaml(data_train['categories'])

    convert_bboxes_to_yolo(data_train, path='train')
    convert_bboxes_to_yolo(data_test, path='test')
    convert_bboxes_to_yolo(data_val, path='val')

