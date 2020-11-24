# -*- coding: utf-8 -*-

"""
Created on Tue Mar 27 13:28:27 2018
@author: PavitrakumarPC
"""

import numpy as np
import cv2
import os
import pandas as pd
import h5py

train_folder = '/home/apple/lee/cvHw2/train'
test_folder = '/home/apple/lee/cvHw2/test'
extra_folder = '/extra'
resize_size = (64, 64)


def collapse_col(row):
    global resize_size
    new_row = {}
    new_row['img_name'] = list(row['img_name'])[0]
    new_row['labels'] = row['label'].astype(np.str).str.cat(sep='_')
    new_row['top'] = max(int(row['top'].min()), 0)
    new_row['left'] = max(int(row['left'].min()), 0)
    new_row['bottom'] = int(row['bottom'].max())
    new_row['right'] = int(row['right'].max())
    new_row['width'] = int(new_row['right'] - new_row['left'])
    new_row['height'] = int(new_row['bottom'] - new_row['top'])
    new_row['num_digits'] = len(row['label'].values)
    return pd.Series(new_row, index=None)


def image_data_constuctor(img_folder, img_bbox_data):
    print 'image data construction starting...'
    imgs = []
    for img_file in os.listdir(img_folder):
        if img_file.endswith('.png'):
            imgs.append([img_file, cv2.imread(os.path.join(img_folder,
                        img_file))])
    img_data = pd.DataFrame([], columns=['img_name', 'img_height',
                            'img_width', 'img', 'cut_img'])
    print 'finished loading images...starting image processing...'
    for img_info in imgs:
        row = img_bbox_data[img_bbox_data['img_name'] == img_info[0]]
        full_img = img_info[1]
        cut_img = full_img.copy()[int(row['top']):int(row['top']
                                  + row['height']), int(row['left']):
                                  int(row['left'] + row['width']), ...]
        row_dict = {
            'img_name': [img_info[0]],
            'img_height': [img_info[1].shape[0]],
            'img_width': [img_info[1].shape[1]],
            'img': [full_img],
            'cut_img': [cut_img],
            }
        img_data = pd.concat([img_data,
                             pd.DataFrame.from_dict(
                                 row_dict, orient='columns')])
    print 'finished image processing...'
    return img_data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = ([hdf5_data[attr.value[i].item()].value[0][0] for i in
                  range(len(attr))] if len(attr)
                  > 1 else [attr.value[0][0]])
        attrs[key] = values
    return attrs


def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file, 'r')
    all_rows = []
    print 'image bounding box data construction starting...'
    bbox_df = pd.DataFrame([], columns=[
        'height',
        'img_name',
        'label',
        'left',
        'top',
        'width',
        ])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df, pd.DataFrame.from_dict(row_dict,
                            orient='columns')])
        for (index, row) in bbox_df.iterrows():
            img_name = row['img_name']
            image = \
                cv2.imread('/home/apple/lee/cvHw2/darknet/data/obj/' +
                           img_name)
            (height, width, _) = image.shape
            print(
                int(row['label']) % 10,
                width,
                height,
                row['img_name'],
                (row['left'] + row['width'] / 2) / width,
                (row['top'] + row['height'] / 2) / height,
                row['width'] / width,
                row['height'] / height,
                )
            exit(0)
    bbox_df['bottom'] = bbox_df['top'] + bbox_df['height']
    bbox_df['right'] = bbox_df['left'] + bbox_df['width']
    print 'finished image bounding box data construction...'
    return bbox_df


def construct_all_data(img_folder, mat_file_name, h5_name):
    img_bbox_data = \
        img_boundingbox_data_constructor(
            os.path.join(img_folder, mat_file_name))
    img_bbox_data_grouped = img_bbox_data.groupby(
        'img_name').apply(collapse_col)
    img_data = image_data_constuctor(img_folder, img_bbox_data_grouped)
    print 'done constructing main dataframes...starting grouping'
    df1 = img_bbox_data_grouped.merge(
        img_data, on='img_name', how='left')
    print 'grouping done'

    df1.to_hdf(os.path.join(img_folder, h5_name), 'table')


darknet_data_path = \
    '/home/apple/lee/cvHw2/darknet/build/darknet/x64/data/'


def construct_for_darknet(img_folder, mat_file_name):
    f = h5py.File(os.path.join(img_folder, mat_file_name), 'r')
    all_rows = []
    print 'YEE! image bounding box data construction starting...'
    bbox_df = pd.DataFrame([], columns=[
        'height',
        'img_name',
        'label',
        'left',
        'top',
        'width',
        ])
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df, pd.DataFrame.from_dict(row_dict,
                            orient='columns')])

    img_name = ''
    for (index, row) in bbox_df.iterrows():
        if row['img_name'] != img_name:  # here comes another image
            img_name = row['img_name']
            with open(darknet_data_path + 'train.txt', 'a') as ff:
                ff.write('data/obj/' + img_name + '\n')
        image = \
            cv2.imread('/home/apple/lee/cvHw2/data/obj/'
                       + img_name)
        (height, width, _) = image.shape
        with open(darknet_data_path + 'obj/' + img_name.split('.')[0]
                  + '.txt', 'a') as ff:
            ff.write(
                str(int(row['label']) % 10) + ' ' +
                str((row['left'] + row['width'] / 2) / width) + ' ' +
                str((row['top'] + row['height'] / 2) / height) + ' ' +
                str(row['width'] / width) + ' ' +
                str(row['height'] / height) + '\n')

    print 'finished image bounding box data construction...'
    return bbox_df


construct_for_darknet(train_folder, 'digitStruct.mat')
