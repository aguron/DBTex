import os
import argparse
import numpy as np
import pandas as pd
import json
import nibabel as nib
from itertools import compress


def main(data_dir=None, 
        out_dir=None,
        nifti_dir=None,
        train_test_split_dir=None,
        csv_paths=None, 
        csv_boxes=None, 
        csv_labels=None,):


    r = 0.25 # defining the lesion depth
    csv_paths = os.path.join(data_dir, csv_paths)
    csv_boxes = os.path.join(data_dir, csv_boxes)
    csv_labels = os.path.join(data_dir, csv_labels)
    df_paths = pd.read_csv(csv_paths)
    df_boxes = pd.read_csv(csv_boxes)
    df_labels = pd.read_csv(csv_labels)
    # Finding file names and create uniqueid for iamges
    df_paths['file_name'] = df_paths.descriptive_path.str.split('/').apply(lambda x:x[-2]+'.nii.gz')
    df_paths['UniqueID'] = df_paths.PatientID+df_paths.StudyUID+df_paths.View
    df_boxes['UniqueID'] = df_boxes.PatientID+df_boxes.StudyUID+df_boxes.View
    df_labels['UniqueID'] = df_labels.PatientID+df_labels.StudyUID+df_labels.View
    # Find 2D bounding boxes for each image
    bbox_col = list()
    n_bbox_col = list()
    slice_col = list()
    slice_col2 = list()
    n_slices_col = list()
    ad_col = list() # for Architectural Distortion
    for fname in df_paths.file_name:
        img = nib.load(os.path.join(nifti_dir,fname))
        n_slices = img.dataobj.shape[0]
        n_slices_col.append(n_slices)
    df_paths['n_slices']=n_slices_col
    for idx in range(len(df_paths)):
        image_record = df_paths.iloc[idx]
        ns = image_record.n_slices
        uid = image_record.UniqueID
        df_boxes_subset = df_boxes[df_boxes.UniqueID==uid]
        bbox_list = list()
        slice_list = list()
        slice_list2 = list()
        ad_list=list()
        if len(df_boxes_subset)>0:
            for i in range(len(df_boxes_subset)):
                record = df_boxes_subset.iloc[i]
                y,x,h,w,s = record.Y, record.X, record.Height, record.Width, record.Slice
                x1,y1,x2,y2 = x, y, x+w, y+h
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                bbox = (x1,y1,x2,y2)
                bbox_list.append(bbox)
                slice_list.append((max(0,int(s-r*ns)),min(ns-1,int(s+r*ns))))
                slice_list2.append(s)
                ad_list.append(int(record.AD))
        bbox_col.append(bbox_list)
        n_bbox_col.append(len(bbox_list))
        slice_col.append(slice_list)
        slice_col2.append(slice_list2)
        ad_col.append(ad_list)
    df_paths['bboxes']= bbox_col
    df_paths['box_slices']= slice_col
    df_paths['n_bboxes']= n_bbox_col
    df_paths['box_centers']= slice_col2
    df_paths['AD']=ad_col
    def produce_slice_annotations(n_slices,bboxes,box_slices,labels):
        bboxes_per_slice = list()
        labels_per_slice = list()
        for s in range(n_slices):
            bboxes_bool = list(map(lambda x: s in range(x[0],x[1]),box_slices))
            bboxes_per_slice.append(list(compress(bboxes,bboxes_bool)))
            labels_per_slice.append(list(compress(labels,bboxes_bool)))
        return bboxes_per_slice, labels_per_slice

    os.makedirs(out_dir,exist_ok=True)
    for i in range(len(df_paths)):
        record = df_paths.iloc[i]
        fname = record.file_name[:-7]
        bboxes = record.bboxes
        labels = record.AD
        n_slices = record.n_slices
        box_slices = record.box_slices
        bboxes_per_slice,labels_per_slice = produce_slice_annotations(n_slices,
                        bboxes,box_slices,labels)
        for s in range(n_slices):
            annotation = {'boxes':bboxes_per_slice[s], 
                        'labels':labels_per_slice[s]}
            json_path = os.path.join(out_dir,'{}_{}.json'.format(fname,s))
            with open(json_path, 'w') as outfile:
                json.dump(annotation, outfile)

    img_files = os.listdir(out_dir)
    img_series = pd.Series(img_files, name='file_name').str[:-7]
    img_series2 = img_series.str.split('_').apply(lambda x:x[0])
    fname_series = pd.read_csv(os.path.join(train_test_split_dir,'train.csv')).file_name
    bool_series = img_series2.isin(fname_series)
    df = pd.DataFrame(img_series[bool_series])
    df.to_csv(os.path.join(train_test_split_dir ,'train-single.csv'),index=False)

    fname_series = pd.read_csv(os.path.join(train_test_split_dir,'test.csv')).file_name
    bool_series = img_series2.isin(fname_series)
    df = pd.DataFrame(img_series[bool_series])
    df.to_csv(os.path.join(train_test_split_dir,'test-single.csv'),index=False)

    slice_fnames = list()
    for i in range(len(df_paths)):
        record = df_paths.iloc[i]
        fname = record.file_name[:-7]
        bboxes = record.bboxes
        box_centers = record.box_centers
        for s in box_centers:
            slice_fnames.append('{}_{}'.format(fname,s))

    slice_fnames_normal = list()
    for i in range(len(df_paths)):
        record = df_paths.iloc[i]
        fname = record.file_name[:-7]
        bboxes = record.bboxes
        if len(bboxes)==0:
            # Then normal volume
            middle_slice = record.n_slices//2
            slice_fnames_normal.append('{}_{}'.format(fname,middle_slice))

    img_series = pd.Series(slice_fnames, name='file_name')
    img_series2 = img_series.str.split('_').apply(lambda x:x[0])
    fname_series = pd.read_csv(os.path.join(train_test_split_dir,'train.csv')).file_name
    bool_series = img_series2.isin(fname_series)
    df = pd.DataFrame(img_series[bool_series])
    df.to_csv(os.path.join(train_test_split_dir ,'train-lesion-center.csv'),index=False)
    img_series_normal = pd.Series(slice_fnames_normal, name='file_name')
    img_series2_normal = img_series_normal.str.split('_').apply(lambda x:x[0])
    fname_series = pd.read_csv(os.path.join(train_test_split_dir,'train.csv')).file_name
    bool_series = img_series2_normal.isin(fname_series)
    df_normal = pd.DataFrame(img_series_normal[bool_series])
    df = pd.concat([df,df_normal],ignore_index=True)
    df.to_csv(os.path.join(train_test_split_dir ,'train-lesion-and-normal-centers.csv'),index=False)


    fname_series = pd.read_csv(os.path.join(train_test_split_dir,'test.csv')).file_name
    bool_series = img_series2.isin(fname_series)
    df = pd.DataFrame(img_series[bool_series])
    df.to_csv(os.path.join(train_test_split_dir,'test-lesion-center.csv'),index=False)
    fname_series = pd.read_csv(os.path.join(train_test_split_dir,'test.csv')).file_name
    bool_series = img_series2_normal.isin(fname_series)
    df_normal = pd.DataFrame(img_series_normal[bool_series])
    df = pd.concat([df,df_normal],ignore_index=True)
    df.to_csv(os.path.join(train_test_split_dir ,'test-lesion-and-normal-centers.csv'),index=False)


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-data_dir',
                            metavar='data_dir',
                            type=str,
                            help='Directory that contains all the raw CSV files')
    arg_parser.add_argument('-nifti_dir',
                            metavar='nifti_dir',
                            type=str,
                            help='Directory that contains all the 3D nifti files')
    arg_parser.add_argument('-csv_paths',
                            metavar='csv_paths',
                            type=str,
                            default='BCS-DBT file-paths-train.csv',
                            help='The raw CSV giving the iamge paths')
    arg_parser.add_argument('-csv_boxes',
                            metavar='csv_boxes',
                            type=str,
                            default='BCS-DBT boxes-train.csv',
                            help='The raw CSV giving the boxes')
    arg_parser.add_argument('-csv_labels',
                            metavar='csv_labels',
                            type=str,
                            default='BCS-DBT labels-train.csv',
                            help='The raw CSV giving the labels')
    arg_parser.add_argument('-out_dir',
                            metavar='csv_labels',
                            type=str,
                            help='Directory to save JSON annotations')
    arg_parser.add_argument('-train_test_split_dir',
                            metavar='train_test_split_dir',
                            type=str,
                            help='Directory to save CSV splits')



    args=vars(arg_parser.parse_args())
    main(**args)