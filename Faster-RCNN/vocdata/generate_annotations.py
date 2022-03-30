import os
import argparse
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split




def main(data_dir=None, 
        out_dir=None,
        train_test_split_dir=None,
        csv_paths=None, 
        csv_boxes=None, 
        csv_labels=None,):
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
    for uid in df_paths.UniqueID:
        df_boxes_subset = df_boxes[df_boxes.UniqueID==uid]
        bbox_list = list()
        if len(df_boxes_subset)>0:
            for i in range(len(df_boxes_subset)):
                record = df_boxes_subset.iloc[i]
                y,x,h,w = record.Y, record.X, record.Height, record.Width
                x1,y1,x2,y2 = x, y, x+w, y+h
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                bbox = (x1,y1,x2,y2)
                bbox_list.append(bbox)
        bbox_col.append(bbox_list)
        n_bbox_col.append(len(bbox_list))
    df_paths['bboxes']= bbox_col
    df_paths['n_bboxes']= n_bbox_col


    os.makedirs(out_dir,exist_ok=True)
    for i in range(len(df_paths)):
        record = df_paths.iloc[i]
        fname = record.file_name[:-7]
        bboxes = record.bboxes
        annotation = {'boxes': bboxes}
        json_path = os.path.join(out_dir,fname+'.json')
        with open(json_path, 'w') as outfile:
            json.dump(annotation, outfile)


    # Train test split
    df_all = df_paths.copy()
    df_positive = df_all[df_all.n_bboxes>0]
    positive_patient_train, positive_patient_test = train_test_split(df_positive.PatientID.unique(),
                                                train_size=0.8,
                                                random_state=0)
    df_negative = df_all[df_all.n_bboxes==0]
    negative_patient_train, negative_patient_test = train_test_split(df_negative.PatientID.unique(),
                                                train_size=0.8,
                                                random_state=0)
    patient_train = np.concatenate((positive_patient_train,
                                negative_patient_train))
    patient_test = np.concatenate((positive_patient_test,
                                negative_patient_test))
    df_train = df_all[df_all.PatientID.isin(patient_train)]
    df_test = df_all[df_all.PatientID.isin(patient_test)]
    files_train = df_train.file_name.str[:-7]
    files_test = df_test.file_name.str[:-7]

    os.makedirs(train_test_split_dir,exist_ok=True)
    csv_path = os.path.join(train_test_split_dir,'train.csv')
    pd.DataFrame(files_train).to_csv(csv_path, index=False)
    csv_path = os.path.join(train_test_split_dir,'test.csv')
    pd.DataFrame(files_test).to_csv(csv_path,index=False)

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-data_dir',
                            metavar='data_dir',
                            type=str,
                            help='Directory that contains all the raw CSV files')
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