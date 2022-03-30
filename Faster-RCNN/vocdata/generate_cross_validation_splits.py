import os
import argparse
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, KFold



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
    print('All patients:')
    print('Positive:',len(df_positive.PatientID.unique()))
    print('Negative:',len(df_negative.PatientID.unique()))
    print('All:',len(df_all.PatientID.unique()))
    # Other splits on the train set
    kf = KFold(n_splits=4, random_state=0, shuffle=True)
    positive_patient_train_split=dict()
    positive_patient_test_split=dict()
    negative_patient_train_split=dict()
    negative_patient_test_split=dict()
    i=1
    for train_index, test_index in kf.split(positive_patient_train):
        positive_patient_train_split[i], positive_patient_test_split[i]\
            = np.concatenate((positive_patient_train[train_index],positive_patient_test)), \
                positive_patient_train[test_index]
        i += 1
    i=1
    for train_index, test_index in kf.split(negative_patient_train):
        negative_patient_train_split[i], negative_patient_test_split[i]\
            = np.concatenate((negative_patient_train[train_index],negative_patient_test)), \
                 negative_patient_train[test_index]
        i += 1
    
    positive_patient_train_split[0], positive_patient_test_split[0]\
            = positive_patient_train, positive_patient_test
    negative_patient_train_split[0], negative_patient_test_split[0]\
            = negative_patient_train, negative_patient_test

    # Load image slices names
    df_train_files = pd.read_csv(os.path.join(train_test_split_dir,'train-lesion-center.csv'))
    df_test_files = pd.read_csv(os.path.join(train_test_split_dir,'test-lesion-center.csv'))
    img_series = pd.concat((df_train_files,df_test_files),ignore_index=True).file_name
    img_series2 = img_series.str.split('_').apply(lambda x:x[0])
    df_train_files_single = pd.read_csv(os.path.join(train_test_split_dir,'train-single.csv'))
    df_test_files_single = pd.read_csv(os.path.join(train_test_split_dir,'test-single.csv'))
    img_series_single = pd.concat((df_train_files_single,df_test_files_single),ignore_index=True).file_name
    img_series2_single = img_series_single.str.split('_').apply(lambda x:x[0])

    print('Folds')
    print('=====')
    for i in range(5):
        patient_train = np.concatenate((positive_patient_train_split[i],
                                    negative_patient_train_split[i]))
        patient_test = np.concatenate((positive_patient_test_split[i],
                                    negative_patient_test_split[i]))
        print(f'{i}:')
        print(f'Train split:{len(patient_train)}')
        print(f'Test split:{len(patient_test)}')
        print(f'All:{len(np.unique(np.concatenate((patient_train,patient_test))))}')
        df_train = df_all[df_all.PatientID.isin(patient_train)]
        df_test = df_all[df_all.PatientID.isin(patient_test)]
        files_train = df_train.file_name.str[:-7]
        files_test = df_test.file_name.str[:-7]
        print(f'Train split volumes:{len(df_train)}')
        print(f'Test split volumes:{len(df_test)}')
        print(f'All volumes:{len(pd.concat((df_train,df_test)))}')
        # Save splits
        bool_series = img_series2.isin(files_train)
        df = pd.DataFrame(img_series[bool_series])
        df.to_csv(os.path.join(train_test_split_dir,f'train-lesion-center-split{i}.csv'),index=False)
        bool_series = img_series2.isin(files_test)
        df = pd.DataFrame(img_series[bool_series])
        df.to_csv(os.path.join(train_test_split_dir,f'test-lesion-center-split{i}.csv'),index=False)
        
        bool_series = img_series2_single.isin(files_train)
        df = pd.DataFrame(img_series_single[bool_series])
        df.to_csv(os.path.join(train_test_split_dir,f'train-single-split{i}.csv'),index=False)
        bool_series = img_series2_single.isin(files_test)
        df = pd.DataFrame(img_series_single[bool_series])
        df.to_csv(os.path.join(train_test_split_dir,f'test-single-split{i}.csv'),index=False)

    # sanity check
    df_train_split_0 = pd.read_csv(os.path.join(train_test_split_dir,f'train-lesion-center-split0.csv'))
    df_train = pd.read_csv(os.path.join(train_test_split_dir,f'train-lesion-center.csv'))
    assert df_train_split_0.equals(df_train), "No match with initial train-test split"
    df_train_split_0 = pd.read_csv(os.path.join(train_test_split_dir,f'train-single-split0.csv'))
    df_train = pd.read_csv(os.path.join(train_test_split_dir,f'train-single.csv'))
    assert df_train_split_0.equals(df_train), "No match with initial train-test split"

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