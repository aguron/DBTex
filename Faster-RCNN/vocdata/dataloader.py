import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.utils.data
import os
from os import listdir
from torchvision import transforms
from numpy import clip
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize
import pandas as pd
import nibabel as nib
import json
import random
from skimage.filters import unsharp_mask
from skimage.transform import resize
import albumentations as A
from MGimage import detect_calcifications_whole_image


def standard_image_preprocess(
            img, *args,
            new_size=None, 
            expand_dim=False, 
            adjust_label=False, 
            normalize=False,
            img_transforms=None,
            **kwargs):
    h, w = img.shape
    if new_size:
        img_new = np.zeros(new_size)
        img_new[0:h,0:w]=img
        img = img_new.copy()
        h, w = img.shape
        del(img_new)

    # Expand dimensions for image if specified
    if expand_dim is True:
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)

    if normalize:
        if img.max() > 1:
            # Remove white color
            img = img*(img<=2**16-2)
            img = img/(2**16-1)
    if img_transforms is not None:
        img = img_transforms(img)
    return img

def unsharp_transform(img):
    return unsharp_mask(img, radius=10., amount=1.5)

def remove_calcs(img, file_name):
    save_dir = '/workspace/temp/'
    os.makedirs(save_dir,exist_ok=True)
    # Find or generate mask
    mask_path = os.path.join(save_dir,file_name+'.nii.gz')
    try:        
        mask = nib.load(mask_path).get_data()
        # print('Mask loaded from file')
    except:
        if len(img.shape)==3:
            img2 =img.squeeze()
        else:
            img2=img
        h,w=img.shape
        mask = detect_calcifications_whole_image(img2, 
                                            erosion=0,
                                            method='Ciecholwski',
                                            thr=25).astype(np.int0)
        nifti_mask = nib.Nifti1Image(mask, affine=np.eye(4))
        nib.save(nifti_mask,mask_path)
        # print('Mask generated and saved')
    img = img*(1-mask)
    return img

class DataProcessor(Dataset):
    def __init__(self, img_dir, annot_dir, csv,*args,
                 resize =(2457, 1996), 
                    transformations=None, 
                    resize_img=False,
                give_annot=True, 
                 only_positive=False, 
                 get_class=False, 
                 give_actual_label=False,
                 give_filename=False,
                preprocess=standard_image_preprocess,
                 augmentations=None, **kwargs):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.df = pd.read_csv(csv)
        self.resize = resize
        self.transformations = transformations
        self.give_annot = give_annot
        self.preprocess = preprocess
        self.give_actual_label = give_actual_label
        self.augmentations=augmentations
        self.give_filename=give_filename
        if only_positive:
            # retain only cases where there are annotations:
            self._keep_positive()
        if get_class:
            self._get_class()
    
    def _augment(self,img,annot):
        if self.augmentations is not None:
            # the augmentations
            bbox_params = A.BboxParams(format='pascal_voc') if annot is not None else None
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),# A.Rotate(p=0.5,border_mode=0,limit=20),
                A.OneOf([   A.RandomBrightness(p=1),
                            A.RandomGamma(p=1),],p=0.9,),
            ], bbox_params=bbox_params)

            boxes_labels = [list(b)+[l] for b,l in zip(annot['boxes'], annot['labels'])]
            if len(img.shape)==3:
                if (img.shape[0]==1) or (img.shape[0]==3):
                    img = img.transpose(1,2,0)
            img = img.astype(np.float32)
            data = transform(image=img, bboxes=boxes_labels)
            boxes, labels = [b[:-1] for b in data['bboxes']], [b[-1] for b in data['bboxes']]
            img, annot['boxes'], annot['labels'] = data['image'],boxes, labels
            if len(annot['labels'])==0:
                annot['boxes']=np.zeros((0,4))
                annot['labels']=np.zeros((0,))
        return img,annot
    
    def _to_tensor(self,img,annot):
        if len(img.shape)==3:
            if (img.shape[2]==1) or (img.shape[2]==3):
                img = img.transpose(2,0,1)
        elif len(img.shape)==2:
            img = img.unsqueeze(0)

        img = torch.as_tensor(img, dtype=torch.float32)
        if (annot is not None) and ('labels' in annot.keys()):
            if len(annot['labels'])==0:
                annot['boxes']=np.zeros((0,4))
                annot['labels']=np.zeros((0,))
            annot['boxes'] = torch.as_tensor(annot['boxes'], dtype=torch.float32)
            annot['labels'] = torch.as_tensor(annot['labels'], dtype=torch.int64)
        return img,annot
 
    def _get_image_and_annot(self,file_name):
        img_path = os.path.join(self.img_dir, file_name+'.nii.gz')
        # Load nifti image
        img = nib.load(os.path.join(img_path)).get_data()
        annot = self._get_annot(file_name)
        return img,annot

    
    def __getitem__(self, i):
        file_name = self.df.iloc[i]['file_name']
        img,annot = self._get_image_and_annot(file_name)
        img_path = os.path.join(self.img_dir, file_name+'.nii.gz')
        if img.shape[0]==3:
            img_list = list()
            for i in range(3):
                img_list.append(self.preprocess(img[i], new_size = self.resize, expand_dim=True,
                             adjust_label=False, normalize=True,
                              img_transforms=self.transformations))
            img = np.array(img_list).squeeze()
        else:
            img = self.preprocess(img, new_size = self.resize, expand_dim=True,
                                 adjust_label=False, normalize=True,
                                  img_transforms=self.transformations)
        img, annot = self._augment(img,annot)
        img, annot = self._to_tensor(img,annot)
        if self.give_annot or self.give_filename:
            return img, annot
        return img

    def __len__(self):
        return len(self.df)
    
    def _get_annot(self, file_name):
        if self.give_annot:
            annot_path = os.path.join(self.annot_dir, file_name+'.json')
            annot = json.load(open(annot_path,'r'))
            num_objs = len(annot['boxes'])
            # treat no boxes case
            if len(annot['boxes'])==0:
                annot['boxes']=np.zeros((0,4))
                annot['labels']=np.zeros((0,))
            annot['boxes'] = list(annot['boxes'])
            labels = [0]*num_objs
            if self.give_actual_label:
                labels = list(annot['labels'])
            labels = [l+1 for l in labels] # Labels should be 1 and above, 0 corresponds to background
            annot["labels"] = labels 
            return annot
        return None

    def _get_class(self):
        print('Reading annotations')
        self.df['annot_exists'] = True
        from tqdm import tqdm
        for idx in tqdm(range(len(self.df))):
            file_name = self.df.iloc[idx]['file_name']
            annot = self._get_annot(file_name)
            if len(annot['labels'])==0:
                self.df['annot_exists'].iloc[idx] = False


    def _keep_positive(self):
        self._get_class()
        self.df = self.df[self.df.annot_exists]
        self.df.reset_index(inplace=True)


def threedpreprocess(
        img, *args,
        new_size=None, 
        expand_dim=False, 
        adjust_label=False, 
        normalize=False,
        img_transforms=None,
        n_splits = 3,
        **kwargs):
    d,h,w = img.shape
    zvalues = np.linspace(0,d-1,n_splits+1).astype(int)
    img_splits = list()
    for i in range(len(zvalues)-1):
        z0, z1 = zvalues[i], zvalues[i+1]
        img_splits.append(img[z0:z1])
    # Find mip
    img_splits = [arr.max(0) for arr in img_splits]

    if new_size is not None:
        for i in range(len(img_splits)):
            im = img_splits[i]
            h, w = im.shape
            img_new = np.zeros(new_size)
            img_new[0:h,0:w]=im
            img_splits[i] = img_new.copy()

    img = np.array(img_splits)

    if normalize:
        if img.max() > 1:
            img = img/(2**16-1)
    return img

def change_slice(file_name):
    list_fname = file_name.split('_')
    slice_id = int(list_fname[-1])
    # shift one of -2,-1,0,1,2 slices
    slice_id = slice_id+random.randint(-2,2)
    list_fname[-1]=str(slice_id)

    return '_'.join(list_fname)

class ModDataProcessor(DataProcessor):
    def _get_image_and_annot(self,file_name):
        file_name = change_slice(file_name)
        img_path = os.path.join(self.img_dir, file_name+'.nii.gz')

        while not os.path.exists(img_path):
            file_name = self.df.iloc[i]['file_name']
            file_name = change_slice(file_name)
            img_path = os.path.join(self.img_dir, file_name+'.nii.gz')
        # Load image
        try:
            img = nib.load(os.path.join(img_path)).get_data()
        except:
            file_name = self.df.iloc[i]['file_name']
            img_path = os.path.join(self.img_dir, file_name+'.nii.gz')
            img = nib.load(os.path.join(img_path)).get_data()
        annot = self._get_annot(file_name)

        return img,annot
            

def shift_slice(file_name,ds):
    list_fname = file_name.split('_')
    slice_id = int(list_fname[-1])
    # shift one of -2,-1,0,1,2 slices
    slice_id = slice_id+ds
    list_fname[-1]=str(slice_id)

    return '_'.join(list_fname)

class ModDataProcessor3slices(DataProcessor):
    def __init__(self, *args, random_slice=False, **kwargs):
        self.random_slice = random_slice
        
        self.give_filename = kwargs['give_filename'] if ('give_filename' \
            in kwargs.keys()) else False
        super().__init__(*args, **kwargs)
    def _get_image_and_annot(self,file_name):
        if self.random_slice:
            file_name = change_slice(file_name)
            img_path = os.path.join(self.img_dir, file_name+'.nii.gz')

            while not os.path.exists(img_path):
                file_name = self.df.iloc[i]['file_name']
                file_name = change_slice(file_name)
                img_path = os.path.join(self.img_dir, file_name+'.nii.gz')
        # Getting the other two slices
        behind_slice = shift_slice(file_name,-1)
        front_slice = shift_slice(file_name,+1)
        img_path_behind = os.path.join(self.img_dir, behind_slice+'.nii.gz')
        img_path_front = os.path.join(self.img_dir, front_slice+'.nii.gz')

        if os.path.exists(img_path_behind) and os.path.exists(img_path_front):
            slices_names = [behind_slice, file_name, front_slice]
        elif (not os.path.exists(img_path_behind)) and os.path.exists(img_path_front):
            slices_names = [file_name, front_slice, shift_slice(file_name,+2)]
        else:
            slices_names = [shift_slice(file_name,-2), behind_slice, file_name]
        img_list = list()
        for f in slices_names:
            img_list.append(nib.load(os.path.join(self.img_dir, f+'.nii.gz')).get_data())
        
        img = np.array(img_list, dtype=np.float32)
        annot = self._get_annot(file_name)
        if self.give_filename:
            return img, {'file_name':file_name}
        return img, annot
            

class ModDataProcessor3slicesNoCalcs(DataProcessor):
    def __init__(self, *args, random_slice=False, **kwargs):
        self.random_slice = random_slice
        
        self.give_filename = kwargs['give_filename'] if ('give_filename' \
            in kwargs.keys()) else False
        super().__init__(*args, **kwargs)
    def _get_image_and_annot(self,file_name):
        if self.random_slice:
            file_name = change_slice(file_name)
            img_path = os.path.join(self.img_dir, file_name+'.nii.gz')

            while not os.path.exists(img_path):
                file_name = self.df.iloc[i]['file_name']
                file_name = change_slice(file_name)
                img_path = os.path.join(self.img_dir, file_name+'.nii.gz')
        # Getting the other two slices
        behind_slice = shift_slice(file_name,-1)
        front_slice = shift_slice(file_name,+1)
        img_path_behind = os.path.join(self.img_dir, behind_slice+'.nii.gz')
        img_path_front = os.path.join(self.img_dir, front_slice+'.nii.gz')

        if os.path.exists(img_path_behind) and os.path.exists(img_path_front):
            slices_names = [behind_slice, file_name, front_slice]
        elif (not os.path.exists(img_path_behind)) and os.path.exists(img_path_front):
            slices_names = [file_name, front_slice, shift_slice(file_name,+2)]
        else:
            slices_names = [shift_slice(file_name,-2), behind_slice, file_name]
        img_list = list()
        for f in slices_names:
            img_list.append(nib.load(os.path.join(self.img_dir, f+'.nii.gz')).get_data())
        # Remove calcs
        for i in range(len(slices_names)):
            fname = slices_names[i]
            img_list[i]= remove_calcs(img_list[i],fname)
        img = np.array(img_list, dtype=np.float32)
        annot = self._get_annot(file_name)
        if self.give_filename:
            return img, {'file_name':file_name}
        return img, annot


## SAMPLER
# From https://github.com/ufoym/imbalanced-dataset-sampler/

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.df.annot_exists.iloc[idx]

                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples