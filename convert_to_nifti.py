import os
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from tomolib.duke_dbt_data import dcmread_image
import ray

@ray.remote
def convert_dicom_to_nifti(dicom_fpath, nifti_path, view):
    '''Converts dicom to nifti
    '''
    # read dicom image
    img = dcmread_image(dicom_fpath, view)
    # save in nifti compressed
    assert(nifti_path[-7:]=='.nii.gz')
    nib_img = nib.Nifti1Image(img, affine=np.eye(4))
    nib.save(nib_img, nifti_path)
    return None

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-dicom_dir',
                            metavar='dicom_dir',
                            type=str,
                            help='Dicom directory')
    arg_parser.add_argument('-out_dir',
                            metavar='out_dir',
                            type=str,
                            help='Output directory for nifti files')
    arg_parser.add_argument('-csv_path',
                            metavar='csv_path',
                            type=str,
                            help='CSV path, to retrieve actual view')
    arg_parser.add_argument('-num_workers',
                            metavar='num_workers',
                            type=int,
                            default=1,
                            help='Number of CPUs to be used')

    args=arg_parser.parse_args()

    print("Parameters\n",args, "\n")
    dicom_dir = args.dicom_dir
    out_dir = args.out_dir
    csv_path = args.csv_path
    num_workers = args.num_workers

    # Read csv
    df = pd.read_csv(csv_path)
    # Find all dcm files in dicom_dir
    files_list = list()
    for wdir,_,files in os.walk(dicom_dir):
        if len(files)!=0:
            for f in files:
                files_list.append(os.path.join(wdir,f))
    print('{} DICOM files found'.format(len(files_list)))

    # Create output dir if doesn't exist
    os.makedirs(out_dir,exist_ok=True)

    print(num_workers)
    ray.init(num_cpus=num_workers)
    ray_tasks=list()
    # Convert files
    for f in files_list:
        # unique identifier for image
        uid = f.split('/')[-2]
        search =df.descriptive_path.str.contains(uid)
        # assert that is indeed unique
        assert(search.sum()==1)
        # find it's location in df
        idx = np.where(search)[0][0]
        view = df.iloc[idx].View
        nifti_path = os.path.join(out_dir,uid+".nii.gz")
        assert (not os.path.exists(nifti_path)), "Nifti file exists"
        ray_tasks.append(convert_dicom_to_nifti.remote(f, nifti_path, view))
    ray.get(ray_tasks)
    ray.shutdown()


if __name__=="__main__":
    main()