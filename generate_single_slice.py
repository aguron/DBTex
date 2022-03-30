import os
import argparse
import nibabel as nib
import numpy as np
import ray
import pandas as pd

@ray.remote
def generate_single_slice(threed_path, twod_img_dir):
    '''Generates maximum intensity projection (MIP) view
    from 3d dbt in nifti 
    '''
    # read nifti image as int array
    assert(os.path.exists(threed_path))
    img = nib.load(threed_path)
    img = img.get_data()
    threedfname = os.path.basename(threed_path)
    fname = threedfname[:-7]
    for i in range(img.shape[0]):
        twodfname = '{}_{}.nii.gz'.format(fname,i)
        twod_path = os.path.join(twod_img_dir, twodfname)
        # save in nifti compressed
        assert(twod_path[-7:]=='.nii.gz')
        nib_img = nib.Nifti1Image(img[i], affine=np.eye(4))
        nib.save(nib_img, twod_path)
    # Check all nifti files
    slices_correct = True
    for i in range(img.shape[0]):
        twodfname = '{}_{}.nii.gz'.format(fname,i)
        twod_path = os.path.join(twod_img_dir, twodfname)
        try:
            nib.load(twod_path).get_data()
        except:
            slices_correct=False
            print('Regenerating slices for{}'.format(threedfname))
    if not slices_correct:
        generate_single_slice(threed_path, twod_img_dir)
    return None

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-in_nifti_dir',
                            metavar='in_nifti_dir',
                            type=str,
                            help='Directory with DBT in nifti format')
    arg_parser.add_argument('-out_dir',
                            metavar='out_dir',
                            type=str,
                            help='Output directory for nifti files')
    arg_parser.add_argument('-num_workers',
                            metavar='num_workers',
                            type=int,
                            default=1,
                            help='Number of CPUs to be used')

    args=arg_parser.parse_args()

    print("Parameters\n",args, "\n")
    in_nifti_dir = args.in_nifti_dir
    out_dir = args.out_dir
    num_workers = args.num_workers

    # Find all nii.gz files in in_nifti_dir
    files_list = list()
    for wdir,_,files in os.walk(in_nifti_dir):
        if len(files)!=0:
            for f in files:
                files_list.append(os.path.join(wdir,f))
    print('{} NIFTI files found'.format(len(files_list)))

    # Create output dir if doesn't exist
    os.makedirs(out_dir,exist_ok=True)

    print(num_workers)
    # ray.init(num_cpus=num_workers)
    # ray_tasks=list()
    # # Convert files
    # for f in files_list:
    #     # unique identifier for image
    #     ray_tasks.append(generate_single_slice.remote(f,out_dir))
    # ray.get(ray_tasks)
    # ray.shutdown()

    csv_out_dir = os.path.dirname(os.path.dirname(out_dir))
    csv_path = os.path.join(csv_out_dir,'csv.csv')
    file_names = [f[:-7] for f in os.listdir(out_dir)]
    pd.DataFrame({'file_name':file_names}).to_csv(csv_path)

if __name__=="__main__":
    main()