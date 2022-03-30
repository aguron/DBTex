# python convert_to_nifti.py\
#     -dicom_dir "../../Data/raw/Breast-Cancer-Screening-DBT/"\
#     -out_dir "../../Output/nifti/"\
#     -csv_path "../../Data/raw/BCS-DBT file-paths-train.csv"\
#     -num_workers 20

# python generate_mip.py\
#     -in_nifti_dir "../../Output/nifti/"\
#     -out_dir "../../Output/mip/"\
#     -num_workers 20


# python generate_single_slice.py\
#     -in_nifti_dir "../../Output/nifti/"\
#     -out_dir "../../Output/nifti_single_slice/"\
#     -num_workers 20

# python generate_single_slice.py\
#     -in_nifti_dir "../../Output/nifti/"\
#     -out_dir "../../Output/nifti_single_slice2/"\
#     -num_workers 20


# python convert_to_nifti.py\
#     -dicom_dir "/workspace/Output/ValidationSetRaw/Breast-Cancer-Screening-DBT/"\
#     -out_dir "/workspace/Output/nifti_VALSET/"\
#     -csv_path "/workspace/Output/ValidationSetRaw/BCS-DBT file-paths-validation.csv"\
#     -num_workers 20


python generate_single_slice.py\
    -in_nifti_dir "/workspace/Output/nifti_VALSET/"\
    -out_dir "/workspace/Output/nifti_single_slice_VALSET/"\
    -num_workers 20