
# python generate_annotations.py\
#     -data_dir "../../../Data/raw/"\
#     -out_dir "../../../Output/mip_annotations/"\
#     -train_test_split_dir "../../../Output/faster_rcnn_split/"

# exp1
# Description: We used the mip images, and the annotations mapped
# in 2d. Annotations are expanded by bounding box of the breast
# in order to make algorithm to run. Although TPR2 (TPR at FP=2) is high, this
# may be attributed to finding the breast correctly in most images?
# This is something that should be checked. Algorithm seems to identify
# the stickers as well.  To check also, whether the annotations correspond 
# to only findings that we are after (Checked, Confirmed). Question:
# Is information being lost with mip images?

# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v1.pth"\
#     -img_dir "../../../Output/mip/"\
#     -annot_dir "../../../Output/mip_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test.csv"\
#     -experiment_name "exp1"


# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v1.pth"\
#     -img_dir "../../../Output/mip/"\
#     -annot_dir "../../../Output/mip_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test.csv"\
#     -experiment_name "exp1/eval"


# exp2
# Description: Data as in exp1. Removed boxes of breast 
# with upgrading torchvision to a more recent version.

# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v2.pth"\
#     -img_dir "../../../Output/mip/"\
#     -annot_dir "../../../Output/mip_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test.csv"\
#     -experiment_name "exp2"\
#     -batch_size 6

# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v2.pth"\
#     -img_dir "../../../Output/mip/"\
#     -annot_dir "../../../Output/mip_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test.csv"\
#     -experiment_name "exp2/eval"


# generate data for Akin
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v2.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/sample_slices.csv"\
#     -out_dir "../../../Output/sample_slices_output/"

# exp3
# Description: Same dataset as in exp1. Now we balance
# positive and negative samples during training. Highest
# result up to now. Performance degradation at later epochs.
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v3.pth"\
#     -img_dir "../../../Output/mip/"\
#     -annot_dir "../../../Output/mip_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test.csv"\
#     -experiment_name "exp3"\
#     -batch_size 6


# exp4
# Description: 3 channel input, each channel is MIP of 1/3 
# of the slices. Balancing samples during training. Highest
# performance so far. Performance degradation. Training
# was interrupted when it wasn't learning
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v4.pth"\
#     -img_dir "../../../Output/nifti/"\
#     -annot_dir "../../../Output/mip_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test.csv"\
#     -experiment_name "exp4"\
#     -batch_size 6

# exp5
# Description: same as exp4 but no pretraining. Resulted in inferior
# Performance.

# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v5.pth"\
#     -img_dir "../../../Output/nifti/"\
#     -annot_dir "../../../Output/mip_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test.csv"\
#     -experiment_name "exp5"\
#     -batch_size 6


# python generate_all_slices_annotations.py\
#     -data_dir "../../../Data/raw/"\
#     -out_dir "../../../Output/nifti_single_slice/"\
#     -nifti_dir "../../../Output/nifti/"\
#     -train_test_split_dir "../../../Output/faster_rcnn_split/"

# exp6
# Description: Working on single slices with 25% depth annotations
# Balancing dataset in both training and validation
# 1200 samples in training epoch, 300 in validating
# Note: score is not comparable to previous experiments

# python train.py\
    # -weights_path "./checkpoints/Tomo_FastRCNN_v6.pth"\
    # -img_dir "../../../Output/nifti_single_slice/"\
    # -annot_dir "../../../Output/single_slice_annotations/"\
    # -train_csv "../../../Output/faster_rcnn_split/train-single.csv"\
    # -val_csv "../../../Output/faster_rcnn_split/test-single.csv"\
    # -experiment_name "exp6"\
    # -batch_size 6


# exp7
# Description: Working on single slices of lesion centers only
# Reduced learning rate by 10 compare to before
# Interrupted; val performance small
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v7.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp7"\
#     -batch_size 6

# exp8
# Description: Working on single slices of lesion centers only
# Learning rate back to 1E-3
# Interrupted because overfitted fast
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v7.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp8"\
#     -batch_size 6\
#     -lr_rate 0.001


# exp9
# Description: Working on single slices of lesion centers only
# Increase weight decay 
# Interrupted because overfitted fast
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v9.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp9"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.005

# exp10
# Description: Working on single slices of lesion centers only
# Increase weight decay 
# No absolute overfitting. 
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v10.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp10"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1

# exp11
# Description: Lesion center slices and around (-2 up +2) slices
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v11.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp11-4"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1


# # exp12
# # Description: Similar to exp11.
# # Lesion center slices and around (-4 up +4) slices
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v12.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp12"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1


# exp 13
# Description: Similar to exp11.
# # Frozen backbone. Interrupted; poor performance
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v13.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp13"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1

# python generate_all_slices_annotations.py\
#     -data_dir "../../../Data/raw/"\
#     -out_dir "../../../Output/single_slice_annotations/"\
#     -nifti_dir "../../../Output/nifti/"\
#     -train_test_split_dir "../../../Output/faster_rcnn_split/"

# exp14
# Description: Lesion center slices and around (-2 up +2) slices
# Added separate labels for AD and masses
# Interrupted when converged
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v14.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp14"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1


# generate data for Akin
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v11.pth"\
#     -img_dir "../../../Output/nifti_single_slice2/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-single.csv"\
#     -out_dir "../../../Output/validation_output/"

# exp15
# Description: Lesion center slices and around (-2 up +2) slices
# Used Anil's weights
# Interrupted: no improvement in performance. 
# Highest performance so far
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v15.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp15"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1

# python generate_all_slices_annotations.py\
#     -data_dir "../../../Data/raw/"\
#     -out_dir "../../../Output/single_slice_annotations/"\
#     -nifti_dir "../../../Output/nifti/"\
#     -train_test_split_dir "../../../Output/faster_rcnn_split/"

# exp16
# Description: 
# Data: Lesion center slices and normal volume center slices
#  and around (-2 up +2) slices, Anil's weights
# Interrupted: no performance and not learning
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v16.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-and-normal-centers.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp16"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1

# exp17
# Description: 
# Data: Lesion center slices and normal volume center slices
#  and around (-2 up +2) slices, load exp15 model, froze backbone
# Changed lr, weight decay, both decreased
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v17.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-and-normal-centers.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp17"\
#     -batch_size 6\
#     -lr_rate 0.00001\
#     -weight_decay 0.00001

# exp18
# Same as in exp17
# Input is 3 consecutive slices with central slices
# the ones used in exp17
# Interrupted when converged. Highest performance so far
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v18.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-and-normal-centers.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp18"\
#     -batch_size 6\
#     -lr_rate 0.00001\
#     -weight_decay 0.00001


# exp19
# Same as in exp11
# Input is 3 consecutive slices with central slices
# the ones used in exp11
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v19.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp19-1"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1

# generate data for Akin
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v18.pth"\
#     -img_dir "../../../Output/nifti_single_slice_VALSET/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/csv.csv"\
#     -out_dir "../../../Output/VALSET_output_model18_correct/"


# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v18.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp18/eval"


# exp20
# Same as in exp15
# Using Noor's weights in the backbone
# Didn't improve compare to Anil's weights
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v20.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp20"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1


# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v18.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "to_remove_eval"\
#     -path_to_output "../../../Output/partial-results/fasterrcnn/"


# exp21
# Continue training exp18 best
# Freesing all components but roi_heads
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v21.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-and-normal-centers.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp21"\
#     -batch_size 6\
#     -lr_rate 0.00001\
#     -weight_decay 0.00001


# exp22
# Data same as in exp13
# Change the transformer of FasterRCNN to effectively not resizing the image
# Performance not large
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v22.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp22-2"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1

# exp23
# From coco weights
# And some augmentations
# Notice it wasn't named correctly
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v23.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "to_remove370"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001





# python train.py\
#     -weights_path "./checkpoints/to_remove2.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-and-normal-centers.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "to_remove390"\
#     -batch_size 6\
#     -lr_rate 0.0001\
#     -weight_decay 0.001


# generate data for Akin
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v23.pth"\
#     -img_dir "../../../Output/nifti_single_slice_VALSET/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/csv.csv"\
#     -out_dir "../../../Output/VALSET_output_model23/"

# generate data for Akin
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v23.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/faster_rcnn_split/test-single.csv"\
#     -out_dir "../../../Output/TRAINSET_TESTPART_output_model23/"


# exp24
# From coco weights
# And some augmentations
# Notice it wasn't named correctly
# Only lesion centers and 3-consecutive slices and the (-2,+2) random slice rule
# Interuupted, stopped improving
# Saving using mAP as rule
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v24.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp24"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001

# exp25
# same as exp24 but
# Lesion centers and normal centers, and balanced sampling in training
# Interrupted stopped improving
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v25.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-and-normal-centers.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp25"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001


# exp26
# same as exp24 but
# but with different normalization of channels in model.transform
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v26.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp26"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001





# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v23.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp23/eval-5"\
#     -path_to_output "../../../Output/partial-results/fasterrcnn23/"

# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v18.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp18/eval-5"\
#     -path_to_output "../../../Output/partial-results/fasterrcnn18/"


# python train.py\
#     -weights_path "./checkpoints/to_remove.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "to_remove5"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.1

# exp27
# same as 24
# add augmentations: flips + noise
# removing the color white 2**16-1
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v27.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp27-2"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001

# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v27.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp27/eval"\
#     -path_to_output "../../../Output/partial-results/fasterrcnn27/"

# generate data for Akin
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v27.pth"\
#     -img_dir "../../../Output/nifti_single_slice_VALSET/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/csv.csv"\
#     -out_dir "../../../Output/VALSET_output_model27/"


# generate data for Akin
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v27.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/faster_rcnn_split/test-single.csv"\
#     -out_dir "../../../Output/TRAINSET_TESTPART_output_model27/"


# python generate_cross_validation_splits.py\
#     -data_dir "../../../Data/raw/"\
#     -out_dir "../../../Output/mip_annotations/"\
#     -train_test_split_dir "../../../Output/faster_rcnn_split/"


# exp 28
# cross-validation of exp27
# add augmentations: flips + noise
# for split in 1 2 3 4
# do 
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v28_split${split}.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center-split${split}.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center-split${split}.csv"\
#     -experiment_name "exp28-split${split}"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001\
#     -num_epochs 45
# done

# for split in 1 2 3 4
# do 
# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v28_split${split}.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-lesion-center-split${split}.csv"\
#     -experiment_name "exp28-split${split}/eval"\
#     -path_to_output "../../../Output/partial-results/fasterrcnn28-split${split}/"
# done


# # generate data for Akin
# for split in 1 2 3 4
# do 
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v28_split${split}.pth"\
#     -img_dir "../../../Output/nifti_single_slice_VALSET/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/csv.csv"\
#     -out_dir "../../../Output/VALSET_output_model28_split${split}/"
# done


# # generate data for Akin
# for split in 1 2 3 4
# do 
# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v28_split${split}.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/faster_rcnn_split/test-single-split${split}.csv"\
#     -out_dir "../../../Output/TRAINSET_TESTPART_output_model28_split${split}/"
# done

# exp29
# same as exp27 without removing the white color
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v29.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp29"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001

# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v29.pth"\
#     -img_dir "../../../Output/nifti_single_slice_VALSET/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/csv.csv"\
#     -out_dir "../../../Output/VALSET_output_model29/"


# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v29.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/faster_rcnn_split/test-single.csv"\
#     -out_dir "../../../Output/TRAINSET_TESTPART_output_model29/"

# exp30
# continue learning on exp27
# # Introduce normal slices
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v30.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-and-normal-centers.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp30"\
#     -batch_size 6\
#     -lr_rate 0.00001\
#     -weight_decay 0.00001


# python train.py\
#     -weights_path "./checkpoints/to_remove.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "to_remove102"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001 

# python evaluate.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v27.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "to_remove"


# exp31
# Training on whole dataset
# python train.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v31.pth"\
#     -img_dir "../../../Output/nifti_single_slice/"\
#     -annot_dir "../../../Output/single_slice_annotations/"\
#     -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
#     -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
#     -experiment_name "exp31"\
#     -batch_size 6\
#     -lr_rate 0.001\
#     -weight_decay 0.001


# python infer.py\
#     -weights_path "./checkpoints/Tomo_FastRCNN_v31.pth"\
#     -img_dir "/workspace/Output/nifti_single_slice_VALSET/"\
#     -annot_dir "./"\
#     -csv "/workspace/Output/csv.csv"\
#     -out_dir "/workspace/Output/VALSET_output_model31/"

# exp32
# Similar to 27 using Noor's weights
# Using DDSM_INbreast_SPIE_augmented.pth
python train.py\
    -weights_path "./checkpoints/Tomo_FastRCNN_v32.pth"\
    -img_dir "../../../Output/nifti_single_slice/"\
    -annot_dir "../../../Output/single_slice_annotations/"\
    -train_csv "../../../Output/faster_rcnn_split/train-lesion-center.csv"\
    -val_csv "../../../Output/faster_rcnn_split/test-lesion-center.csv"\
    -experiment_name "exp32"\
    -batch_size 6\
    -lr_rate 0.001\
    -weight_decay 0.001