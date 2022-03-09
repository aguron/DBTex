# [DBTex (Phase One) Submission](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM-NCI+DAIR+Digital+Breast+Tomosynthesis+Lesion+Detection+Challenge+%28DBTex%29+-+Phase+1)

## Faster R-CNN 2D (In-Slice) Detection Ensembling followed by 3D Reconstruction

### Team Members
Akinyinka Omigbodun, Chrysostomos Marasinou, Noor Nakhaei, Anil Yadav, Nina Capiro, Bo Li, Anne Hoyt, William Hsu

### Summary
The algorithm for our submission had three main stages:
* a first stage to detect biopsied tissue in DBT slices with a convolutional neural network (CNN) detector,
* followed by a stage to reconstruct biopsied tissue volumes from detections in slices, and
* a final stage to remove false-positive predictions with clinically significant skin markers using a detector with simple geometric rules.

Our algorithm achieved a mean sensitivity of 0.814 on the DBTex test set and placed 4th in the challenge.

### 2D (In-slice) Lesion Detection Ensembling
We utilized Faster R-CNN, a 2D detector with a Feature Pyramid Network (FPN) feature extractor and a Resnet-50 backbone. The model was initialized with weights from pre-training with the COCO train2017 data split. The input to the model was 3 consecutive slices that were concatenated as a 3-channel image. For training, slices around the lesion centers z (in the range, z-3 to z+3) were used with random flipping, brightness changes, and gamma adjustment augmentations applied. To monitor the performance during training, a portion of the training set was held out (20% patient-wise); The training set was split in 5, and 5 models were trained in a cross-validation scheme. The mean sensitivity on the held-out data was 0.780 ± 0.07 at 2 false positives per slice. During inference on the DBTex validation and testing data, all 5 models were applied to every slice, generating 5 sets of bounding box predictions for each slice. Detections from the 3 individually best of 5 models on the DBTex validation data were selected for the final ensembling [1], as the mean sensitivity was better than with all 5 models.

### 3D Lesion Candidate Generation
We then took the 2D in-slice bounding box predictions and combined them into 3D candidates based on:
1. the proximity of slices in a DBT scan (being within 50% of the total number of slices of each other),
2. 2D bounding box score threshold (at least 0.85 on a scale of 0 to 1), and
3. 2D bounding boxes with an intersection over smaller intersecting box (IoSIB) greater than 0.75.

IoSIB gave better results than the standard intersection over union (IoU). 3D candidates with a depth of 1 (a single slice) were removed. The score assigned to a 3D candidate was the average of the top 10 scores of all 2D bounding boxes involved in its construction.

### Blob Detection for False Positive Removal
To remove false positives (with clinical circular markers), we used the SimpleBlobDetector in the python OpenCV module (cv2) as summarized in the table below.

| Parameter         | Value               |
| ---               | ---                 |
| filterbyArea      | True                |
| minArea           | 1000                |
| filterByConvexity | True                |
| minConvexity      | 0.001               |
| maxConvexity      | 0.4                 |
| filterByInertia   | True                |
| minInertiaRatio   | 0.01                |
| maxInertiaRatio   | 1                   |
| minThreshold      | 50 ([0, 255] range)  |
| maxThreshold      | 150 ([0, 255] range) |

References
1. Casado-García, Á., & Heras, J. (2020). Ensemble methods for object detection. In *ECAI 2020* (pp. 2688-2695). IOS Press.
