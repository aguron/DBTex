import numpy as np
import torch
import cv2
import torch.nn as nn
from dataloader import DataProcessor, ImbalancedDatasetSampler, threedpreprocess, ModDataProcessor, ModDataProcessor3slices, unsharp_transform
from dataloader import remove_calcs, ModDataProcessor3slicesNoCalcs
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from skimage import io
import torchvision
from skimage.color import rgb2gray
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from skimage.util import img_as_float, img_as_ubyte
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import os
import pickle
from vocdata.new_dataset import VOCWrapper
import math, sys


from torchvision.models.detection import FasterRCNN, backbone_utils
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.transform import GeneralizedRCNNTransform


from torchvision.models.detection.rpn import AnchorGenerator



class FasterRCNN:
    def __init__(self, model_path=None,weights_path=None, 
        backbone_weights_path=None,
        num_classes=3, state_dict=True):
        torch.manual_seed(0)

        self.model_path = model_path
        if self.model_path is not None:
            self.model = torch.load(model_path)
        else:
            self.model = self._get_model_instance_segmentation(num_classes)
        
        if weights_path is not None:
            if state_dict:
                state_dict  = torch.load(weights_path)
                self.model.load_state_dict(state_dict)
                print('Weights loaded')
            else:
                self.model = torch.load(weights_path)
                print('Model loaded')

        if backbone_weights_path is not None:
            backbone = resnet50_fpn_backbone_modified(
                weights_path=backbone_weights_path, state_dict=state_dict)
            self.model.backbone.body.load_state_dict(backbone.state_dict(),
                         strict=False)


        # Change of anchorgenerator
        # Change RPN
        # self._model_change_rpn()
        # Change aspect ratios
        # self.model.rpn.anchor_generator.aspect_ratios = ((0.54,1,1.85),)*5
        # self.model.rpn.anchor_generator.sizes = ((16,),(32,), (64,), (128,), (256,))        # Check for CUDA
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            self.device = torch.device("cpu")
            print("="*30)
            print("Running on CPU")
            print("=" * 30)
        else:
            print("=" * 30)
            self.device = torch.device("cuda:0")
            print("CUDA is available!")
            print("=" * 30)

        # Load model on CUDA/CPU
        self.model.to(self.device)

    # HELPER FUNCTIONS
    def _get_model_instance_segmentation(self, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
        return model
    def _model_change_rpn(self):    
        # Define RPN 
        from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork


        out_channels = self.model.backbone.out_channels
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.3,0.5,1,2,3.),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )
        rpn_head = RPNHead(
            out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n_train=2000
        rpn_pre_nms_top_n_test=1000
        rpn_post_nms_top_n_train=2000
        rpn_post_nms_top_n_test=1000
        rpn_nms_thresh=0.7
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn_fg_iou_thresh=0.7
        rpn_bg_iou_thresh=0.3
        rpn_batch_size_per_image=256
        rpn_positive_fraction=0.5
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        self.model.rpn = rpn

    def _get_default_transform(self):
        custom_transforms = []
        # custom_transforms.append(unsharp_transform)
        # custom_transforms.append(torchvision.transforms.ToTensor())
        return torchvision.transforms.Compose(custom_transforms)



    def _collate_fn(self, batch):
        return tuple(zip(*batch))

    # TRAIN FUNCTION
    def train_model(self, 
        path_to_images, 
        path_to_annotations,
        train_csv,
        val_csv,
        transformation=None, 
        batch_size=5, 
        lr_rate=0.0001, 
        weight_decay = 0.0005,
        num_epochs=200,
        model_path=None,
        experiment_name = None,
        num_workers=12):

        writer = SummaryWriter("runs/{}".format(experiment_name))


        if transformation is None:
            transformation=self._get_default_transform()
        

        train_ds = ModDataProcessor3slices(path_to_images,path_to_annotations,train_csv, 
            transformations=transformation, get_class=True, random_slice=True, augmentations=True)
        val_ds = ModDataProcessor3slices(path_to_images,path_to_annotations,val_csv, 
            transformations=transformation, get_class=True)
        train_ds_eval = ModDataProcessor3slices(path_to_images,path_to_annotations,train_csv, 
            transformations=transformation, get_class=True,  augmentations=True)
        # To add the validation set in training
        # train_ds.df = pd.concat((train_ds.df,val_ds.df),ignore_index=True)
        # val_ds.df = train_ds.df.copy()
        # train_ds_eval.df= train_ds_eval.df.iloc[0:1]

        # train_ds_eval.df = train_ds_eval.df.iloc[0:100]
        # train_ds = ModDataProcessor(path_to_images,path_to_annotations,train_csv, 
        #     transformations=transformation, augmentations=True)
        # val_ds = DataProcessor(path_to_images,path_to_annotations,val_csv, 
        #     transformations=transformation)
        # train_ds_eval = DataProcessor(path_to_images,path_to_annotations,train_csv, 
        #     transformations=transformation)
        # train_ds_eval.df = train_ds_eval.df.iloc[0:100]

        # VOC dataset
        # voc_dir = '/workspace/Projects/newFasterRCNN/simple-faster-rcnn-pytorch/dataset/VOCdevkit/VOC2007/'
        # ds = VOCWrapper(voc_dir)
        # n_samples = len(ds.voc_ds.ids)
        # train_ds = VOCWrapper(voc_dir)
        # train_ds.voc_ds.ids = train_ds.voc_ds.ids[0:int(n_samples*0.8)]
        # train_ds_eval = VOCWrapper(voc_dir)
        # train_ds_eval.voc_ds.ids = train_ds_eval.voc_ds.ids[0:int(n_samples*0.2)]
        # val_ds = VOCWrapper(voc_dir)
        # val_ds.voc_ds.ids = val_ds.voc_ds.ids[int(n_samples*0.8):]

        n_val = len(val_ds)
        n_train = len(train_ds)
        print("Images for Training:", n_train)
        print("Images for Validation:", n_val)
        trainloader = DataLoader(train_ds, 
            batch_size=batch_size, 
            sampler=None,#ImbalancedDatasetSampler(train_ds,num_samples=170), 
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=num_workers)
        
        validloader = DataLoader(val_ds, 
            batch_size=batch_size,
            sampler=None, 
            shuffle=False, 
            collate_fn=self._collate_fn,
            num_workers=num_workers)
        trainevalloader = DataLoader(train_ds_eval, 
            batch_size=batch_size,
            sampler=None,#ImbalancedDatasetSampler(train_ds,num_samples=100), 
            shuffle=False, 
            collate_fn=self._collate_fn,
            num_workers=num_workers)

        print("=" * 30)
        # Disable training for backbone, freezing
        # for p in self.model.backbone.parameters():
        #     p.requires_grad = False
        # Freeze all except roi heads
        # for p in self.model.parameters():
        #     p.requires_grad = False
        # for p in self.model.roi_heads.parameters():
        #     p.requires_grad = True

        # self.model.transform.max_size = self.model.transform.max_size-100
        # self.model.transform.min_size = (self.model.transform.min_size[0]-100,)
        # self.model.transform.image_mean = [0.5,0.5,0.5]
        # self.model.transform.image_std = [0.125,0.225,0.125]
        # print(self.model.transform)  
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr_rate, momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        # changes the resizing module of the network
        # model_change_scheduler = ChangeModelOnPlateau(model=self.model,mode='max', patience=5, verbose=True)
        run_tpr2 = 0.0
        i = 0
        for epoch in range(num_epochs):
            train_loss = 0.0
            iou = 0.0
            epoch_loss = []
            self.model.train()
            for imgs, annotations in tqdm(trainloader, desc="Training Epoch {}".format(epoch)):
                i += 1
                imgs = torch.stack(imgs).to(self.device, dtype=torch.float)
                annotations = [{k: v.to(self.device) for k, v in t.items()} for t in annotations]
                loss_dict = self.model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())
                if not math.isfinite(losses.item()):
                    print("Loss is {}, stopping training".format(losses.item()))
                    print(loss_dict)
                    sys.exit(1)
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                epoch_loss.append(float(losses.item() * len(imgs)))
                detached_loss = losses.detach().item() * len(imgs)
                train_loss += detached_loss
                writer.add_scalar("Loss",losses.item(),i)
            scheduler.step(np.mean(epoch_loss))
            epoch_tpr2 = evaluate_epoch(self.model,
                                        validloader,
                                        self.device,
                                        epoch,
                                        writer,
                                        msg="Validation")
            # model_change_scheduler.step(epoch_tpr2)
            # For comparison purposes
            evaluate_epoch(self.model,
                            trainevalloader,
                            self.device,
                            epoch,
                            writer,
                            msg="Train")
            if epoch_tpr2 > run_tpr2:
                run_tpr2 = epoch_tpr2
                torch.save(self.model,open(model_path,'wb'))
                print("Model saved")
            if epoch_tpr2>0.90:
                print('Early stopping')
                sys.exit(1)
                    
# EVALUATE FUNCTION
    def evaluate_model(self, 
        path_to_images, 
        path_to_annotations,
        csv,
        path_to_output=None,
        transformation=None, 
        batch_size=5, 
        model_path=None,
        experiment_name = None):

        writer = SummaryWriter("runs/{}".format(experiment_name))


        if transformation is None:
            transformation=self._get_default_transform()

        ds = ModDataProcessor3slices(path_to_images,path_to_annotations,csv, 
            transformations=transformation, get_class=True, give_actual_label=True)

        n_samples = len(ds)
        print("Images for Evaluation:", n_samples)
        dataloader = DataLoader(ds, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._collate_fn,
            num_workers=8)
        print("=" * 30)


        tpr = evaluate_epoch(self.model,
                        dataloader,
                        self.device,
                        1,
                        writer,
                        msg="eval",
                        n_batches_visualize = math.inf, # to visualize all 
                        path_to_output=path_to_output,
                        )
        print(tpr)
        writer.close()

# INFER FUNCTION
    def infer(self, 
        path_to_images, 
        path_to_annotations,
        csv,
        path_to_output,
        transformation=None, 
        batch_size=10, 
        model_path=None,
        ):


        if transformation is None:
            transformation=self._get_default_transform()

        ds = ModDataProcessor3slices(path_to_images,path_to_annotations,csv, 
            transformations=transformation, give_annot=False, give_filename=True)
        n_samples = len(ds)
        print("Images for Inference:", n_samples)
        dataloader = DataLoader(ds, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self._collate_fn,
            num_workers=20)
        print("=" * 30)



        with torch.no_grad():
            self.model.eval()
            total_fnames=set()
            for imgs, annotations in tqdm(dataloader):
                imgs = torch.stack(imgs).to(self.device, dtype=torch.float)
                fname_list = [annot['file_name'] for annot in annotations]
                output = self.model(imgs)
                output = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in output]
                for fname, output_single in zip(fname_list,output):
                    out_path = os.path.join(path_to_output,fname+'.pkl')
                    total_fnames.add(out_path)
                    pickle.dump(output_single, open(out_path,'wb'))


            

                




def evaluate_batch(y_pred, y_true):
    y_pred = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in y_pred]
    y_true = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in y_true]
    df_eval = pd.DataFrame()
    n_objects = 0
    for i in range(len(y_pred)):
        df_gt_boxes = pred2boxes(y_true[i], threshold=1.0)
        n_objects += len(df_gt_boxes)
        df_gt_boxes["GTID"] = np.random.randint(10e10) * (1 + df_gt_boxes["X"])
        df_pred_boxes = pred2boxes(y_pred[i])
        df_pred_boxes["PID"] = np.random.randint(10e12)
        df_pred_boxes["TP"] = 0
        if len(df_gt_boxes)>0:
            df_pred_boxes["GTID"] = np.random.choice(
                list(set(df_gt_boxes["GTID"])), df_pred_boxes.shape[0]
            )
        for index, pred_box in df_pred_boxes.iterrows():
            tp_list = [
                (j, is_tp(pred_box, x_box), iou_is(pred_box,x_box)) for j, x_box in df_gt_boxes.iterrows()
            ]
            if any([tp[1] for tp in tp_list]):
                tp_index = [tp[0] for tp in tp_list if tp[1]][0]
                df_pred_boxes.at[index, "TP"] = 1
                df_pred_boxes.at[index, "IoU"] = tp_list[tp_index][2]
                df_pred_boxes.at[index, "GTID"] = df_gt_boxes.at[tp_index, "GTID"]
        df_eval = df_eval.append(df_pred_boxes, ignore_index=True, sort=False)
        
    return df_eval, n_objects

def pred2boxes(pred, threshold=None):

    scores = pred['scores']
    boxes = pred['boxes']
    
    xmid = (boxes[:,0]+boxes[:,2])/2
    ymid = (boxes[:,1]+boxes[:,3])/2
    w = np.abs(boxes[:,0]-boxes[:,2])
    h = np.abs(boxes[:,1]-boxes[:,3])
    
    df_dict = {"Score": scores, "X": xmid, "Y": ymid, "Width": w, "Height": h}
    df_boxes = pd.DataFrame(df_dict)
    df_boxes.sort_values(by="Score", ascending=False, inplace=True)
    return df_boxes

def is_tp(pred_box, true_box, min_iou=0.5):
    # box: center point + dimensions
    y,x,h,w = pred_box["Y"], pred_box["X"], pred_box["Height"],pred_box["Width"]
    x1,x2,y1,y2 = x-w/2, x+w/2, y-h/2,y+h/2
    pred_bb = (x1,x2,y1,y2)
    y,x,h,w = true_box["Y"], true_box["X"], true_box["Height"],true_box["Width"]
    x1,x2,y1,y2 = x-w/2, x+w/2, y-h/2,y+h/2
    true_bb = (x1,x2,y1,y2)
    iou = bboxIoU(pred_bb,true_bb)
    return iou >= min_iou

def iou_is(pred_box, true_box):
    # box: center point + dimensions
    y,x,h,w = pred_box["Y"], pred_box["X"], pred_box["Height"],pred_box["Width"]
    x1,x2,y1,y2 = x-w/2, x+w/2, y-h/2,y+h/2
    pred_bb = (x1,x2,y1,y2)
    y,x,h,w = true_box["Y"], true_box["X"], true_box["Height"],true_box["Width"]
    x1,x2,y1,y2 = x-w/2, x+w/2, y-h/2,y+h/2
    true_bb = (x1,x2,y1,y2)
    iou = bboxIoU(pred_bb,true_bb)
    return iou

def bboxOverlap(bb1, bb2):
    
    x1,x2,y1,y2 = bb1
    x1b,x2b,y1b,y2b = bb2
    
    x_criterion = (x2 >= x1b) and (x2b >= x1)
    y_criterion = (y2 >= y1b) and (y2b >= y1)
    
    if x_criterion and y_criterion:
        return True
    return False


def bboxIoU(bb1, bb2):
    
    iou=0
    if bboxOverlap(bb1,bb2):
        x1,x2,y1,y2 = bb1
        x1b,x2b,y1b,y2b = bb2
        dx = min(x2b-x1,x2-x1b)
        dy = min(y2b-y1,y2-y1b)
        intersect = dx*dy
        union = (x2-x1)*(y2-y1)+(x2b-x1b)*(y2b-y1b)-intersect
        iou = intersect/union
    return iou


def froc(df,total_pos, return_thresholds=False, max_fps=4.0):
    total_slices = len(df.drop_duplicates(subset=["PID"]))
    tpr = [0.0]
    fps = [0.0]
    max_fps = max_fps
    thresholds = sorted(df[df["TP"] == 1]["Score"], reverse=True)
    thresholds_return = [1.0]
    for th in thresholds:
        df_th = df[df["Score"] >= th]
        df_th_unique_tp = df_th.drop_duplicates(subset=["PID", "TP", "GTID"])
        num_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = num_tps_th / total_pos
        num_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = num_fps_th / total_slices
        if fps_th > max_fps:
            tpr.append(tpr[-1])
            fps.append(max_fps)
            thresholds_return.append(th)
            break
        tpr.append(tpr_th)
        fps.append(fps_th)
        thresholds_return.append(th)
    if np.max(fps) < max_fps:
        tpr.append(tpr[-1])
        fps.append(max_fps)
        thresholds_return.append(0.)
    if return_thresholds:
        return tpr, fps,thresholds_return
    return tpr, fps


def evaluate_epoch(model,dataloader,device,epoch,writer,msg="Validation Epoch",
                n_batches_visualize = 1, path_to_output=None):
    with torch.no_grad():
        model.eval()
        ival=0
        idx=0
        total_n_objects = 0
        df_validation_pred = pd.DataFrame()


        for images, annotations in tqdm(dataloader,desc="{} {}".format(msg,epoch)):
            imgs = list(img.to(device, dtype=torch.float) for img in images)
            # annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            output = model(imgs)

            for annot in annotations:
                annot['scores'] = torch.ones(annot['boxes'].size(0))
            # Process output from model
            df_batch_pred, n_objects = evaluate_batch(output,annotations)
            total_n_objects += n_objects
            df_validation_pred = df_validation_pred.append(
                    df_batch_pred, ignore_index=True, sort=False
                    ) 
            # Visualize for sample
            if ival<n_batches_visualize:
                for img, annot, out in zip(imgs,annotations,output):
                    resize_factor=2
                    img = img.detach().squeeze().cpu().numpy()
                    if len(img.shape)==3:
                        if img.shape[0]==1:
                            img = img[img.shape[0]//2]
                        elif img.shape[0]==3:
                            img = img.transpose(1,2,0)
                    if (img.max()>1) or (img.min()<0):
                        img = (img-img.min())/(img.max()-img.min())
                    img = (img*(2**8-1)).astype(np.uint8)
                    img = Image.fromarray(img).convert('RGB')
                    draw = ImageDraw.Draw(img)
                    for box, label in zip(annot['boxes'].numpy(),
                     annot['labels'].numpy()):
                        if label==1:
                            draw.rectangle(box, fill=None, outline='blue',width=5)
                        else:
                            draw.rectangle(box, fill=None, outline='green',width=5)
                    for box,score in zip(out['boxes'].detach().cpu().numpy(),
                                            out['scores'].detach().cpu().numpy()):
                        if score>0.1:
                            draw.rectangle(box, fill=None, outline=(int(score*255),0,0),width=5)
                    img_array = np.array(img.resize((img.width//resize_factor, img.height//resize_factor)))
                    img_msg = "epoch{}/{}".format(epoch,msg)
                    img_msg += "/{}/".format(dataloader.dataset.df.file_name.iloc[idx])
                    writer.add_image(img_msg,img_array,dataformats='HWC')

                    annot['file_name'] = dataloader.dataset.df.file_name.iloc[idx]

                    out = {k: v.detach().cpu().numpy() for k, v in out.items()} 

                    if path_to_output is not None:
                        out_path = os.path.join(path_to_output,annot['file_name']+'.pkl')
                        os.makedirs(os.path.dirname(out_path),exist_ok=True)
                        saving_dict = {'output':out,
                                        'annotations':annot}
                        pickle.dump(saving_dict, open(out_path,'wb'))
                    idx+=1

            ival+=1
        tpr, fps, thresholds = froc(df_validation_pred, total_n_objects, return_thresholds=True, max_fps=20) 
        epoch_tpr2 = np.interp(2.0, fps, tpr)
        epoch_tpr1 = np.interp(1.0, fps, tpr)
        epoch_tpr4 = np.interp(4.0, fps, tpr)
        mAP = (epoch_tpr1+epoch_tpr2+epoch_tpr4)/3
        epoch_tpr20 = np.interp(20.0, fps, tpr)
        # print(epoch_tpr2)
        thr = np.interp(2.0, fps, thresholds)
        # print('Threshold',thr)
        iou_mean=0
        if len(df_validation_pred[df_validation_pred.Score>=thr])!=0:
            iou_mean = df_validation_pred[df_validation_pred.Score>=thr].IoU.mean() 
        writer.add_scalar('TPR2/{}'.format(msg),epoch_tpr2,global_step=epoch)
        writer.add_scalar('TPR20/{}'.format(msg),epoch_tpr20,global_step=epoch)
        writer.add_scalar('mAP/{}'.format(msg),mAP,global_step=epoch)
        writer.add_scalar('IoU2/{}'.format(msg),iou_mean,global_step=epoch)
        return mAP


def resnet50_fpn_backbone_modified(trainable_layers=3,
    weights_path = None ,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    returned_layers=None,
    extra_blocks=None,
    num_classes=2,
    state_dict=True):

    
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    
    backbone = resnet.__dict__['resnet50'](
        pretrained=False,
        norm_layer=norm_layer,
        num_classes=num_classes)

    
    if weights_path is not None:
        backbone_state_dict = torch.load(weights_path)
        if not state_dict:
            backbone_state_dict = backbone_state_dict.state_dict()
        backbone.load_state_dict(backbone_state_dict)
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


# For changing the model's tranformer resize to higher
class ChangeModelOnPlateau(object):

    def __init__(self, model=None,mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        self.model = model
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._change_model(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

#         self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _change_model(self, epoch):
        if self.model is not None:
            # Change Model's transformer
            self.model.transform.max_size = self.model.transform.max_size+100
            self.model.transform.min_size = (self.model.transform.min_size[0]+100,)
            if self.verbose:
                print('Epoch {:5d}: changing model transformer to:'.format(epoch))
                print(self.model.transform)                

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -math.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


