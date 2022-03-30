import argparse
from model_class import FasterRCNN
import warnings
import os
warnings.filterwarnings("ignore")


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-weights_path',
                            metavar='weights_path',
                            type=str,
                            help='`pth` path of model weights')
    arg_parser.add_argument('-img_dir',
                            metavar='img_dir',
                            type=str,
                            help='Image directory')
    arg_parser.add_argument('-annot_dir',
                            metavar='annot_dir',
                            type=str,
                            help='Annotation Directory')
    arg_parser.add_argument('-csv',
                            metavar='csv',
                            type=str,
                            help='CSV')
    arg_parser.add_argument('-out_dir',
                            metavar='out_dir',
                            type=str,
                            help='Output Directory')


    args=arg_parser.parse_args()
    weights_path = args.weights_path
    img_dir = args.img_dir
    annot_dir = args.annot_dir
    csv = args.csv
    out_dir = args.out_dir

    os.makedirs(out_dir,exist_ok=True)

    
    model_obj = FasterRCNN(model_path=weights_path, num_classes=2)
    # TO START TRAINING
    model_obj.infer(
        img_dir, 
        annot_dir,
        csv,
        out_dir,
        model_path=weights_path,
        batch_size=4
        )
