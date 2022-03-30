import argparse
from model_class import FasterRCNN
import warnings
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
    arg_parser.add_argument('-experiment_name',
                            metavar='experiment_name',
                            type=str,
                            help='Experiment Name for tensorflow')
    arg_parser.add_argument('-path_to_output',
                            metavar='path_to_output',
                            type=str,
                            default=None,
                            help='Directory for output')

    args=arg_parser.parse_args()
    weights_path = args.weights_path
    img_dir = args.img_dir
    annot_dir = args.annot_dir
    csv = args.csv
    experiment_name = args.experiment_name
    path_to_output=args.path_to_output

    
    model_obj = FasterRCNN(model_path=weights_path, num_classes=2)
    print('yes')
    # TO START TRAINING
    model_obj.evaluate_model(
        img_dir, 
        annot_dir,
        csv,
        model_path=weights_path,
        experiment_name=experiment_name,
        path_to_output=path_to_output)
