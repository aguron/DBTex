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
    arg_parser.add_argument('-train_csv',
                            metavar='train_csv',
                            type=str,
                            help='Train CSV')
    arg_parser.add_argument('-val_csv',
                            metavar='val_csv',
                            type=str,
                            help='Validation CSV')
    arg_parser.add_argument('-experiment_name',
                            metavar='experiment_name',
                            type=str,
                            help='Experiment Name for tensorflow')
    arg_parser.add_argument('-batch_size',
                            metavar='batch_size',
                            type=int,
                            help='Batch_size')
    arg_parser.add_argument('-lr_rate',
                            metavar='lr_rate',
                            type=float,
                            help='Learning rate')
    arg_parser.add_argument('-weight_decay',
                            metavar='weight_decay',
                            type=float,
                            help='Learning rate')
    arg_parser.add_argument('-num_epochs',
                            metavar='num_epochs',
                            type=int,
                            default=100,
                            help='number of epochs')

    args=arg_parser.parse_args()
    weights_path = args.weights_path
    img_dir = args.img_dir
    annot_dir = args.annot_dir
    train_csv = args.train_csv
    val_csv = args.val_csv
    experiment_name = args.experiment_name
    batch_size=args.batch_size
    lr_rate = args.lr_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs

    print(args)
    
    model_obj = FasterRCNN(num_classes=2,#weights_path='./checkpoints/Tomo_FastRCNN_v27.pth',
                backbone_weights_path='./checkpoints/DDSM_INbreast_SPIE_augmented.pth',
                 state_dict=True)
    # TO START TRAINING
    model_obj.train_model(
        img_dir, 
        annot_dir,
        train_csv,
        val_csv, 
        lr_rate=lr_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        model_path=weights_path,
        batch_size=batch_size,
        experiment_name=experiment_name)