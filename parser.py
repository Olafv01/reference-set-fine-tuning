import argparse
import os
import torch


def parse_arguments(arguments=[]):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    parser.add_argument("--datasets_folder", type=str, default="../datasets_vg/datasets",
                        help="path/to/datsets")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="...")
    parser.add_argument("--method",type=str,default="crica",
                        help="name of the method to use",choices=["salad","boq","crica"])
    
    parser.add_argument("--original_training_data",type=str,default="gsv",
                        help="Name of the dataset on which the selected model was trained.")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="_")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="set to 1 if database images may have different resolution")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed to use")
    parser.add_argument("--infer_batch_size",type=int,default=16,
                        help="set to 1 if database images may have different resolution")
    parser.add_argument("--mining",default="full",choices=["full","partial","random"],
                        help="select triplet mining strategy")
    parser.add_argument("--lr",type=float, default=1e-7,
                        help="learning rate to fine-tune the model with")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',
                        choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd"])
    parser.add_argument("--negs_num_per_query",type=int, default=2,
                        help="Number of negives to use in use triplet")
    parser.add_argument("--epochs_num",type=int,default=100,help="_")
    parser.add_argument("--log_frequency",type=int,default=1008,
                        help="amount of triplets to pass before evaluating the model, ")
    
    parser.add_argument("--ablation", action="store_true",
                        help="set to true if you want to perform the nordland like experiments (10% of queries for validation for other datasets")
    parser.add_argument("--ablation_nordland", action="store_true",
                        help="set to true if you want to perform the nordland experiment with augmented references.")
#     parser.add_argument("--save_models",  type=bool, default=True,
#                         help="switch to false if you do not need the final and best trained models.")
    parser.add_argument('--save_models', action='store_true')
    parser.add_argument('--save_no_models', dest='save_models', action='store_false')
    parser.set_defaults(save_models=True)
    parser.add_argument("--scheduled_lr", action="store_true",
                        help="set to true if you want to use a simple decaying lr based on args.gamma_lr and args.step_lr.")
    parser.add_argument("--gamma_lr", type=float, default=0.5,
                        help="reduction variable for the lr scheduler.")
    parser.add_argument("--step_lr", type=int, default=10,
                        help="amount of logging steps to pass before reducing the lr with the scheduler.")
    
    parser.add_argument("--test_val_queries",type=float,default=0.1,
                        help="ONLY IF ABLATION OR NORDLAND, percentage of test queries to use for validation.")
    
    parser.add_argument("--grayscale", action="store_true",
                        help="set to true if you want to train and validate on grayscale images, test stays original.")
    parser.add_argument("--exact_match", action="store_true",
                        help="set to true if you want to train with the same image being used as the query and the positive within the triplets.")
    
    parser.add_argument("--resume",type=str,default=None,
                        help="Currently not used, neeeds to be defined to create sublog folder to store results")
    parser.add_argument("--queries_per_epoch",type=int,default=None,
                        help="How much queries to use per epoch, if set to -1, the whole reference database will be used each epoch")
    parser.add_argument("--patience",type=int,default=5,
                        help="Early stopping patience, stop after so much epochs without improvement of R@1 for the validation set (NOT IMPLEMENTED)")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--augments",type=bool,default=True, 
                        help="Set to True to use augmentation during training on queries, also on the validation set" )
    parser.add_argument("--create_augments",action="store_true", 
                        help="Set to True to create augmented images for validation and save them " )
    
    parser.add_argument("--use_val_augments",type=bool,default=False, 
                        help="Set to True to use augmented images save in val_save_dir, when you have created them" )
    
    parser.add_argument("--val_save_dir", type=str, default="/home/osverburg/validation",
                        help="place where to save augmented images.")
    
    parser.add_argument("--log_dir", type=str, default="../logs",
                        help="experiment name, output logs will be saved under logs/log_dir")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="_")
    parser.add_argument("--gpu_id",type=str, default="0", help="Edit which CUDA GPU is visible, only applicable if device == cuda")
    parser.add_argument("--pca", action="store_true", help="True if you want to use smaller descriptor dimensions")
    parser.add_argument("--pca_folder",type=str, default=None, help="If you have saved pca's for the methods enter the folder where these are saved.")
    
    
    parser.add_argument('--random_vals', action='store_true',
                        help="set to true, to use randomly chosen images from the validation set.")
    parser.add_argument('--no_random_vals', dest='random_vals', action='store_false')
    parser.set_defaults(random_vals=True)
    
    parser.add_argument("--stop_after_patience", action="store_true",
                        help="Set to true to stop training after the early stopping patience was reached.")
    
    parser.add_argument("--val_split", type=float, default=0.3,
                        help="Part of the database images to use as validation queries.")
    
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    
    parser.add_argument("--backbone", type=str, default=None,
                        choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
                        help="_")
    
    parser.add_argument("--descriptors_dimension", type=int, default=None,
                        help="_")
    
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="values for recall (e.g. recall@1, recall@5)")
    parser.add_argument("--no_labels", action="store_true",
                        help="set to true if you have no labels and just want to "
                        "do standard image retrieval given two folders of queries and DB")
    parser.add_argument("--resize", type=int, default=None, nargs="+",
                        help="Only touch if you really want to, if None the correct size for the model is selected")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    
    
    if len(arguments)>0:
        args = parser.parse_args(args=arguments)
    else:
        args= parser.parse_args()
        
        
    return args
    