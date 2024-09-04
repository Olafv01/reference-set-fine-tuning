import argparse
import os
import torch


def parse_arguments(arguments=[]):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    parser.add_argument("--datasets_folder", type=str, required=True,
                        help="path/to/datsets")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="...")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="_")
    parser.add_argument("--rerank_num", type=int, default=100,
                        help="amount of database images to rerank default 100.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="set to 1 if database images may have different resolution")
    parser.add_argument("--n_features", type=int, default=None,
                        help="amount of features to save for each query for each method, default uses all features")
    parser.add_argument("--fuse_method", type=str, default="avg",
                        help="Select which method to use for combining the different models, avg for average of all L2 distances, SUE-avg for SUE weighted L2 distances, SUE-max for most confident method only", choices=["avg","SUE-avg","SUE-max"])
    parser.add_argument("--log_dir", type=str, default="default",
                        help="experiment name, output logs will be saved under logs/log_dir")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="_")
    parser.add_argument("--method_pths", type=str, nargs="+", default=None, help="use this if finetuned models should be used")
    
    parser.add_argument("--methods", type=str, nargs="+", default=None, help="use this if youre lazy and what one finetuned model of each method", choices=["boq", "sela", "crica"])
    
    parser.add_argument("--method_folder", type=str, default="weights", required=True,
                        help="Place where the state dicts are stored")
    parser.add_argument("--gpu_id",type=str, default="0", help="Cuda visible devices")
    parser.add_argument("--pca", type=bool, default=False, help="True if you want to use smaller descriptor dimensions")
    parser.add_argument("--pca_folder",type=str, default=None, help="If you have saved pca's for the methods enter the folder where these are saved.")
    
    
    
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance (in meters) for a prediction to be considered a positive")
    
    parser.add_argument("--backbone", type=str, default=None,
                        choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
                        help="_")
    parser.add_argument("--registers", type=bool, default=False,
                        help="_")
    
    parser.add_argument("--descriptors_dimension", type=int, default=None,
                        help="_")
    
    parser.add_argument('--dense_feature_map_size', type=int, default=[61,61,128], nargs=3, 
                        help="size of dense feature map of the selaVPR method (a 61x61 grid 128-dim local features)")
    
    parser.add_argument("--recall_values", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="values for recall (e.g. recall@1, recall@5)")
    parser.add_argument("--no_labels", action="store_true",
                        help="set to true if you have no labels and just want to "
                        "do standard image retrieval given two folders of queries and DB")
    parser.add_argument("--image_size", type=int, default=None, nargs="+",
                        help="Resizing shape for images (HxW). If a single int is passed, set the"
                        "smallest edge of all images to this value, while keeping aspect ratio")
    
    
    if len(arguments)>0:
        args = parser.parse_args(args=arguments)
    else:
        args= parser.parse_args()
        
        
    return args
    
