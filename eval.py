import sys

import test,datasets_ws
from parser import parse_arguments
from util import combining_methods, boq_output_only_model


import numpy as np

import torch
from glob import glob
from tqdm import tqdm
import logging
import os
from os.path import join, exists
from IPython.display import clear_output
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
import torch.nn as nn
import faiss 
import kornia

args=parse_arguments()


np.random.seed(args.seed)
torch.manual_seed(args.seed)

console="debug"
info_filename="info.log"
debug_filename="debug.log"
base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

if info_filename != None:
    info_file_handler = logging.FileHandler(join('.', info_filename))
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(base_formatter)
    logger.addHandler(info_file_handler)

if debug_filename != None:
    debug_file_handler = logging.FileHandler(join('.', debug_filename))
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(base_formatter)
    logger.addHandler(debug_file_handler)

if console != None:
    console_handler = logging.StreamHandler()
    if console == "debug": console_handler.setLevel(logging.DEBUG)
    if console == "info":  console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(base_formatter)
    logger.addHandler(console_handler)
    
if args.gpu_id !=None:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id )
    logging.info(f"using GPU with id {args.gpu_id}")
else:
    logging.info(f"no gpu defined, using the first available gpu")
    

if args.method=="boq":
    if args.backbone not in [None, "DINOv2","ResNet50"]:
        raise ValueError("When using BOQ the backbone must be None or resnet50 or Dinov2")
    if args.backbone is None:
        args.backbone="DINOV2"
    if args.backbone.lower() == "dinov2":
        args.descriptors_dimension=12288
        args.image_size=[322,322]
    elif args.backbone.lower() == "resnet50":
        args.descriptors_dimension=16384
        args.image_size=[384,384]
        
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq", backbone_name=args.backbone.lower(), output_dim=args.descriptors_dimension)
    
    args.features_dim=args.descriptors_dimension
    model=boq_output_only_model(model)
    
elif args.method == "crica":
    args.image_size=[224,224]
    args.descriptors_dimension = 10752
    model=torch.hub.load("Lu-Feng/CricaVPR", "trained_model")
    args.features_dim=args.descriptors_dimension
    
    for name, param in model.module.backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
            
elif args.method == "salad":
    args.image_size=[322,322]
    args.descriptors_dimension=8448
    args.features_dim=args.descriptors_dimension
    model = torch.hub.load("serizba/salad", "dinov2_salad")
    for blk in model.backbone.model.blocks[:-model.backbone.num_trainable_blocks]:
    #     print(blk)
        for param in blk.parameters():
            param.requires_grad=False
            
if args.resume!=None:
    state_dict=torch.load(args.resume)["model_state_dict"]
    model.load_state_dict(state_dict)
model.cuda()
model.eval()
test_ds=datasets_ws.BaseDataset(args,datasets_folder=args.datasets_folder,dataset_name=args.dataset_name,split='test') 
test_1,test_str = test.test(args, test_ds, model)
logging.info(f"Evaluated model: {args.method} on dataset {args.dataset_name}: {test_str}")