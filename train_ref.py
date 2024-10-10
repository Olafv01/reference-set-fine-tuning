from dataloader import *

from SelaVPR import datasets_ws
import sys
from SUE_ensamble import test
from SUE_ensamble.parser import parse_arguments
from SUE_ensamble.utils import combining_methods
sys.path.append("SelaVPR")
import SelaVPR.test as sela_test
from SelaVPR import util
import vg_parser
from datetime import datetime
import torch.nn as nn
import math
import time
import copy
import numpy as np
import pandas as pd
import json
import uuid
import torch
from tqdm import tqdm
import faiss 
import logging
from os.path import join
from IPython.display import clear_output
from SelaVPR.local_matching import local_sim
from loss import LocalFeatureLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
import kornia

args=parse_arguments()
# args.features_dim=8448

if args.gpu_id !=None:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id )
    

args.is_trainref = False
np.random.seed(0)
args.ckpnt_path=None

args.train_backbone="original with model.eval"
args.train_aggregator= True
import sys
sys.path.insert(1,"/scratch/mzaffar/olaf/VPR-methods-evaluation//")


# import parser
    # print(args)
if args.method=="sela":
    import vpr_models.SelaVPR
    from vpr_models import get_model
    args.dense_feature_map_size=[61,61,128]
    args.image_size=[224,224]
    args.features_dim=1024
    args.foundation_model_path=None
    rerank , model= vpr_models.SelaVPR.get_model(args)
    model.cuda()

    new_state=torch.load(args.resume)
    if new_state!=None:
        state_dict=new_state
        if "model_state_dict" in state_dict.keys():
                    state_dict=state_dict["model_state_dict"]

        #state_dict = torch.load(args.resume)["model_state_dict"]   
#        keys= state_dict.copy().keys()
 #       for key in keys:
  #          if key[:7]=="module.":
   #             state_dict[key[7:]]=state_dict[key]
    #            del state_dict[key]

        model.load_state_dict(state_dict)
        print("model on {} loaded!".format(args.resume))
    else:
        print("No new state defined")
elif args.method=="boq":
    
    from vpr_models import get_model
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
    model= get_model(args.method, args.backbone, args.descriptors_dimension)
    args.features_dim=args.descriptors_dimension
#     if not args.train_backbone:
#         model.backbone.num_unfrozen_blocks=0
#         for i in range(len(model.backbone.dino.blocks) - model.backbone.num_unfrozen_blocks):
#                     model.backbone.dino.blocks[i].requires_grad_(False)
#     else:
#         model.backbone.num_unfrozen_blocks= len(model.backbone.dino.blocks)
    
#     for name, param in model.aggregator.named_parameters():
#         param.requires_grad= args.train_aggregator# agregator should be trained (should be True)
elif args.method == "crica":
    args.image_size=[224,224]
    args.descriptors_dimension = 10752
    model=torch.hub.load("Lu-Feng/CricaVPR", "trained_model")
    
    args.features_dim=args.descriptors_dimension

if args.image_size!=None:
    args.resize=args.image_size
        
database_folder= join("datasets_vg/datasets",args.dataset_name,"test","database")
if not os.path.exists(database_folder):
    database_folder= join("datasets_vg/datasets",args.dataset_name,"images","test","database")

listdata = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))

mask=np.zeros(len(listdata),dtype=bool)

mask[:int(0.3*(len(mask)))]=True
np.random.shuffle(mask)
val_mask=mask
# print(len(val_mask),val_mask.sum())
train_mask=~mask
# print(len(train_mask),train_mask.sum())

# if args.dataset_name=="amstertime":
#     val_ds=datasets_ws.BaseDataset(args,datasets_folder="datasets_vg/datasets",dataset_name=args.dataset_name,split='test')
# else:
# val_ds=datasets_ws.BaseDataset(args,datasets_folder="/scratch/mzaffar/fabian",dataset_name=args.dataset_name,split='train')  
val_ds=RefDataset(args,datasets_folder="datasets_vg/datasets",dataset_name=args.dataset_name,split='test',indices=val_mask,val=True)
test_ds=datasets_ws.BaseDataset(args,datasets_folder="datasets_vg/datasets",dataset_name=args.dataset_name,split='test')  
if args.method=="sela":
    print("creating triplets for sela")
    triplets_ds=TripletsDataset_rerank(args, datasets_folder="datasets_vg/datasets", dataset_name=args.dataset_name, split="test", negs_num_per_query=args.negs_num_per_query,ref_query_split=0.01,indices=train_mask)
else:
    print("creating triplets for boq or crica")
    triplets_ds=TripletsDataset(args, datasets_folder="datasets_vg/datasets", dataset_name=args.dataset_name, split="test", negs_num_per_query=args.negs_num_per_query,ref_query_split=0.01,indices=train_mask)
    
model = model.cuda()
model = model.train()

args.queries_per_epoch=triplets_ds.queries_num if args.queries_per_epoch == -1 else args.queries_per_epoch
args.cache_refresh_rate=args.queries_per_epoch

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
GlobalTriplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

if args.method == "sela":
    MNNLocalFeatureLoss = LocalFeatureLoss().to(args.device)

if args.method=="boq":
    model=boq_output_only_model(model)

# args.epochs_num=50

not_improved_num = 0
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

    
if 'finetuned' in args.resume:
    filename=args.resume.split('/')[-1]
    save_dir=args.resume[:-len(filename)]
    load_state=torch.load(args.resume)
    resume_log="logfile"+filename
    model.load_state_dict(load_state["model_state_dict"])
    print(f"loaded model in {args.resume} into the model")
    resume_logfile=torch.load(join(save_dir,resume_log))
    losses=resume_logfile["losses"]
    all_recalls=resume_logfile["recalls"]
    test_recalls=resume_logfile["test_set_recalls"]
    best_epoch_nums=resume_logfile["best_epoch"]
    args=resume_logfile["args"]
    start_epoch_num=resume_logfile["epoch_num"]
    times_per_epoch=resume_logfile["times_per_epoch"]
    print(f"the results will be save to {save_dir}/logfile{filename}")
    
else:
    save_dir=join(args.log_dir,args.resume.split(".")[0].split('/')[-1])
    filename="{}_finetuned_on_{}_{}_.pth".format(args.resume.split(".")[0].split('/')[-1],args.dataset_name.split("/")[0],str(args.lr))

    os.makedirs(save_dir,exist_ok=True)
    while os.path.exists(join(save_dir,"logfile"+filename)):
         print(f"file {filename} already exists add something to it")
         filename=filename[:-4]
         try:
             int(filename.split("_")[-1])
         except:
             filename=filename+"0"+".pth"
         else:
             try_number=int(filename.split("_")[-1])+1
             filename=filename[:-len(filename.split("_")[-1])]
             filename=filename+str(try_number)+".pth"

                
    print(f"the results will be save to {save_dir}/logfile{filename}")
    
    if args.method=="sela":
        recalls, recalls_str = sela_test.vervang_vooreigen(args, val_ds, model)
        test_1, _ = sela_test.vervang_vooreigen(args, test_ds, model)
    else:
       recalls, recalls_str = test.val(args, val_ds, model)

       test_1,test_str = test.test(args, test_ds, model)

    best_r5=recalls[1]
    # best_modelstate=model.state_dict()
    print(recalls, args, '\n',test_1)
    losses=[]
    all_recalls=[]
    all_recalls.append(recalls)
    test_recalls=[]
    best_epoch_nums=[]
    times_per_epoch=[]
    test_recalls.append(test_1)
    best_epoch_num=0
    start_epoch_num=0
                 

model = model.eval()
triplets_ds.is_inference=True
print('computing triplets')
triplets_ds.compute_triplets(args, model)
triplets_ds.is_inference=False
# if args.method=="boq":
#     model=model.model
triplets_dl=torch.utils.data.DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device == "cuda"),
                                 drop_last=True)
clear_output(wait=True)
print("created triplets with {} mining".format(args.mining))



# og_params=copy.deepcopy(list(model.named_parameters()))
# og_dict = copy.deepcopy(model.state_dict())
# current_params=copy.deepcopy(list(model.named_parameters()))

# for i,param in enumerate(og_params):
#     assert torch.equal(param[1],current_params[i][1]) , f"wrong param {param[0]}"

# current_model=copy.deepcopy(model.state_dict())
# for key in og_dict.keys():
#     if not torch.equal(og_dict[key],current_model[key]):
#         print(key)
         
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start finetuning")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    model = model.train()
    
        # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")
        
        # Compute triplets to use in the triplet loss
        model = model.eval()
        # images shape: (train_batch_size*4)*3*H*W
        for images, triplets_local_indexes, _ in triplets_dl:    
            # Flip all triplets or none
            if args.augments:
                images[0]=combining_methods(images[0])
            
            # Compute features of all images (images contains queries, positives and negatives)
            if args.method=="sela":
                local_features, global_features = model(images.to(args.device))
#             elif args.method=="boq":
#                 global_features,_= model(combining_methods(images.to(args.device)))
            else:
                global_features = model(images.to(args.device))
            total_loss = 0
            global_loss = 0
            local_loss = 0

            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
            for triplets in triplets_local_indexes:
                queries_indexes, positives_indexes, negatives_indexes = triplets.T

                global_loss += GlobalTriplet(global_features[queries_indexes],
                                                  global_features[positives_indexes],
                                                  global_features[negatives_indexes])
                global_loss /= (args.train_batch_size * args.negs_num_per_query)
                
                if args.method=="sela":
                    local_loss += MNNLocalFeatureLoss([local_features[queries_indexes],
                                                  local_features[positives_indexes],
                                                  local_features[negatives_indexes]])
                    local_loss /= (args.train_batch_size * args.negs_num_per_query)

                
                           
            total_loss = global_loss #+ local_loss 

            del global_features
            if args.method=="sela":
                del local_features
# #             try: del local_features 
# #             except: 1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_loss = total_loss.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del total_loss
        
#         logging.debug(f"global loss = {global_loss.item():.6f},  ")
#         if args.method == "sela":
#             logging.debug(f"local loss = {local_loss.item():.6f},  ")            
        losses.append(epoch_losses)   
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                      f"current batch triplet loss = {batch_loss:.4f}, " +
                      f"average epoch triplet loss = {epoch_losses.mean():.5f}")
    args.is_trainref=True
    model = model.eval()
        
    if args.method=="sela":
        recalls, recalls_str = sela_test.vervang_vooreigen(args, val_ds, model)
    else:
        recalls, recalls_str = test.val(args,val_ds, model)
    logging.info(f"Final recalls on val set {val_ds}, with reranknum {args.rerank_num}: {recalls_str}")
    
    
#     current_model=copy.deepcopy(model.state_dict())
#     for key in og_dict.keys():
#         if not torch.equal(og_dict[key],current_model[key]):
#             if key not in list(model.named_parameters()):
#                 print(key)
#                 model.state_dict()[key]=og_dict[key]
    
    
#     for i,param in enumerate(og_params):
#         assert torch.equal(param[1],current_params[i][1]) , f"wrong param {param[0]}"
    
          
    all_recalls.append(recalls)
    
    logging.info(f"Finetuned:  best R@5 = {best_r5:.1f}, current R@5 = {(recalls[1]):.1f}")
    
    if recalls[1]>best_r5 or epoch_num==args.epochs_num-1:
        
        logging.info(f"Best model saved, will now test on test set for validation")
#         best_modelstate=model.state_dict()
        best_r5=recalls[1]
        best_epoch_num=epoch_num
        
    if args.method=="sela":
        test_best, test_str = sela_test.vervang_vooreigen(args, test_ds, model)
    else:
       test_best,test_str = test.test(args,test_ds, model)
    
    test_recalls.append(test_best)
    best_epoch_nums.append(best_epoch_num)
    
    logging.info(f"Final recalls on test set {test_ds}, with reranknum {args.rerank_num}: {test_str}")
        
    times_per_epoch.append(datetime.now() - epoch_start_time)
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
   
    args.save_dir=save_dir
    util.save_checkpoint(args,{"epoch_num":epoch_num, "losses":losses,"lr":args.lr, "args":args,"recalls":all_recalls,"test_set_recalls":test_recalls,"best_epoch":best_epoch_nums, "times_per_epoch":times_per_epoch},False,filename="logfile"+filename)
    util.save_checkpoint(args, {"epoch_num": epoch_num,"best_epoch":best_epoch_num, "model_state_dict": model.state_dict(),  "all_losses":losses, "recalls":all_recalls}, False, filename=filename)

    not_improved_num = 0
    
    
print(args)