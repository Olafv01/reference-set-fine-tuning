import sys

from SUE_ensamble import test,datasets_ws
from SUE_ensamble.parser import parse_arguments
from SUE_ensamble.utils import combining_methods, boq_output_only_model
sys.path.append("SelaVPR")

from datetime import datetime
import math
import time
import copy
import numpy as np
import pandas as pd
import json
import uuid
import os

import torch
from glob import glob
from tqdm import tqdm
import logging
from os.path import join, exists
from IPython.display import clear_output
import SelaVPR.test as sela_test
from SelaVPR.local_matching import local_sim
from loss import LocalFeatureLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
import torch.nn as nn
import faiss 
import kornia

args=parse_arguments()


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
    
args.is_trainref = False

np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.ckpnt_path=None

args.train_backbone="original with model.eval"
args.train_aggregator= True

if args.method=="sela":
    import vpr_models.SelaVPR
    from vpr_models import get_model
    args.dense_feature_map_size=[61,61,128]
    args.image_size=[224,224]
    args.features_dim=1024
    args.foundation_model_path=None
    model=network.GeoLocalizationNet(args)
    model = model.to(args.device)
    
    rerank , model= vpr_models.SelaVPR.get_model(args)
    model.cuda()

    new_state=torch.load(args.resume)
    if new_state!=None:
        state_dict=new_state
        if "model_state_dict" in state_dict.keys():
                    state_dict=state_dict["model_state_dict"]

        model.load_state_dict(state_dict)
        logging.info("model on {} loaded!".format(args.resume))
    else:
        logging.info("No new state defined")
        
elif args.method=="boq":
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
#     for param in model.backbone.named_parameters():
#         if param[1].requires_grad:
#             param[1].requires_grad=False
    args.features_dim=args.descriptors_dimension
    
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
if args.image_size!=None:
    args.resize=args.image_size
   


    
if 'finetuned' in args.resume:
    og_resume=args.resume
    add_epochs=args.epochs_num
    filename=args.resume.split('/')[-1]
    save_dir=args.resume[:-len(filename)]
    load_state=torch.load(args.resume)
    resume_log="logfile"+filename
    model.load_state_dict(load_state["model_state_dict"])
    logging.info(f"loaded model in {args.resume} into the model")
    resume_logfile=torch.load(join(save_dir,resume_log))
    losses=resume_logfile["losses"]
    all_recalls=resume_logfile["recalls"]
    test_recalls=resume_logfile["test_set_recalls"]
    best_epoch_nums=resume_logfile["best_epoch"]
    args=resume_logfile["args"]
    args.resume= og_resume
    args.epochs_num+=add_epochs
    start_epoch_num=resume_logfile["epoch_num"]
    times_per_epoch=resume_logfile["times_per_epoch"]
    logging.info(f"the results will be save to {save_dir}/logfile{filename}")
    logging.info(f"Will continue training with these arguments")
    print(args)
    best_recalls=np.array(resume_logfile["recalls"])[:,1]
    best_epoch_num=best_epoch_nums[-1]
    best_r5 = max(best_recalls)
    if "times_per_loop" in resume_logfile.keys():
        times_per_loop = resume_logfile["times_per_loop"]
    
database_folder= join("datasets_vg/datasets",args.dataset_name,"test","database")
if not os.path.exists(database_folder):
    database_folder= join("datasets_vg/datasets",args.dataset_name,"images","test","database")

listdata = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))



    
if args.dataset_name=="nordland":
    train_mask=np.ones(len(listdata),dtype=bool)
    train_mask[:int(0.3*(len(train_mask)))]=False
    np.random.shuffle(train_mask)
    
    queries_folder= join("datasets_vg/datasets",args.dataset_name,"test","queries")
    if not os.path.exists(queries_folder):
        queries_folder= join("datasets_vg/datasets",args.dataset_name,"images","test","queries")
    
    listdata = sorted(glob(join(queries_folder, "**", "*.jpg"), recursive=True))
    mask=np.zeros(len(listdata),dtype=bool)
    mask[int(0.9*(len(mask))):]=True
#     np.random.shuffle(mask)
    val_ds=datasets_ws.RefDataset(args,datasets_folder="datasets_vg/datasets", dataset_name=args.dataset_name, split='test', indices=mask, val=True)
    test_ds=datasets_ws.RefDataset(args,datasets_folder="datasets_vg/datasets", dataset_name=args.dataset_name, split='test', indices=~mask, val=True)
#     test_ds=datasets_ws.BaseDataset(args,datasets_folder="datasets_vg/datasets",dataset_name=args.dataset_name,split='test') 
    
    
    
else:
    train_mask=np.ones(len(listdata),dtype=bool)
    train_mask[:int(0.3*(len(train_mask)))]=False
    np.random.shuffle(train_mask)
    val_mask=~train_mask
    val_ds=datasets_ws.RefDataset(args,datasets_folder="datasets_vg/datasets", dataset_name=args.dataset_name, split='test', indices=val_mask, val=True)
    test_ds=datasets_ws.BaseDataset(args,datasets_folder="datasets_vg/datasets",dataset_name=args.dataset_name,split='test') 




if args.method=="sela":
    logging.info("creating triplets for sela")
    triplets_ds=datasets_ws.TripletsDataset_rerank(args, datasets_folder="datasets_vg/datasets", dataset_name=args.dataset_name, split="test", negs_num_per_query=args.negs_num_per_query,ref_query_split=0.01,indices=train_mask)
else:
    logging.info("creating triplets for boq or crica")
    triplets_ds=datasets_ws.TripletsDataset(args, datasets_folder="datasets_vg/datasets", dataset_name=args.dataset_name, split="test", negs_num_per_query=args.negs_num_per_query,ref_query_split=0.01,indices=train_mask)
    
model = model.cuda()
model = model.train()

args.queries_per_epoch=triplets_ds.queries_num if args.queries_per_epoch == -1 else args.queries_per_epoch
args.cache_refresh_rate=args.queries_per_epoch
args.log_frequency= 1008 if args.queries_per_epoch >1008 else args.queries_per_epoch
assert args.log_frequency / args.train_batch_size == int(args.log_frequency / args.train_batch_size) or args.log_frequency==args.queries_per_epoch, f"log_frequency ({args.log_frequency}) needs to be a multiple of the batch size({args.train_batch_size})"
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
GlobalTriplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

if args.method == "sela":
    MNNLocalFeatureLoss = LocalFeatureLoss().to(args.device)

if args.method=="boq":
    model=boq_output_only_model(model)

# args.epochs_num=50

not_improved_num = 0

if args.create_augments:
    os.makedirs(join(args.val_save_dir,args.dataset_name),exist_ok=True)
    queries_subset_ds = Subset(val_ds, list(range(val_ds.database_num, val_ds.database_num+val_ds.queries_num)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                    batch_size=1, pin_memory=(args.device == "cuda"))

    for image,index in tqdm(queries_dataloader):
        og_im_path= val_ds.images_paths[index]
        im_name = og_im_path.split("/")[-1]

        new_im_path= join(args.val_save_dir,args.dataset_name,im_name)
        
        if exists(new_im_path):
            continue
           
        image=combining_methods(image)[0]
        
        image=transforms.functional.to_pil_image(image)
        image.save(new_im_path)
    
    
if 'finetuned' not in args.resume:
    save_dir=join(args.log_dir,args.resume.split(".")[0].split('/')[-1])
    filename="{}_finetuned_on_{}_{}_.pth".format(args.resume.split(".")[0].split('/')[-1],args.dataset_name.split("/")[0],str(args.lr))

    os.makedirs(save_dir,exist_ok=True)
    while os.path.exists(join(save_dir,"logfile"+filename)):
         logging.info(f"file {filename} already exists add something to it")
         filename=filename[:-4]
         try:
             int(filename.split("_")[-1])
         except:
             filename=filename+"0"+".pth"
         else:
             try_number=int(filename.split("_")[-1])+1
             filename=filename[:-len(filename.split("_")[-1])]
             filename=filename+str(try_number)+".pth"

                
    logging.info(f"the results will be save to {save_dir}/logfile{filename}")
    
    if args.method=="sela":
        recalls, recalls_str = sela_test.vervang_vooreigen(args, val_ds, model)
        logging.info(recalls)
        test_1, _ = sela_test.vervang_vooreigen(args, test_ds, model)
    else:
       recalls, recalls_str = test.val(args, val_ds, model)
       logging.info(recalls)
        
       test_1,test_str = test.test(args, test_ds, model)

    best_r5=recalls[1]
    previous_best=best_r5
    # best_modelstate=model.state_dict()
    
    logging.info(test_1)
    args.early_stopping=False
    args.early_stopping_epoch=-1
    print(args)
    losses=[]
    all_recalls=[]
    all_recalls.append(recalls)
    test_recalls=[]
    best_epoch_nums=[]
    best_epoch_log_nums=[]
    
    times_per_epoch=[]
    times_per_loop=[]
    test_recalls.append(test_1)
    best_epoch_num=0
    best_epoch_log_num=0
    start_epoch_num=0
                 

model = model.eval()
triplets_ds.is_inference=True
logging.info('computing triplets')
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
logging.info("created triplets with {} mining".format(args.mining))



# og_params=copy.deepcopy(list(model.named_parameters()))
# og_dict = copy.deepcopy(model.state_dict())
# current_params=copy.deepcopy(list(model.named_parameters()))

# for i,param in enumerate(og_params):
#     assert torch.equal(param[1],current_params[i][1]) , f"wrong param {param[0]}"

# current_model=copy.deepcopy(model.state_dict())
# for key in og_dict.keys():
#     if not torch.equal(og_dict[key],current_model[key]):
#         print(key)
logging.info(f"Start finetuning")     
for epoch_num in range(start_epoch_num, args.epochs_num):
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    model = model.train()
    
        # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = 1
    loop_start_time= datetime.now()
    # Compute triplets to use in the triplet loss
    model = model.eval()
    # images shape: (train_batch_size*4)*3*H*W
    for i, (images, triplets_local_indexes, _) in enumerate(triplets_dl):
      


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
        if (i % (args.log_frequency/args.train_batch_size)==0 and i!=0) or (args.queries_per_epoch==args.log_frequency and int(i)==int(args.queries_per_epoch/args.train_batch_size)-1):

            losses.append(epoch_losses)   
            logging.info(f"Epoch[{epoch_num:02d}]({(i+1)*args.train_batch_size}/{triplets_ds.queries_num}): " +
                          f"current batch triplet loss = {batch_loss:.4f}, " +
                          f"average epoch triplet loss = {epoch_losses.mean():.5f}")
            args.is_trainref=True
            model = model.eval()

            if args.method=="sela":
                recalls, recalls_str = sela_test.vervang_vooreigen(args, val_ds, model)
            else:
                recalls, recalls_str = test.val(args,val_ds, model)
            
            logging.info(f"Final recalls on val set {val_ds}, with reranknum {args.rerank_num}: {recalls_str}")
            prev_best=best_r5
            
            is_best=recalls[1]>best_r5
            if not args.early_stopping:
                if (is_best or epoch_num==args.epochs_num-1):
                    logging.info(f"Best model saved, will now test on test set for validation")
                    best_r5=recalls[1]
                    best_epoch_num=epoch_num
                    best_epoch_log_num= i * args.train_batch_size/args.log_frequency
                    
                    not_improved_num = 0
                    
                else:
                    if epoch_num>args.early_stopping_epoch:
                        args.early_stopping_epoch=epoch_num
                        not_improved_num+=1

            logging.info(f"Finetuned:  best R@5 = {prev_best:.1f}, current R@5 = {(recalls[1]):.1f}, not improved for {not_improved_num}")
            
            if args.method=="sela":
                test_best, test_str = sela_test.vervang_vooreigen(args, test_ds, model)
            else:
               test_best,test_str = test.test(args,test_ds, model)

            all_recalls.append(recalls)

            test_recalls.append(test_best)

            best_epoch_nums.append(best_epoch_num)
            best_epoch_log_nums.append(best_epoch_log_num)

            logging.info(f"Final recalls on test set {test_ds}, with reranknum {args.rerank_num}: {test_str}")

            times_per_epoch.append(datetime.now() - epoch_start_time)
            times_per_loop.append(datetime.now() - loop_start_time)

            logging.info(f"Finished loop {epoch_num:02d} in {str(datetime.now() - loop_start_time)[:-7]}, "
                         f"average epoch triplet loss = {epoch_losses.mean():.4f}")

            args.save_dir=save_dir
            loop_start_time= datetime.now()
            log_path = join(save_dir, "logfile"+filename)
            state_log={"epoch_num":epoch_num, "losses":losses,"lr":args.lr, "args":args,"recalls":all_recalls,"test_set_recalls":test_recalls,"best_epoch":best_epoch_nums, "times_per_epoch":times_per_epoch, "times_per_loop":times_per_loop}

            try:
                torch.save(state_log, log_path)
            except:
                print(f'error during saving {log_path}, will try again next loop')
            

            model_path=join(save_dir, filename)
            model_state={"epoch_num": epoch_num,"best_epoch":best_epoch_num, "model_state_dict": model.state_dict(),  "all_losses":losses, "recalls":all_recalls, "test_set_recalls":test_recalls, "best_epoch":best_epoch_nums, "times_per_epoch":times_per_epoch, "times_per_loop":times_per_loop}

            try:
                torch.save(model_state, model_path)
            except:
                print(f'error during saving {model_path}, will try again next loop')
            
            best_model_path = join(save_dir, "best_model_"+filename)
            if is_best and not args.early_stopping:
                try:
                    torch.save(model_state, best_model_path)
                except:
                    print(f'error during saving {model_path}, will try again next loop')
                
                
                

            if not_improved_num>args.patience:
                args.early_stopping=True


        
#         else:
#             logging.info(f"i {i}, queries_per_epoch,{args.queries_per_epoch}, other {int(args.queries_per_epoch/args.train_batch_size)-1}")
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")

    
    
print(args)
logging.info(f"Finished fine-tuning with final recalls: {test_str}")