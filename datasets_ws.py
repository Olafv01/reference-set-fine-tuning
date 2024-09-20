
import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as T
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader

print("own database ws")
base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
#     # print(batch[0][1].shape)
#     for e in batch:
#         print(e)
#         print(e[0])
#         print(e[1])
        
#         print(e[1].shape,'and 2',e[2].shape)
#         break
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    # triplets_local_indexes = torch.tensor([torch.tensor(e[1]) for e in batch])
    # triplets_global_indexes = torch.tensor([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path):
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
    
    def __getitem__(self, index):
        return base_transform(path_to_pil_img(self.images_paths[index]))
    
    def __len__(self):
        return len(self.images_paths)


class BaseDataset_normal(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, split)
        if not os.path.exists(self.dataset_folder):
            self.dataset_folder=join(datasets_folder, dataset_name,"images",split)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        
        #the msls dataset contains more query images than the msls val test split used for most papers
        if dataset_name="msls":
            val_queries=[]
            with open("/scratch/mzaffar/olaf/MSLS-val/key_mslsval.txt") as f:
                val_queries = list(f)[0]
            all_queries_paths = sorted(glob(join(queries_folder, "**", "*.jpg"),  recursive=True))
            self.queries_paths=[]
            

            for query_path in all_queries_paths:
                pano_id= query_path.split('@')[7]
            #     print(pano_id)
                if pano_id in val_queries:
                    self.queries_paths.append(query_path)
                    
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=args.positive_dist_threshold,
                                                             return_distance=False)
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)

#         if len(self.queries_paths)<args.cache_refresh_rate:
#             args.cache_refresh_rate=len(self.queries_paths)
#             args.queries_per_epoch=5*len(self.queries_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
        args.database_num=self.database_num
        args.queries_num=self.queries_num
    
    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        # With database images self.test_method should always be "hard_resize"
        img = T.functional.resize(img, self.resize)
        return img, index
        
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query

