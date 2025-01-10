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
from itertools import compress
import kornia

base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_transform_gray=T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels=3),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
])

val_augment_transform = T.Compose([
    T.ToTensor(),
])
val_augment_transform_gray = T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels=3),
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
    images                  = torch.cat([e[0] for e in batch])
    triplets_local_indexes  = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class RefDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self,args, datasets_folder="datasets", dataset_name="pitts30k", split="train",indices=np.array([None]),val=True,grayscale=False):
        super().__init__()
        self.val=val
        if val:
            self.use_val_augments=args.use_val_augments
        else:
            self.use_val_augments=False
            
        args.triplets=False
        self.grayscale=grayscale
        if self.grayscale:
            print("using gray images")
            self.base_transform=base_transform_gray
            self.val_augment_transform=val_augment_transform_gray
        else:
            
            self.base_transform=base_transform
            self.val_augment_transform=val_augment_transform
            
        
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, split)
        if not os.path.exists(self.dataset_folder):
            self.dataset_folder=join(datasets_folder, dataset_name,"images",split)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        self.resize = args.resize
                
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
           
        
        if args.create_augments:
            # if augmented validation images are created all images should be selected for both queries and references,
            # only the queries are used, but if the database_paths is not defined, it will give errors
            print("used for creating augmented images")
            paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            self.queries_paths= paths
            self.database_paths = paths
            
        elif val and args.use_val_augments and ((dataset_name !="nordland" and not args.ablation) or args.ablation_nordland):
            # Loading the saved augmented validation images.
            assert os.path.exists(join(args.val_save_dir,args.dataset_name)), f" no augmented versions found for this dataset"
            print(f"using val and indices for small datasets")
            query_indices=list(indices)
            
            query_paths= sorted(glob(join(args.val_save_dir,args.dataset_name, "**", "*.jpg"), recursive=True))
            print(f"{len(query_paths)} queries found augmented")
            self.queries_paths = list(compress( query_paths,query_indices))
            
            paths=sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            self.database_paths = paths
         
        elif val and indices.all()!=None and (dataset_name=="nordland" or args.ablation and not args.ablation_nordland):
            # to prove our fine-tuning, we use part of the test query set as validation, which is selected here
            print(f"Using part of the Nordland queries for evaluation to prove the fine tuning we designed")
            queries_folder= join(self.dataset_folder, "queries")
            if not os.path.exists(queries_folder):
                raise FileNotFoundError(f"Folder {queries_folder} does not exist")
            query_indices=list(indices)
            
            queries_paths=sorted(glob(join(queries_folder,"**","*.jpg"),recursive=True)) 
            self.queries_paths = list(compress( queries_paths,query_indices))
            
            paths=sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            self.database_paths =paths
            
        elif val and indices.all()!=None :
            # using random data augmentations during validation, NOT RECOMMENDED as it results in noisy validation recalls
            print(f"using part of database images as validation set")
            query_indices=list(indices)
            
            paths=sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            self.queries_paths = list(compress( paths,query_indices))
            self.database_paths = paths

        elif (not val) and indices.all()!=None:
            # This is used to select the images to be used to construct triplets, the indices used here is a inverse of the indices used for validation, so no overlap is present
            print('triplets, with other indices') 
            indices=list(indices)
            
            paths=sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            self.queries_paths = list(compress( paths,indices))
            self.database_paths = list(compress( paths,indices))
            
        else:
            # fail save, if the settings were not set correctly, both the queries and database images are just the database images of the dataset
            print('you did not do good')
            self.queries_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            
            
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                         radius=args.val_positive_dist_threshold,
                                                         return_distance=False)
        if val:
            num_softpositives=[]
            for softs in self.soft_positives_per_query:
                num_softpositives.append(len(softs))
                
               
                
            num_softpositives=np.array(num_softpositives)
            print(f"the queries had an average of {np.mean(num_softpositives)}, with an standard deviation of {np.std(num_softpositives)}.")
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
    
    def __getitem__(self, index):
                      
        img = path_to_pil_img(self.images_paths[index])
        if self.val and self.use_val_augments and index >=len(self.database_paths):
            img = self.val_augment_transform(img)
            img = T.functional.resize(img, self.resize,antialias=True)
            img.to(torch.float64)
        else:
            img = self.base_transform(img)
            #With database images self.test_method should always be "hard_resize"
            img = T.functional.resize(img, self.resize,antialias=True)
        
        return img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num};"# #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query


class TripletsDataset(RefDataset):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", negs_num_per_query=2,indices=np.array([None]),grayscale=False):
        super().__init__(args, datasets_folder, dataset_name, split,indices=indices,val=False,grayscale=grayscale)
        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num if self.database_num>args.neg_samples_num else self.database_num # Number of negatives to randomly sample
        self.negs_num_per_query = negs_num_per_query  # Number of negatives per query in each batch
        if self.mining == "full":  # "Full database mining" keeps a cache with last used negatives
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)]
        self.is_inference = False
        
        identity_transform = T.Lambda(lambda x: x)
        self.resized_transform = T.Compose([
            T.Resize(self.resize,antialias=True) if self.resize is not None else identity_transform,
            self.base_transform
        ])
        
        self.query_transform = T.Compose([
                self.resized_transform,
        ])
        
        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10/25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                             radius=args.train_positives_dist_threshold,  # 10/25 meters
                                             return_distance=False))
        num_hardpositives=[]
        
        for i in range(len(self.hard_positives_per_query)):
            num_hardpositives.append(len(self.hard_positives_per_query[i]))
            if len(self.hard_positives_per_query[i])>1:
                if args.exact_match:
                    self.hard_positives_per_query[i]= self.hard_positives_per_query[i][np.where(self.hard_positives_per_query[i] == i)]
                else:
                    self.hard_positives_per_query[i]= np.delete( self.hard_positives_per_query[i], np.where(self.hard_positives_per_query[i] == i))
               
            
        
        #### Some queries might have no positive, we should remove those queries.
        
        num_hardpositives=np.array(num_hardpositives)
        
        self.num_hardpositives=num_hardpositives
        self.avg_hardpositives=np.mean(num_hardpositives)
        self.std_hardpositives=np.std(num_hardpositives)
        print(f"the positives had an average of {np.mean(num_hardpositives)}, with an standard deviation of {np.std(num_hardpositives)}.")
        queries_without_any_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
            print(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
            
        # Remove queries without positives
            self.hard_positives_per_query = np.delete(self.hard_positives_per_query, queries_without_any_hard_positive)
            self.queries_paths = np.delete(self.queries_paths, queries_without_any_hard_positive)
        
        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_num = len(self.queries_paths)
        
        
    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)
        query_index, best_positive_index, neg_indexes = torch.split(self.triplets_global_indexes[index], (1, 1, self.negs_num_per_query))
        query = self.query_transform(path_to_pil_img(self.queries_paths[query_index]))
        
        positive = self.resized_transform(path_to_pil_img(self.database_paths[best_positive_index]))
        negatives = [self.resized_transform(path_to_pil_img(self.database_paths[i])) for i in neg_indexes]
        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        return images, triplets_local_indexes, self.triplets_global_indexes[index]
    
    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)
    
    def compute_triplets(self, args, model):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model)
        elif self.mining == "partial":
            self.compute_triplets_partial(args, model)
        elif self.mining == "random":
            self.compute_triplets_random(args, model)
    
    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
                               batch_size=args.infer_batch_size, shuffle=False,
                               pin_memory=(args.device == "cuda"))
        
        model = model.eval()
        
        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        with torch.no_grad():
            for images, indexes in tqdm(subset_dl, ncols=100):
                images = images.to(args.device)
                features = model(images)
                cache[indexes.numpy()] = features.cpu().numpy()
        return cache
    
    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(f"For query {self.queries_paths[query_index]} " +
                               f"with index {query_index} features have not been computed!\n" +
                               "There might be some bug with caching")
        return query_features
    
    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item()
        return best_positive_index
    
    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.negs_num_per_query)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes
    
    def compute_triplets_random(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]  # Flatten list of lists to a list
        positives_indexes = list(np.unique(positives_indexes))
        
        # Compute the cache only for queries and their positives, in order to find the best positive
        subset_ds = Subset(self, positives_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.random.choice(self.database_num, size=self.negs_num_per_query+len(soft_positives), replace=False)
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[:self.negs_num_per_query]
            
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    
    def compute_triplets_full(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all database indexes
        database_indexes = list(range(self.database_num))
        #  Compute features for all images and store them in cache
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            # Choose 1000 random database images (neg_indexes)
            neg_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
            # Remove the eventual soft_positives from neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
            # Concatenate neg_indexes with the previous top 10 negatives (neg_cache)
            neg_indexes = np.unique(np.concatenate([self.neg_cache[query_index], neg_indexes]))
            # Search the hardest negatives
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            # Update nearest negatives in neg_cache
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    
    def compute_triplets_partial(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        
        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))
        
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, cache_shape=(len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            
            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)

        
class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train",svox_type=None):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, split)
        if not os.path.exists(self.dataset_folder):
            self.dataset_folder=join(datasets_folder, dataset_name,"images",split)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        self.resize = args.resize
        self.test_method = args.test_method
        
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        if svox_type!=None:
            queries_folder=join(self.dataset_folder, svox_type)
        else:
            queries_folder  = join(self.dataset_folder, "queries")
        if not os.path.exists(database_folder): raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder) : raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
#         self.queries_paths  = sorted(glob(join(queries_folder, "**", "*.jpg"),  recursive=True))
        
        if dataset_name=="msls_val":
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
        else:
            self.queries_paths  = sorted(glob(join(queries_folder, "**", "*.jpg"),  recursive=True))
                    
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float)
        self.queries_utms  = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        if split=="train":
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms, 
                                                                radius=args.train_positve_dist_threshold,
                                                                return_distance=False)
        else:
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms, 
                                                                args.val_positive_dist_threshold,
                                                                return_distance=False)            
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        
        self.database_num = len(self.database_paths)
        self.queries_num  = len(self.queries_paths)
        
        num_softpositives=[]
        for i in range(len(self.soft_positives_per_query)):
            num_softpositives.append(len(self.soft_positives_per_query[i]))
            
            
        
        #### Some queries might have no positive, we should remove those queries.
        
        num_softpositives=np.array(num_softpositives)
        self.num_softpositives=num_softpositives
        self.avg_softpositives=np.mean(num_softpositives)
        self.std_softpositives=np.std(num_softpositives)
        
    
    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        # With database images self.test_method should always be "hard_resize"
        img = T.functional.resize(img, self.resize)
        
        return img, index
    
    def __len__(self):
        return len(self.images_paths)
    def __repr__(self):
        return  (f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >")
    def get_positives(self):
        return self.soft_positives_per_query

    
        
class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]
    
    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)
    
    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]

class RAMEfficient4DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 3D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]
    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        assert vals.shape[2] == self.shape[2], f"{vals.shape[2]} {self.shape[2]}"
        assert vals.shape[3] == self.shape[3], f"{vals.shape[3]} {self.shape[3]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)
    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]
        
    

        
class MSLRefs_dataset(data.Dataset):
    """Dataset with images from reference database, can be used to train the model using the Multi similarity loss.
    """
    # Setup for code to use with online triplet mining in combination with multi similarity loss
    # did not finish this code and it is not usable yet
    
    def __init__(self,args, datasets_folder="datasets", dataset_name="pitts30k", split="train",indices=None,val=True,batch_size=1000):
        super().__init__()
        self.val=val
        self.batch_size=batch_size
        args.triplets=False
        
        self.resize = args.resize
        identity_transform = T.Lambda(lambda x: x)
        self.resized_transform = T.Compose([
            T.Resize(self.resize,antialias=True) if self.resize is not None else identity_transform,
            base_transform
        ])
        
        self.query_transform = T.Compose([
                self.resized_transform,
        ])
        
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, split)
        if not os.path.exists(self.dataset_folder):
            self.dataset_folder=join(datasets_folder, dataset_name,"images",split)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        
        #### Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
           
        self.queries_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.database_paths = sorted(glob(join(database_folder, "**", "*.jpg"), recursive=True))
            
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                         radius=25,
                                                         return_distance=False)
        
        self.images_paths = list(self.database_paths) + list(self.queries_paths)

        if len(self.queries_paths)<args.cache_refresh_rate:
            args.cache_refresh_rate=len(self.queries_paths)-1
            args.queries_per_epoch=5*(len(self.queries_paths)-1)
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
    
    
    def __getitem__(self, index):
        if self.val:
            img = path_to_pil_img(self.images_paths[index])
            img = base_transform(img)
            # With database images self.test_method should always be "hard_resize"
            
            img = T.functional.resize(img, self.resize,antialias=True)

            return img, index
    
        if not self.val:
            database_img_ids=np.random.choice(range(len(self.database_paths)),self.batch_size,replace=False)
            query_image_ids=database_img_ids+len(self.database_paths)
            batch_img_ids=np.append(database_img_ids,query_image_ids).flatten()
            
            
            images = [self.resized_transform(path_to_pil_img(self.images_paths[i])) for i in batch_img_ids]
#             print(images[0].type())
        
#             images = torch.Tensor(images)
            images= torch.stack((images[0], *images[1:]), 0)
            
            
            triplets_local_indexes=torch.Tensor(np.linspace(0,self.batch_size-1,self.batch_size))
            batch_img_ids=torch.Tensor(batch_img_ids)
            return images, triplets_local_indexes, batch_img_ids
    
    def __len__(self):
        if self.val:
            return len(self.images_paths)
        if not self.val:
            return 30
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num};"# #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query
