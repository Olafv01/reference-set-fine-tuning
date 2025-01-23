# VPR reference set fine-tuning

This git contains code to fine-tune existing VPR techniques on reference images of existing test datasets.
To use this codebase, the packages in the environment file need to be installed.

Curretly, only CricaVPR, BoQ and SALAD models can be used to fine-tune on.
All datasets downloaded using the code of VPR dataset downloader ((https://github.com/gmberton/VPR-datasets-downloader)) can be used to fine-tune on.

To reproduce the results, first validation images need to be saved, this can be done by using the argument `--create_augments`.
After that, the code stops and experiments can be performed on that dataset.

To reproduce the BoQ fine-tuning on Nordland use the following line:

First create the augments.
```
python train_ref.py --dataset_name nordland --method boq --create_augments
```
Then, to fine-tune use this:
```

python train_ref.py --dataset_name nordland --method boq --use_val_augments True --patience 5 --train_batch_size 16 --lr 1e-7 
```
this is a list of all available settings and their effect:
| Argument     | Default     | Explanation | 
| ------------- | ------------- |------------- |
| `--datasets_folder` | "../datasets_vg/datasets" |location of the downloaded datasets |
| `--dataset_name` | *Required* | name of the dataset as defined by the dataset folder name|
| `--method` | "crica"| method to fine-tune (only crica, salad or boq and without capitols) |
| `--original_training_data` | gsv| only used to create the correct logging folder |
| `--num_workers` |6 | amount of process to use during fine-tuning|
| `--train_batch_size` | 16| |train batch size (set to 1 if images have different sizes)
| `--infer_batch_size` | 16| validation batch size (set to 1 if images have different sizes)|
| `--seed` |0| torch and numpy random seed |
| `--mining` |full | triplet mining (only full, partial or random)|
| `--lr` | 1e-7 | learning rate to use for fine-tuning|
| `--margin` | 0.1| triplet loss margin|
| `--negs_num_per_query` |2 | number of negatives to used in each triplet|
| `--epochs_num` |100 | max amount of triplets, if the patience is not reached/used|
| `--log_frequency` |1008 | amount of iterations between validation steps (must be a mulitple of train_batch_size)|
| `--ablation` | False | Set to true, if you want to use part of the test queries during fine-tuning|
| `--ablation_nordland` |False | Set to true, if you want to use the created validation images on the Nordland dataset|
| `--save_models` |True | saves the fine-tuned best model and fine-tuned final model|
| `--save_no_models` | -| Sets save_models to false, only if you want the results of the experiments and do not want to use the fine-tuned models|
| `--test_val_queries` |0.1 | split of the test queries to use as validation|
| `--grayscale` |False | Set to True, if you want all images to be grayscale during validation|
| `--exact_match` | False| Set to True, if you want to only allow the exact match to be used as a positive during training|
| `--patience` | 5| Number of epochs to wait after the R@5 did not improve on the validation set |
| `--create_augments` |False | Set to True, if you want to create validation augmented images stops the program after creating the images, does not continue with fine-tuning|
| `--use_val_augments` |False | Set to True, if you want to use the prior created images instead of creating random ones during validation ( training images are always random) |
| `--val_save_dir` | /home/osverburg/validation | location where to save the augmented validation images|
| `--log_dir` | ../logs| location where to save all log files|
| `--gpu_id` | 0| id of the GPU to use, if applicable |
| `--random_vals` |True | use random image for training and validation|
| `--no_random_vals` |- | set random-vals to False, which results in the training and validation splits to be seperate (first part training, final part validations)|
| `--stop_after_patience` |False |Set to True, if you want to stop the program after the patience limit has been reached|
| `--val_split` | 0.3| part of the reference set to use as validation images|
| `--val_positive_dist_threshold` |25 | positive threshold to used during validation|
| `--train_positive_dist_threshold` |10 | positive threshold to used during training|
| `--recall_values` |[1,5,10,20] | list of recall values that are wanted (R@1,R@5,R@10,R@20 are used in my paper)|



## Used githubs
For this project, the following gits were used:

Dataset downloading

https://github.com/gmberton/VPR-datasets-downloader

CricaVPR:

https://github.com/Lu-Feng/CricaVPR/tree/main

BoQ:

https://github.com/amaralibey/Bag-of-Queries/tree/main

SALAD:

https://github.com/serizba/salad
