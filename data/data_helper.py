from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from random import sample, random

#from data.JigsawDatasetLoader import JigsawDataset, TestDataset, get_split_dataset_info, _dataset_info
from data.TasksDatasetLoader import JigsawDataset, TestDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset

pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]

available_datasets = pacs_datasets


# Source Dataloaders per train e val
def get_train_dataloader_JiGen(args, device, task="DG"):
 
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    
    target = args.target

    if task == "DA":
        if target in dataset_list:
            dataset_list.remove(target)
    
    datasets = []
    val_datasets = []
    img_transformer, patch_transformer = get_train_transformers(args)
    val_trasformer = get_val_transformer(args)

    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', dname+'.txt'), args.val_size)
        
        if target == dname:
            name_train += name_val
            labels_train += labels_val
        
        train_dataset = JigsawDataset(name_train, labels_train, args.path_dataset, args.scrambled, args.rotated, args.odd, args.jigen_transf, args.grid_size, args.permutation_number, device, task, target_name=args.target, img_transformer=img_transformer, patch_transformer=patch_transformer)
        datasets.append(train_dataset)

        if target != dname:
            val_dataset = TestDataset(name_val, labels_val, args.path_dataset, img_transformer=val_trasformer)
            val_datasets.append(val_dataset)

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    return loader, val_loader

# Target dataloaders per train e test
def get_target_loader(args, device, task):
    
    img_transformer, patch_transformer = get_train_transformers(args)
    val_trasformer = get_val_transformer(args)

    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', args.target+'.txt'))
        
    train_dataset = JigsawDataset(names, labels, args.path_dataset, args.scrambled, args.rotated, args.odd, args.jigen_transf, args.grid_size, args.permutation_number, device, task, target_name=args.target, img_transformer=img_transformer, patch_transformer=patch_transformer)

    val_dataset = TestDataset(names, labels, args.path_dataset, img_transformer=val_trasformer)

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    return loader, val_loader

# Target dataloader per test
def get_val_dataloader(args):

    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', args.target+'.txt'))
    img_tr = get_val_transformer(args)

    val_dataset = TestDataset(names, labels,args.path_dataset, img_transformer=img_tr)
    dataset = ConcatDataset([val_dataset])

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    return loader

def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]

    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))

    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))   
                
    patch_transformer = []
        
    if args.random_grayscale:
        patch_transformer.append(transforms.RandomGrayscale(args.random_grayscale))
        
    patch_transformer += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(patch_transformer)


def get_val_transformer(args):

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)
