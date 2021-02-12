import argparse

import torch
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.MyLogger import Logger
import numpy as np
import torch.nn.functional as func
import itertools


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--path_dataset", default="/content/gdrive/MyDrive/Progetto_ML/progetto/IndexTeam", help="Path where the dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float,help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="tasks_Resnet", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")
    
    # JiGen args
    parser.add_argument("--permutation_number", type=int, default=30, help="Permutation number used")
    parser.add_argument("--scrambled", "-s", type=float, default=0.4, help="Percentage of non-permutated images")
    parser.add_argument("--rotated", "-r", type=float, default=0.0, help="Percentage of non-permutated images")
    parser.add_argument("--jig_weight_target", "-jwt", type=float, default=0.7, help="Target jigen loss weight during training")
    parser.add_argument("--jig_weight_source", "-jws", type=float, default=0.7, help="Source jigen loss weight during training")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid dimension")
    
    # Rotation Task
    parser.add_argument("--rot_weight_target", "-rwt", type=float, default=0.7, help="Target rotation loss weight during training")
    parser.add_argument("--rot_weight_source", "-rws", type=float, default=0.7, help="Source rotation loss weight during training")
    parser.add_argument("--target_entropy_weight", "-wet", type=float, default=0.1, help="Target class loss weight during training")
    
    # Transfert arg
    parser.add_argument("--jigen_transf", "-jt", type=bool, default=False, help="Enable transfer style for Jigen Puzzle")
    
    # Odd One Out args
    parser.add_argument("--odd", "-o", type=float, default=0.0, help="Percentage of Odd One Out images")
    parser.add_argument("--odd_weight_target", "-owt", type=float, default=0.0, help="Target Odd One Out task loss weight during training")
    parser.add_argument("--odd_weight_source", "-ows", type=float, default=0.0, help="Source Odd One Out task loss weight during training")


    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        
        self.args = args
        self.device = device
        
        if args.scrambled == 0:
            args.jig_weight_source = 0
            args.jig_weight_target = 0
          
        if args.rotated == 0:
            args.rot_weight_source = 0
            args.rot_weight_target = 0
            
        if args.odd == 0:
            args.odd_weight_source = 0
            args.odd_weight_target = 0


        model = model_factory.get_network(args.network)(classes=args.n_classes, odd_classes = args.grid_size**2)
        self.model = model.to(device)
        
        # Source Loaders
        self.source_loader, self.val_loader = data_helper.get_train_dataloader_JiGen(args, device, "DA")
        
        # Target Loaders
        self.target_train_loader, self.target_loader = data_helper.get_target_loader(args, device, "DA")

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        
        size = len(self.source_loader.dataset) + len(self.target_train_loader.dataset)
        print("Dataset size: train %d, val %d, test %d" % (size, len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all)

        self.n_classes = args.n_classes
        
        # Set JiGen parameters
        self.jig_alpha_t = args.jig_weight_target
        self.jig_alpha_s = args.jig_weight_source
        self.permutation_number = args.permutation_number
        
        
        # Set Routate parameters
        self.rot_alpha_t = args.rot_weight_target
        self.rot_alpha_s = args.rot_weight_source
        
        # Set target loss weight
        self.target_loss_weight = args.target_entropy_weight

        # Set Odd parameters
        self.odd_alpha_t = args.odd_weight_target
        self.odd_alpha_s = args.odd_weight_source
        
        self.epoch_count = 0
        self.tot_epoch = args.epochs
                        
                        

    def _do_epoch(self):
        
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, (source_batch, target_batch) in enumerate(zip(self.source_loader, itertools.cycle(self.target_train_loader))):
            
            source_data, source_task_label, source_class_l, source_task_type = source_batch 
            target_data, target_task_label, _, target_task_type = target_batch
            
            source_data, source_task_label, source_class_l, source_task_type = source_data.to(self.device), source_task_label.to(self.device), source_class_l.to(self.device), source_task_type.to(self.device)
                                                                                                                    
            target_data, target_task_label, target_task_type = target_data.to(self.device), target_task_label.to(self.device), target_task_type.to(self.device)

            self.optimizer.zero_grad()           
            
            # Calculation source logit
            source_class_logit, source_jigen_logit, source_rotation_logit, source_odd_logit = self.model(source_data)
                                                                                                                                
            # Calculation target logit
            target_class_logit, target_jigen_logit, target_rotation_logit, target_odd_logit = self.model(target_data)
                      
               
            # Source Class loss
            source_class_loss = criterion(source_class_logit[source_task_type == 0], source_class_l[source_task_type == 0])
            
            # Target Class loss
            target_class_loss = torch.sum(-func.softmax(target_class_logit[target_task_type == 0], 1) * func.log_softmax( target_class_logit[ target_task_type == 0], 1), 1).mean()
                       
            # Source Jigen loss
            source_jigen_loss = criterion(source_jigen_logit[(source_task_type == 0) | (source_task_type == 1)], source_task_label[(source_task_type == 0) | (source_task_type == 1)])
            
            # Target Jigen loss
            target_jigen_loss = criterion(target_jigen_logit[(target_task_type == 0) | (target_task_type == 1)], target_task_label[(target_task_type == 0) | (target_task_type == 1)])
            
            # Source Rotation loss
            source_rotation_loss = criterion(source_rotation_logit[(source_task_type == 0) | (source_task_type == 2)], source_task_label[(source_task_type == 0) | (source_task_type == 2)])
            
            # Target Rotation loss
            target_rotation_loss = criterion(target_rotation_logit[(target_task_type == 0) | (target_task_type == 2)], target_task_label[(target_task_type == 0) | (target_task_type == 2)])
            
            # Source Odd loss
            source_odd_loss = criterion(source_odd_logit[source_task_type == 3], source_task_label[source_task_type == 3])
            
            # Target Odd loss
            target_odd_loss = criterion(target_odd_logit[target_task_type == 3], target_task_label[target_task_type == 3])

                        
            if 0 not in source_task_type:  
                class_pred = []
            else:
                _, class_pred = source_class_logit[source_task_type == 0].max(dim=1)
                
            if self.args.scrambled == 0 or 1 not in source_task_type:
                source_jigen_pred = []
            else:
                _, source_jigen_pred = source_jigen_logit[(source_task_type == 0) | (source_task_type == 1)].max(dim=1)
                
            if self.args.scrambled == 0 or 1 not in target_task_type:
                target_jigen_pred = []
            else:
                _, target_jigen_pred = target_jigen_logit[(target_task_type == 0) | (target_task_type == 1)].max(dim=1)
            
            if self.args.rotated == 0 or 2 not in source_task_type:
                source_rotation_pred = []
            else:
                _, source_rotation_pred = source_rotation_logit[(source_task_type == 0) | (source_task_type == 2)].max(dim=1)
                
            if self.args.rotated == 0 or 2 not in target_task_type:
                target_rotation_pred = []
            else:
                _, target_rotation_pred = target_rotation_logit[(target_task_type == 0) | (target_task_type == 2)].max(dim=1)
           
            if self.args.odd == 0 or 3 not in source_task_type:
                source_odd_pred = []
            else:
                _, source_odd_pred = source_odd_logit[source_task_type == 3].max(dim=1)
                
            if self.args.odd == 0 or 3 not in target_task_type:
                target_odd_pred = []
            else:
                _, target_odd_pred = target_odd_logit[target_task_type == 3].max(dim=1)
            
            # Loss calculation
            class_loss = source_class_loss + target_class_loss * self.target_loss_weight
            jigen_loss = target_jigen_loss * self.jig_alpha_t + source_jigen_loss * self.jig_alpha_s
            rotation_loss = target_rotation_loss * self.rot_alpha_t + source_rotation_loss * self.rot_alpha_s
            odd_loss =  target_odd_loss * self.odd_alpha_t + source_odd_loss * self.odd_alpha_s
            loss = class_loss + jigen_loss + rotation_loss + odd_loss

            loss.backward()
            self.optimizer.step()
            
            losses = [source_class_loss, target_class_loss, source_jigen_loss, source_rotation_loss, target_jigen_loss, target_rotation_loss, source_odd_loss, target_odd_loss]
            
            
            preds  = [class_pred, source_jigen_pred, source_rotation_pred, target_jigen_pred, target_rotation_pred, source_odd_pred, target_odd_pred]
            
            loss_dict, acc_dict = self.get_dicts(losses, preds, source_task_type, source_task_label, source_class_l, target_task_type, target_task_label)
            
                            
            self.logger.log(it, len(self.source_loader), loss_dict, acc_dict)
            
            del  source_class_logit, target_class_logit, source_jigen_logit, target_jigen_logit, source_class_loss, target_class_loss, source_jigen_loss, target_jigen_loss, source_rotation_loss, target_rotation_loss, class_loss, jigen_loss, rotation_loss, loss, source_odd_logit, target_odd_logit, source_odd_loss, target_odd_loss


        self.epoch_count +=1
        self.model.eval()
        print("Test Phase:")
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():               
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"Classification Accuracy": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, (data, class_l) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit, _, __, ___ = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
            self.scheduler.step()

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        self.logger.save_best()
        return self.logger, self.model
    
    
    def get_dicts(self, losses, preds, source_task_type, source_task_label, source_class_l, target_task_type, target_task_label):
        
        loss_dict = dict()
        acc_dict = dict()
        
        if type(preds[0]).__name__ == "Tensor":
            loss_dict["Class Source Loss "] =  losses[0].item()
            acc_dict ["Class Source Accuracy "] = torch.sum(preds[0] == source_class_l[source_task_type == 0].data).item()/len(preds[0])
            

        
        loss_dict["Class Target Loss "] =  losses[1].item()
        
        del preds[0]
        del losses[0]
        del losses[1]
        
        names = ["Source JiGen", "Source Rotation", "Target JiGen", "Target Rotation"]
        weights = [self.jig_alpha_s, self.rot_alpha_s, self.jig_alpha_t, self.rot_alpha_t]
        task_types = [1, 2, 1, 2]
        
        for name, weight, loss, pred, tt, it in zip(names, weights, losses, preds, task_types, range(len(names))):
           
            if it < 2:
                task_type = source_task_type
                task_label = source_task_label
            else:
                task_type = target_task_type
                task_label = target_task_label
                
            if weight != 0 and len(pred) > 0:
                                
                loss_dict[name + " Loss"] = loss.item()
                acc_dict [name + " Accuracy"] =  torch.sum(pred == task_label[(task_type == 0) | (task_type == tt)].data).item() / len(preds[it])
                
        
        if self.odd_alpha_s != 0 and len(preds[-2]) > 0:
            loss_dict["Odd Source Loss"] = losses[-2].item()
            acc_dict ["Odd Source Accuracy"] =  torch.sum(preds[-2] == source_task_label[source_task_type == 3].data).item()/len(preds[-2])
            
        if self.odd_alpha_t != 0 and len(preds[-1]) > 0:
            loss_dict["Odd Target Loss"] = losses[-1].item()
            acc_dict ["Odd Target Accuracy"] =  torch.sum(preds[-1] == target_task_label[target_task_type == 3].data).item()/len(preds[-1])
            
        return loss_dict, acc_dict


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # Check if percentage values are correct
    if args.scrambled + args.rotated + args.odd > 1:
        print("ERROR: the sum of the percentage valuesis greater than 1!")
        return
    elif args.scrambled + args.rotated + args.odd == 1:
        print("WARNING: with the current percentage values you are not training the classifier!")
        
    for i in range(1,4):
        print("-------> Iterazione: ", i, "<-------")
        trainer = Trainer(args, device)
        trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()