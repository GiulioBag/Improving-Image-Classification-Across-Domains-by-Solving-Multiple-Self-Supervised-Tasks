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
from torchvision.utils import save_image

indexxx = 0


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
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="tasks_Alexnet", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")
    
    # JiGen args
    parser.add_argument("--permutation_number", type=int, default=30, help="Permutation number used")
    parser.add_argument("--scrambled", "-s", type=float, default=0.4, help="Percentage of permutated images")
    parser.add_argument("--jig_weight", "-jw", type=float, default=0.7, help="Jigen loss weight during training")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid dimension")

    # Transfert arg
    parser.add_argument("--jigen_transf", "-jt", type=bool, default=False, help="Enable transfer style for Jigen Puzzle")
    
    
    # Rotated args
    parser.add_argument("--rotated", "-r", type=float, default=0.0, help="Percentage of rotated images")
    parser.add_argument("--rot_weight", "-rw", type=float, default=0.0, help="Rotation task loss weight during training")
    
    # Odd One Out args
    parser.add_argument("--odd", "-o", type=float, default=0.0, help="Percentage of Odd One Out images")
    parser.add_argument("--odd_weight", "-ow", type=float, default=0.0, help="Odd One Out task loss weight during training")


    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device          
        
        if args.scrambled == 0:
            args.jig_weight = 0
            
        if args.rotated == 0:
            args.rot_weight = 0
            
        if args.odd == 0:
            args.odd_weight = 0
        
        
        # Set JiGen parameters
        self.alpha_jig = args.jig_weight
        self.permutation_number = args.permutation_number
        
        # Set rotation parameters
        self.alpha_rot = args.rot_weight
        
        # Set Odd parameters
        self.alpha_odd = args.odd_weight
        
        
        model = model_factory.get_network(args.network)(classes=args.n_classes, jigen_classes=self.permutation_number, odd_classes=args.grid_size**2)
        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader_JiGen(args, self.device)
        self.target_loader = data_helper.get_val_dataloader(args)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all)
        
        self.n_classes = args.n_classes



    def _do_epoch(self):
        
        global indexxx
               
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, (data, task_label, class_l, task_type) in enumerate(self.source_loader):

            data, task_label, class_l, task_type = data.to(self.device), task_label.to(self.device), class_l.to(self.device), task_type.to(self.device)
                           
            self.optimizer.zero_grad()
            
            # Forward pass
            class_logit, jigen_logit, rotation_logit, odd_logit = self.model(data)   
            
            # Loss Calculation
            class_loss = criterion(class_logit[task_type == 0], class_l[task_type == 0])
            jigen_loss = criterion(jigen_logit[(task_type == 0) | (task_type == 1)], task_label[(task_type == 0) | (task_type == 1)])
            rotation_loss = criterion(rotation_logit[(task_type == 0) | (task_type == 2)], task_label[(task_type == 0) | (task_type == 2)])
            odd_loss = criterion(odd_logit[task_type == 3], task_label[task_type == 3])
        
            if 0 not in task_type:  
                class_pred = []
            else:
                _, class_pred = class_logit.max(dim=1)
                
            if self.args.scrambled == 0 or 1 not in task_type:
                jigen_pred = []
            else:
                _, jigen_pred = jigen_logit.max(dim=1)
                
            if self.args.rotated == 0 or 2 not in task_type:
                rotation_pred = []
            else:     
                _, rotation_pred = rotation_logit.max(dim=1)
                
            if self.args.odd == 0 or 3 not in task_type:
                odd_pred = []
            else:
                _, odd_pred = odd_logit.max(dim=1)
            
            # Loss calculation
            w_jigen_loss = self.alpha_jig * jigen_loss
            w_rotation_loss = self.alpha_rot * rotation_loss 
            w_odd_loss = self.alpha_odd * odd_loss 
            loss = class_loss + w_jigen_loss + w_rotation_loss + w_odd_loss

            loss.backward()
            self.optimizer.step()

            losses = [class_loss, jigen_loss, rotation_loss, odd_loss]
            preds = [class_pred, jigen_pred, rotation_pred, odd_pred]
            loss_dict, acc_dict = self.get_dicts(losses, preds, task_type, task_label, class_l)
            
            
            self.logger.log(it, len(self.source_loader), loss_dict, acc_dict)
            
            
            del loss, class_loss, class_logit, jigen_loss, jigen_logit, rotation_loss, rotation_logit, odd_loss, odd_logit

        self.model.eval()
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
    
    
    def get_dicts(self, losses, preds, task_type, task_label, class_l):
        
        loss_dict = dict()
        acc_dict = dict()
        
        if type(preds[0]).__name__ == "Tensor":
            loss_dict["Class Loss "] =  losses[0].item()
            acc_dict ["Class Accuracy "] = torch.sum(preds[0][task_type == 0] == class_l[task_type == 0].data).item() / len(preds[0][task_type == 0])
        
        names = ["JiGen", "Rotation"]
        
        del preds[0]
        del losses[0]

        weights =  [self.alpha_jig, self.alpha_rot]
        task_types = [1, 2]
        
        for name, weight, loss, pred, tt in zip(names, weights, losses, preds, task_types):
           
            if weight != 0 and len(pred) > 0:
                loss_dict[name + " Loss"] = loss.item()
                acc_dict [name + " Accuracy"] =  torch.sum(pred[(task_type == 0) | (task_type == tt)] == task_label[(task_type == 0) | (task_type == tt)].data).item()/len(pred[(task_type == 0) | (task_type == tt)])
                
                
        if self.alpha_odd != 0 and len(preds[-1]) > 0:
            loss_dict["Odd Loss"] = losses[-1].item()
            acc_dict ["Odd Accuracy"] =  torch.sum(preds[-1][task_type == 3] == task_label[task_type == 3].data).item()/len(preds[-1][task_type == 3])
            
        return loss_dict, acc_dict
    
def main():
    args = get_args()
    # Check if percentage values are correct
    if args.scrambled + args.rotated + args.odd > 1:
        print("ERROR: the sum of the percentage valuesis greater than 1!")
        return
    elif args.scrambled + args.rotated + args.odd == 1:
        print("WARNING: with the current percentage values you are not training the classifier!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    for i in range(3):
        print("-------> Iterazione: ", i, "<-------")
        trainer = Trainer(args, device)
        trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()