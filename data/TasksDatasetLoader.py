import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.utils as utils
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
from PIL import Image
from random import sample, random, shuffle
from torchvision.utils import save_image
import copy
from models.style_transfer_model import Model
from os.path import join, dirname
import ast
from time import time
import torch.nn.functional as F

def get_random_subset(names, labels, percent):
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)



class JigsawDataset(data.Dataset):
    # beta indica la percentuale di imagini che NON saranno scomposte
    # grid_size dimensione lato griglia
    # num_perm numero di permutazioni prese in considerazione
    # jigen_transf enables jigen transfer style
    def __init__(self, names, labels, path_dataset, beta_jig, beta_rot, beta_odd, jigen_transf, grid_size, num_perm, device, type_domain_shift, target_name, img_transformer=None, patch_transformer=None, style_transfer_type="all"):
                
        self.data_path = path_dataset
        
        self.names = names
        self.labels = labels
        
        self._image_transformer = img_transformer
        self.patch_transformer = patch_transformer
        
        self.device = device
        
        # JiGen Parameters
        self.beta_jig = beta_jig
        self.grid_size = grid_size
        self.num_perm = num_perm
        
        self.permutations = self.get_perm()
        
        # Rot parametrs
        self.beta_rot = beta_rot
        
        # Style Transfert Parameter
        self.style_transfer_type = style_transfer_type
        self.target_domain = target_name
        self.jigen_transf = jigen_transf
        self.type_domain_shift = type_domain_shift
        self.style_model = self.model_acquire()


        
        # Odd One Out parameters
        self.beta_odd = beta_odd
        
    def get_perm(self):
        name = "permutations_" + str(self.num_perm) 
        output_path = "permutations/"

        file = open(join(output_path, name) + ".txt", "r")
        for row in file.readlines():
            permutations = ast.literal_eval(row)

        return permutations
        
    def model_acquire(self):

        model = Model()

        model_names = os.listdir("models/pretrained/")
        for name in model_names:
          if "style_transfer_model_" in name and self.target_domain in name and self.type_domain_shift == "DG":
            break
          elif self.type_domain_shift == "DA" and "all" in name:
            break
        print("Modello scelto: ", name)
        
        model_state = torch.load("models/pretrained/" + name, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_state, strict=False)
        model = model.to(self.device)
        return model
    
    def style_acquire(self, domain, index):
               
        path_domain = self.data_path + '/' + self.names[index].split("/")[0] + '/' + self.names[index].split("/")[1]
        style_images = []
        
        
        if self.type_domain_shift == "DG":
            domains = os.listdir(path_domain)
            domains.remove(domain)
            domains.remove(self.target_domain)       
                    
        elif self.style_transfer_type == "all":
            domains = os.listdir(path_domain)
            domains.remove(domain)
        else:
            domains = [self.target_domain]
            
        while len(style_images) != 9:
            image_domain = domains[np.random.randint(len(domains))]
            incr_path = path_domain + "/" + image_domain
            image_class = os.listdir(incr_path)[np.random.randint(len(os.listdir(incr_path)))]
            incr_path += "/" + image_class 
            image_style = os.listdir(incr_path)[np.random.randint(len(os.listdir(incr_path)))]
            incr_path += "/" + image_style
            
            if incr_path[-4:] == ".jpg" or  incr_path[-4:] == ".png":
                style_images.append(self._image_transformer(Image.open(incr_path).convert('RGB')))
                                
        return style_images

    def style_changer(self, content_image, domain, index):
        global indexxx;
        
        self.style_images = self.style_acquire(domain, index)
        
        size = content_image.size[0]       
        edge_len = float(size)/self.grid_size 
                
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        
        std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(self.device)
        mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(self.device)
                
        style_index = 0
        aux_patches = []

        '''
        if indexxx % 10 == 0:
          save_image( transforms.ToTensor()(content_image), "Prova img/" + str(indexxx) + "prima.jpg");
        '''

        for i in range(self.grid_size):
            for j in range (self.grid_size):
                
                style_image = self.style_images[style_index]
                s_tensor = trans(style_image).unsqueeze(0).to(self.device) 

                c_tensor = trans(content_image.crop([i * edge_len, j * edge_len, (i + 1) * edge_len, (j + 1) * edge_len])).unsqueeze(0).to(self.device) 
                

                with torch.no_grad():
                    out = self.style_model.generate(c_tensor, s_tensor, 1)

                out = torch.clamp(out * std + mean, 0, 1)
                out = torch.squeeze(out)
                aux_patches.append(out)
                
                style_index+=1
        
        task_label = np.random.randint(len(self.permutations)) + 1
        permutation = self.permutations[task_label - 1]
              
        patches = []
        for aux_index in permutation:        
            patches.append(aux_patches[aux_index])

        img_rows = []

        for tile, tile_index in zip(patches, range(len(patches))):
            if tile_index % self.grid_size == 0:
                img_rows.append(tile)
            else:                  
                img_rows[-1] = torch.cat((img_rows[-1], tile), 1)

        img = img_rows[0]
        for row in img_rows[1:]:
            img = torch.cat((img, row), 2)


        img = transforms.ToPILImage(mode="RGB")(img)
        img = transforms.Resize([size, size])(img)
        img = transforms.ToTensor()(img)

        '''
        if indexxx % 10 == 0:
          save_image(img, "Prova img/" + str(indexxx) + "nonordinata.jpg");
        indexxx += 1
        '''
        
        del self.style_images
        return img, task_label

        
    # from PILImg to Tensor
    def patchTransfmormations(self, img, task_type):
        
        task_label = 0
        aux_patches = []
        edge_len = float(img.size[0])/self.grid_size 
        for i in range(self.grid_size):
            for j in range (self.grid_size):
                aux_patches.append(self.patch_transformer(img.crop([i * edge_len, j * edge_len, (i + 1) * edge_len, (j + 1) * edge_len]))) 
                
        if task_type == 1:
            task_label = np.random.randint(len(self.permutations)) + 1
            permutation = self.permutations[task_label - 1]
        else:
            permutation = list(range(self.grid_size * self.grid_size))

        patches = []
        for aux_index in permutation:        
            patches.append(aux_patches[aux_index])

        img_rows = []

        for tile, tile_index in zip(patches, range(len(patches))):
            if tile_index % self.grid_size == 0:
                img_rows.append(tile)
            else:                  
                img_rows[-1] = torch.cat((img_rows[-1], tile), 1)

        img = img_rows[0]
        for row in img_rows[1:]:
            img = torch.cat((img, row), 2)

        return img, task_label
   
    def oddPatches(self, img, index):
        
        delta = np.random.randint(4) + 1
        
        if index + delta >= len(self.names):
            odd_index = index - delta
        else:
            odd_index = index + delta
        
        framename = self.data_path + '/' + self.names[odd_index]
        odd_img = Image.open(framename).convert('RGB')
        odd_img = self._image_transformer(odd_img)

        crop = transforms.ToTensor()(transforms.RandomCrop(odd_img.size[0]/self.grid_size)(odd_img))
        
        aux_patches = []
        edge_len = float(img.size[0])/self.grid_size 
        for i in range(self.grid_size):
            for j in range (self.grid_size):
                aux_patches.append(self.patch_transformer(img.crop([i * edge_len, j * edge_len, (i + 1) * edge_len, (j + 1) * edge_len]))) 
                
        in_patch_selection = np.random.randint(self.grid_size**2)
        
        permutation_index = np.random.randint(len(self.permutations))
        permutation = self.permutations[permutation_index]
        
        patches = []
        for aux_index in permutation:        
            patches.append(aux_patches[aux_index])
            
        patches[in_patch_selection] = crop

        img_rows = []

        for tile, tile_index in zip(patches, range(len(patches))):
            if tile_index % self.grid_size == 0:
                img_rows.append(tile)
            else:                  
                img_rows[-1] = torch.cat((img_rows[-1], tile), 1)

        img = img_rows[0]
        for row in img_rows[1:]:
            img = torch.cat((img, row), 2)
            
        return img, in_patch_selection
        

    def __getitem__(self, index):

        global indexxx
        
        framename = self.data_path + '/' + self.names[index]
              
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        task_selection = random()

        edge_1 = self.beta_jig
        edge_2 = self.beta_jig + self.beta_rot
        edge_3 = self.beta_jig + self.beta_rot + self.beta_odd
       
        
        # Jigsaw task
        if  task_selection < edge_1:
            task_type = 1

            if self.jigen_transf:
                # Transfer Jigen Puzzle
                domain = self.names[index].split("/")[2]
                img, task_label = self.style_changer(img, domain, index)
            else:            
                # Nomral Jigen Puzzle
                img, task_label = self.patchTransfmormations(img, task_type)

            
        # Rotate task
        elif task_selection < edge_2:
            task_type = 2  
            task_label = np.random.randint(3) + 1
            img = transforms.functional.rotate(img, task_label * 90)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            #img, _ = self.patchTransfmormations(img, task_type)        
            
        #  Odd One Out Task
        elif task_selection < edge_3:
            task_type = 3
            img, task_label = self.oddPatches(img, index)
            
        # Normal task
        else:
            task_type = 0
            img, task_label = self.patchTransfmormations(img, task_type)

                                                  
        return img, task_label, int(self.labels[index]), task_type
        


    
    def __len__(self):
        return len(self.names)

    
class TestDataset(data.Dataset):
    def __init__(self, names, labels, path_dataset, img_transformer=None):
               
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):        
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        return img, int(self.labels[index])
    
    def __len__(self):
        return len(self.names)
    
    


