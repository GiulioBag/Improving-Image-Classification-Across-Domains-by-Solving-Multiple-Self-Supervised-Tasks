from time import time
from os.path import join, dirname
import os

_log_path = join(dirname(__file__), '../logs')

class Logger():
    def __init__(self, args, update_frequency=10):
        # Setto i parametri con i valori iniziali
        self.start_time = time()
        self.current_epoch = 0
        self.update_fre = update_frequency

        self.accuracies_val = []
        self.accuracies_test = []
        
        
        # Controllo se la catella esiste altrimenti la creo, non controlla l'ordine dei domini
        self.sources = ""
                
        for source in args.source:
            self.sources += "" + source + "_"
            
        if "jig_weight_target" in args.__dict__.keys():
            str_init = "DA_"
        else:
            str_init = "DG_"
        
        self.folder_name = str_init + "from_" + self.sources + "to_" + args.target  
        
        log_path = join(_log_path, self.folder_name)
        
        if not os.path.isfile(log_path):
            try:
                os.mkdir(log_path)
            except OSError:
                print ("")
                
        # Acquisisco i parametri
        self.text = "\n           <------------------------------------------------------------------------------------------------------------------->          \n"
        self.getArgsParam(args)
        self.getName(args)
        
    def getName(self, args):
        args_dict = args.__dict__
        self.name = ""
        keys = list(args_dict.keys())
        complete_name = ["learning_rate", "scrambled", "jig_weight", "jig_weight_target", "jig_weight_source", "target_entropy_weight"]
        name = ["lr", "beta", "alpha", "alpha_t", "alpha_s", "entro"]
        
        for cn, n in zip(complete_name, name):
            if cn in keys:
                self.name += n + "=" + str(args_dict[cn]) + "_"
        self.name = self.name[:-1]        
                
    def getSpace(self, len_k):
        return (self.longest_key - len_k) * " "
                
    def getArgsParam(self, args):
        
        args_dict = args.__dict__
        self.text += "Params: \n"
        
        self.tot_epoch = args_dict["epochs"]
        
        self.longest_key = 0
        for k in args_dict.keys():
            if len(k) > self.longest_key:
                self.longest_key = len(k)
        self.longest_key += 1
        
        for k, v in args_dict.items():
            
            if type(v).__name__ != "list":
                self.text += " " *4 + k + self.getSpace(len(k) - 4) + ":  " + str(v) + "\n"
            else:
                self.text += " " *4 + k + self.getSpace(len(k) - 4) + ":  "
                first = True
                for ele in v:
                    if first:
                        self.text += str(ele) + "\n"
                        first = False
                    else:
                        self.text += self.getSpace(-11) + str(ele) + "\n"
        self.text += "\n\n"
                                
    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.text += "Epoch " + str(self.current_epoch) + "/" + str(self.tot_epoch) + ", learning rate: " + str(learning_rates[0]) + "\n"
        print("Epoch " + str(self.current_epoch) + "/" + str(self.tot_epoch) + ", learning rate: " + str(learning_rates[0]) + "\n")
        
    def log(self, it, iters, losses, accuracies):
        
        if it % self.update_fre == 0:
            
            aux_str = " " * 4 + "Iterations: " + str(it) + "/" + str(iters) + "\n" 
            for k, v in losses.items():
                aux_str += " " * 8 + k + ": " + str(round(v, 4)) + "\n"
                
            for k, v in accuracies.items():
                aux_str += " " * 8 + k + ": " + str(round(v*100, 4)) + "\n"
                
            print(aux_str)
            aux_str += "\n"
            self.text += aux_str
                
    def log_test(self, phase, accuracies):
        
        aux_aux_str = ""
        for k, v in accuracies.items():
            aux_aux_str += k + " : " + str(round(v * 100, 3))
        
        aux_str = " " * 4 + "Accuracies on %s: " % phase + " " + aux_aux_str + "\n"
        self.text += aux_str
        print(aux_str)
        
        if phase == "val":
            self.accuracies_val.append(round(list(accuracies.values())[0] * 100,3))
        else:
            self.accuracies_test.append(round(list(accuracies.values())[0] * 100,3))
        
    def save_best(self):
        aux_str = "\n\nIt took %g" % (time() - self.start_time) + "\n"
        print(aux_str)
        self.text += aux_str
        
        max_val_value = 0
        max_test_value = 0
        test_value_max_val = 0
        
        best_val_epoch = 0
        best_test_epoch = 0
        
        for epoch, (val, test) in enumerate(zip(self.accuracies_val, self.accuracies_test)):
            
            if val > max_val_value:
                max_val_value = val
                test_value_max_val = test
                best_val_epoch = epoch
                
            if test > max_test_value:
                max_test_value = test
                best_test_epoch = epoch
                
        aux_str = "Best value in validation: " + str(round(max_val_value, 3))+ " corresponding test value: " + str(round(test_value_max_val, 3)) + " reached in " + str(best_val_epoch) + "\n"
        
        aux_str += "Best value in test: " + str(round(max_test_value, 3)) + " reached in " + str(best_test_epoch) + "\n"
        
        self.text += aux_str
        print(aux_str)
   
        aux_str = "Validation Accuracies: [" 
        for val in self.accuracies_val:
            aux_str += str(val) + ", "
        aux_str = aux_str[:-2] + "]\n"
        
        aux_str += "Test Accuracies: [" 
        for val in self.accuracies_test:
            aux_str += str(val) + ", "
        aux_str = aux_str[:-2] + "]\n"
        
        self.text += aux_str
        
                
        # Write logger      
        f = open(join(_log_path, self.folder_name, self.name) + ".txt", "a")
        f.write(self.text)
        f.close()
        
            





                
            
            
        
        
        