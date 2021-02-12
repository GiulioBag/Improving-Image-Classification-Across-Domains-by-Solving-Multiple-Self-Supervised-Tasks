import itertools
import argparse
import random
from time import time
from os.path import join, dirname

def get_args():
    parser = argparse.ArgumentParser(description="Script to create permutation set.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--permutation_number", type=int, default=30, help="Number of permutation created")
    parser.add_argument("--set_dimension",  default=9, type=int, help="Max value in permutation")
    
    return parser.parse_args()

def Hamming_dist(l1, l2):
    count = 0
    for ele_1, ele_2 in zip(list(l1), list(l2)):
        if ele_1 != ele_2:
            count += 1
    return count



def main():
    print("Inizio generazione permutazioni: ")
    args = get_args()
    start_time = time()
    
    permutation_number = args.permutation_number
    set_dimension = args.set_dimension
    output_path = "../permutations/"
    
    numbers = list(range(set_dimension))
    permutations_list = list(itertools.permutations(numbers))
    
    tup = tuple(range(set_dimension))
    permutations_list.remove(tup)
    
    indexes = random.sample(range(len(permutations_list)), len(permutations_list))
    
    print("Numero di permutazioni: ", len(permutations_list))
    print("Index massimo: ", max(indexes))
    
    good_permutations = [list(permutations_list[indexes[0]])]    
    print("Permutazione 1 trovata: ", permutations_list[indexes[0]])
    del indexes[0]
    
    while len(good_permutations) < permutation_number:
        
        max_min_distance = 0
        
        index_to_remove = None
        permutation_to_remove = None
        
        for it, index in enumerate(indexes):
                        
            permutation = permutations_list[index]
            
            max_dist_count = 0
            
            for g_permutation in good_permutations:
                
                dist = Hamming_dist(permutation, g_permutation)
                #print("->", dist, "--", permutation, "--", g_permutation)
                if dist == set_dimension:
                    max_dist_count += 1
                    
                elif dist > max_min_distance:
                    max_min_distance = dist
                    index_to_remove = it
                    permutation_to_remove = index
                                
            if max_dist_count == len(good_permutations):                
                permutation_to_remove = index
                index_to_remove = it
                break
                                     
        good_permutations.append(list(permutations_list[permutation_to_remove]))
        print("Permutazione" , len(good_permutations),  " trovata: ", permutations_list[permutation_to_remove])
        del indexes[index_to_remove]
    
    name = "permutations_" + str(permutation_number) 
    file = open(join(output_path, name) + ".txt", "w")
    
    text = str(good_permutations)
    file.write(text)
    file.close()
    
    print("Total time: ", time() - start_time)
                
    
    

if __name__ == "__main__":
    main()