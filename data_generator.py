'''
@ Description: This code is used to generated the trainable data through propsoed propositionalization mothod.
@ If the data is large, we can choose the subsample rate. If the data is not large enough, we make 'dt' as the default format. 
@ Date: 2022.04.11
@ Author: Kun Gao
@ Version: 2.0
'''

# -*- coding: utf-8 -*-
from multiprocessing import set_start_method
import os
import sys

from pathlib import Path
import csv
import time
import random

from matplotlib.pyplot import show

#path = os.getcwczd()
path = Path(os.getcwd())
parent_path = path.parent.absolute()
sys.path.append(os.getcwd())
sys.path.append(str(parent_path))    
#sys.path.append(os.getcwd())
import logging
logger = logging.getLogger()
logger.basicConfig = logging.basicConfig(level=logging.DEBUG) # this is my first trial f
import itertools
import pickle 
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def first_order_features_example(body, head):

    feature = {
        'body': _bytes_feature(serialize_array(body)),
        'head': _bytes_feature(serialize_array(head)),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))


def classifier(all_relation_path, variable_number, original_data_path, t_relation = ''):
    '''
    Get some information about the relation, including the relation and objects, etc. 
    Comments: There is no random operations in this function. Run multiple times at different time from the same input will get the same output.
    '''
    all_rlation = {}
    save_all_relation = {}
    all_predicate = []
    all_object = {}
    relation = {}
    arity_relation = {}
    pro = {}
    with open(all_relation_path,'r') as f:
        new_single_line  = f.readline()        
        while new_single_line:
            one_perd = []
            # retrieval the relation name and two objects
            predicate = new_single_line[0:new_single_line.index(')')]
            if '#' in new_single_line:
                if 'TEST' in new_single_line:
                    new_single_line = f.readline()
                    continue
                if 'T.#' in new_single_line or 'T#' in new_single_line:
                    probability = 1
                else:
                    probability = new_single_line[new_single_line.index('.'):new_single_line.index('#')].replace('.','').replace(' ','').replace('#','')
                    probability = '0.'+probability
                    probability = float(probability) 
            else:
                probability = 1
            single_line = predicate.split('(')
            relation_name = single_line[0]
            the_rest = single_line[1].split(",")
            first_obj = the_rest[0]
            second_obj = the_rest[1]
            one_perd.append(relation_name)
            one_perd.append(first_obj)
            one_perd.append(second_obj)
            one_perd.append(probability)
            all_predicate.append(one_perd)
            if first_obj not in all_object:
                all_object[first_obj] = set([])
            if second_obj not in all_object:
                all_object[second_obj] = set([])
            if relation_name not in all_rlation:
                all_rlation[relation_name] = [set([]),set([])]
            relation[relation_name] = []
            arity_relation[relation_name] = 1 #  initial arity for all relation are 1
            new_single_line = f.readline()
            pro[predicate+')'] = probability
        f.close()
    all_relation_list = list(all_rlation)
    
    
    # Save all predicate name into the file 
    for i in all_relation_list:
        save_all_relation[i] = i
    with open(original_data_path+'/all_relation_dic.dt','wb') as f:
        pickle.dump(save_all_relation,f)
        f.close()
    with open(original_data_path+'/all_relation_dic.txt','w') as f:
        print(str(save_all_relation),file = f)
        f.close()  
    
    for pred in all_predicate:
        first_string = str(all_relation_list.index(pred[0])) + '-1'
        second_string = str(all_relation_list.index(pred[0])) + '-2'
        all_object[pred[1]].add(first_string)
        all_object[pred[2]].add(second_string)

    for pred in all_predicate:
        one_tuple = []
        one_tuple.append(pred[1])
        one_tuple.append(pred[2])
        one_tuple = tuple(one_tuple) 
        relation[pred[0]].append(one_tuple) # {'relation_name': (),(),...,()}

        # check the arity of all predicate 
        if pred[1] != pred[2] and arity_relation[pred[0]] == 1:
            arity_relation[pred[0]] = 2
    

    # Make variable - object dictionary
    variable_objects = {} # TODO this dictionary stores the classification of each variables in the task 
    for i in range(variable_number):
        variable_objects[i] = set([])
    
    target_arity = arity_relation[t_relation]
    if target_arity == 2:     
        for pred in all_predicate:
            if pred[0] == t_relation:
                variable_objects[0].add(pred[1])
                variable_objects[1].add(pred[2])
        # add variable into the rest of the variable 
        for obj in all_object:
            for variable_name in range(2,variable_number):
                variable_objects[variable_name].add(obj)
    elif target_arity == 1: # TODO. The arity of targetr predicate is 1. In order to make both postive and negative predicate, we ask all variables have the same classification 
        for obj in all_object:
            for variable_name in range(variable_number):
                variable_objects[variable_name].add(obj)
                # variable_objects[1].add(obj)
                # variable_objects[2].add(obj)
    
    # Save relation
    with open(original_data_path+'/relation_entities.dt','wb') as f:
        pickle.dump(relation, f)
        f.close()
    # Save the probability of all relational facts
    
                
    with open(original_data_path+'/pro.dt','wb') as f:
        pickle.dump(pro,f)
        f.close()
    with open(original_data_path+'/pro.txt', 'w') as f:
        print(pro,file=f)
        f.close()
    
    return variable_objects, list(relation.keys()), relation, arity_relation, target_arity


def classifier_with_separate_files(facts_path, train_path, variable_number, original_data_path, t_relation=''):
    '''
    Get information about relations from two separate files:
    - facts_path: Background knowledge (facts.pl) - predicates used for rule body generation
    - train_path: Target edges (train.pl) - facts to learn definitions for, not used in rule inference
    
    Only predicates from facts.pl are included in the domain for rule bodies.
    Train.pl facts are not included in background when computing rule inference.
    '''
    # First, load background knowledge from facts.pl
    background_relation = {}
    background_arity_relation = {}
    background_all_predicate = []
    all_object = {}
    pro = {}
    
    logging.info(f"Loading background knowledge from: {facts_path}")
    with open(facts_path, 'r') as f:
        new_single_line = f.readline()
        while new_single_line:
            one_perd = []
            # Skip empty lines
            if not new_single_line.strip():
                new_single_line = f.readline()
                continue
            # Skip TEST lines
            if '#' in new_single_line and 'TEST' in new_single_line:
                new_single_line = f.readline()
                continue
            
            # Parse predicate
            try:
                predicate = new_single_line[0:new_single_line.index(')')]
            except ValueError:
                new_single_line = f.readline()
                continue
                
            # Handle probability
            if '#' in new_single_line:
                if 'T.#' in new_single_line or 'T#' in new_single_line:
                    probability = 1
                else:
                    probability = new_single_line[new_single_line.index('.'):new_single_line.index('#')].replace('.','').replace(' ','').replace('#','')
                    probability = '0.'+probability
                    probability = float(probability)
            else:
                probability = 1
            
            single_line = predicate.split('(')
            relation_name = single_line[0]
            the_rest = single_line[1].split(",")
            first_obj = the_rest[0]
            second_obj = the_rest[1]
            
            one_perd.append(relation_name)
            one_perd.append(first_obj)
            one_perd.append(second_obj)
            one_perd.append(probability)
            background_all_predicate.append(one_perd)
            
            if first_obj not in all_object:
                all_object[first_obj] = set([])
            if second_obj not in all_object:
                all_object[second_obj] = set([])
            if relation_name not in background_relation:
                background_relation[relation_name] = []
            background_arity_relation[relation_name] = 1
            new_single_line = f.readline()
            pro[predicate+')'] = probability
        f.close()
    
    logging.info(f"Loaded {len(background_all_predicate)} background facts with {len(background_relation)} predicates")
    
    # Now load target edges from train.pl
    target_all_predicate = []
    target_relation_facts = {}
    
    logging.info(f"Loading target edges from: {train_path}")
    with open(train_path, 'r') as f:
        new_single_line = f.readline()
        while new_single_line:
            one_perd = []
            # Skip empty lines
            if not new_single_line.strip():
                new_single_line = f.readline()
                continue
            # Skip TEST lines
            if '#' in new_single_line and 'TEST' in new_single_line:
                new_single_line = f.readline()
                continue
            
            # Parse predicate
            try:
                predicate = new_single_line[0:new_single_line.index(')')]
            except ValueError:
                new_single_line = f.readline()
                continue
            
            # Handle probability
            if '#' in new_single_line:
                if 'T.#' in new_single_line or 'T#' in new_single_line:
                    probability = 1
                else:
                    probability = new_single_line[new_single_line.index('.'):new_single_line.index('#')].replace('.','').replace(' ','').replace('#','')
                    probability = '0.'+probability
                    probability = float(probability)
            else:
                probability = 1
            
            single_line = predicate.split('(')
            relation_name = single_line[0]
            the_rest = single_line[1].split(",")
            first_obj = the_rest[0]
            second_obj = the_rest[1]
            
            one_perd.append(relation_name)
            one_perd.append(first_obj)
            one_perd.append(second_obj)
            one_perd.append(probability)
            target_all_predicate.append(one_perd)
            
            # Add objects from train.pl to all_object
            if first_obj not in all_object:
                all_object[first_obj] = set([])
            if second_obj not in all_object:
                all_object[second_obj] = set([])
            
            # Store target relation facts separately
            if relation_name not in target_relation_facts:
                target_relation_facts[relation_name] = []
            
            new_single_line = f.readline()
            pro[predicate+')'] = probability
        f.close()
    
    logging.info(f"Loaded {len(target_all_predicate)} target facts")
    
    # Build the relation list from ONLY background predicates (facts.pl)
    all_relation_list = list(background_relation.keys())
    save_all_relation = {}
    for i in all_relation_list:
        save_all_relation[i] = i
    
    with open(original_data_path+'/all_relation_dic.dt','wb') as f:
        pickle.dump(save_all_relation,f)
        f.close()
    with open(original_data_path+'/all_relation_dic.txt','w') as f:
        print(str(save_all_relation),file = f)
        f.close()
    
    # Process background predicates
    for pred in background_all_predicate:
        first_string = str(all_relation_list.index(pred[0])) + '-1'
        second_string = str(all_relation_list.index(pred[0])) + '-2'
        all_object[pred[1]].add(first_string)
        all_object[pred[2]].add(second_string)
    
    # Build relation tuples from background predicates
    for pred in background_all_predicate:
        one_tuple = []
        one_tuple.append(pred[1])
        one_tuple.append(pred[2])
        one_tuple = tuple(one_tuple)
        background_relation[pred[0]].append(one_tuple)
        
        # Check arity
        if pred[1] != pred[2] and background_arity_relation[pred[0]] == 1:
            background_arity_relation[pred[0]] = 2
    
    # Make variable-object dictionary
    variable_objects = {}
    for i in range(variable_number):
        variable_objects[i] = set([])
    
    # Get target arity - first check train.pl, then background
    target_arity = None
    if t_relation in target_relation_facts:
        # Infer arity from target facts
        target_arity = 2  # Default
        for pred in target_all_predicate:
            if pred[0] == t_relation:
                if pred[1] == pred[2]:
                    target_arity = 1
                else:
                    target_arity = 2
                    break
    elif t_relation in background_arity_relation:
        target_arity = background_arity_relation[t_relation]
    else:
        logging.warning(f"Target relation {t_relation} not found in either train.pl or facts.pl, assuming arity 2")
        target_arity = 2
    
    # Build variable objects based on target relation from BOTH files
    all_target_predicates = [p for p in target_all_predicate if p[0] == t_relation]
    all_background_target_predicates = [p for p in background_all_predicate if p[0] == t_relation]
    
    if target_arity == 2:
        for pred in all_target_predicates + all_background_target_predicates:
            variable_objects[0].add(pred[1])
            variable_objects[1].add(pred[2])
        # Add all objects to rest of variables
        for obj in all_object:
            for variable_name in range(2, variable_number):
                variable_objects[variable_name].add(obj)
    elif target_arity == 1:
        for obj in all_object:
            for variable_name in range(variable_number):
                variable_objects[variable_name].add(obj)
    
    # Save relation (background only, as these are used for rule generation)
    with open(original_data_path+'/relation_entities.dt','wb') as f:
        pickle.dump(background_relation, f)
        f.close()
    
    # Save probability
    with open(original_data_path+'/pro.dt','wb') as f:
        pickle.dump(pro,f)
        f.close()
    with open(original_data_path+'/pro.txt', 'w') as f:
        print(pro,file=f)
        f.close()
    
    logging.info(f"Classifier complete: {len(all_relation_list)} background predicates, target arity: {target_arity}")
    
    return variable_objects, list(background_relation.keys()), background_relation, background_arity_relation, target_arity

def get_all_object(): # This fun is made for country dataset 
    '''
    This function may be uncalled in the whole project. We may delete this function in the later. 
    '''
    all_objects = {}
    objects_1 = [] 
    
    with open(original_data_path+'countries.csv') as c_csv:
        csv_reader = csv.reader(c_csv, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line+=1
                continue
            row = row[0].split(',')[0].lower().replace(' ','_')
            objects_1.append(row)
            line += 1
        c_csv.close()
    print("👉 All countries are:")
    print(objects_1)

    #Assemble all subregions
    objects_2 = []
    with open(original_data_path+'subregions.txt') as f_sub_r:
        sub_r = f_sub_r.read()
        sub_r = sub_r.split('\n')
        objects_2 = sub_r
        f_sub_r.close()    
        
    print("👉 All subrgions are:")
    print(objects_2)
    
    
    objects_3 = []
    with open(original_data_path+'regions.txt') as f_r:
        r = f_r.read()
        r = r.split('\n')
        objects_3 = r
        f_r.close()    
        
    print("👉 All rgions are:")
    print(objects_3)
    
    all_objects[1] = objects_1
    all_objects[2] = objects_2
    all_objects[3] = objects_3
    return all_objects

def get_random_batch_elements(initial_list,substitutation, batch_size, buffer_size):
    '''
    Generate a ramdom batch sized elelmetns from the iterative objects.
    Buffer_size should be larger than the batch_size
    '''
    ini_number = len(initial_list)

    while ini_number < buffer_size:
        try:
            initial_list.append(next(substitutation))
            ini_number += 1
        except StopIteration:
            random.shuffle(initial_list)
            return initial_list, []
    random.seed(4)
    random.shuffle(initial_list)
    random_batch = initial_list[:batch_size]
    rest_list = initial_list[batch_size:]
    
    return random_batch, rest_list


def get_all_valid_predicate_subsample(all_substitution, relation, all_variable_permination, t_relation, t_index, actual_n_substitutation, buffer_size, batch_size):
    '''
    The propositionalization method 2 mentioned by the paper:
    - Check whether there is a preidicate's value is 0 for always, and generated the corresponding valid body predicate. 
    - Different with the original function, this function adapts the subsampling process.  
    '''
    logging.info("The number of substitutation (circle in 'computed memory then stored in disk' manner):")
    logging.info(actual_n_substitutation)
    if actual_n_substitutation >= 0.2e7: # old threshold: 1.5e7
        return -1, -1
    flag_predicates = {}
    relation_index = 0
    relation_name = list(relation.keys())
    target_predicate_index = relation_name.index(t_relation)
    print("Target predicate is")
    print(target_predicate_index)
    for i in (relation):
        flag_predicates[relation_index] = [0] * len(all_variable_permination[i])
        relation_index += 1
    sub_index = 0
    start_time = time.time()

    finished_sub = 0
    initial_list = []
        
    # generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|... 
    while finished_sub < actual_n_substitutation:
        if finished_sub >= actual_n_substitutation:
            print(finished_sub, '/',actual_n_substitutation, 'Finish')
            break
        random_batch, initial_list = get_random_batch_elements(initial_list, all_substitution, batch_size, buffer_size)
        # Doing substitutions
        for i in random_batch:
            current_relation_index = 0
            for relation_name in relation:  
                current_data = relation[relation_name]
                current_predicate_index = 0
                current_permination = all_variable_permination[relation_name]
                for j in current_permination:
                    first_variable = j[0]
                    second_variable = j[1]
                    target_tuple = []
                    target_tuple.append(i[first_variable])
                    target_tuple.append(i[second_variable])
                    target_tuple = tuple(target_tuple)
                    if target_tuple in current_data:
                        flag_predicates[current_relation_index][current_predicate_index] = 1
                    current_predicate_index += 1
                current_relation_index += 1
            
            sub_index+=1
            if sub_index % 500 == 0:
                start_time = show_process(500, actual_n_substitutation, sub_index, start_time, note='[Valid Features with Percent]')
        finished_sub += batch_size

    print("All predicate info is")
    print(flag_predicates)
    valid_index = []
    number_of_predicate = 0
    for i in flag_predicates:
        info = flag_predicates[i]
        for j in info:
            if j == 1 and number_of_predicate != t_index:
                valid_index.append(number_of_predicate)
            number_of_predicate += 1
            
    # Compute whether all boolean variable in the body are zero
    logging.info("Valid predicare index are")
    logging.info(valid_index)
    template = {}
    number_of_predicate = 0
    for i in flag_predicates:
        template[i] = []
        for j in range(len(flag_predicates[i])):
            if flag_predicates[i][j] == 1 and number_of_predicate != t_index:
                template[i].append(j)
            number_of_predicate += 1 
            
    logging.info("Valid template")
    logging.info(template)
    logging.info("Check Valid Predicate Success")
    return valid_index, template

def get_all_valide_predicate(all_substitution, relation, all_variable_permination, t_relation, t_index, number_all_sub = 0):
    '''
    The propositionalization method 2 mentioned by the paper:
    - Check whether there is a preidicate's value is 0 for always, and generated the corresponding valid body predicate. 
    '''
    logging.info('The total number of all substitutions is (compute from memory then disk at once time.):')
    logging.info(number_all_sub)
    if number_all_sub >= 1.5e7:
        return -1, -1
    flag_predicates = {}
    relation_index = 0
    relation_name = list(relation.keys())
    target_predicate_index = relation_name.index(t_relation)
    print("Target predicate is")
    print(target_predicate_index)
    for i in (relation):
        flag_predicates[relation_index] = [0] * len(all_variable_permination[i])
        relation_index += 1
    sub_index = 0
    
    start_time = time.time()
    # generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
    for i in all_substitution:  
        current_relation_index = 0
        for relation_name in relation: #! fix at here 
            current_data = relation[relation_name]
            current_predicate_index = 0
            current_permination = all_variable_permination[relation_name]
            for j in current_permination:
                first_variable = j[0]
                second_variable = j[1]
                target_tuple = []
                target_tuple.append(i[first_variable])
                target_tuple.append(i[second_variable])
                target_tuple = tuple(target_tuple)
                if target_tuple in current_data:
                    flag_predicates[current_relation_index][current_predicate_index] = 1
                current_predicate_index += 1
            current_relation_index += 1

        
        sub_index+=1
        if sub_index % 2000 == 0:
            start_time = show_process(2000, number_all_sub, sub_index,start_time, note='[Find Valid Features]')
            # print(sub_index, flush=True)   
    
    print("All predicate info is")
    print(flag_predicates)
    valid_index = []
    number_of_predicate = 0
    for i in flag_predicates:
        info = flag_predicates[i]
        for j in info:
            if j == 1 and number_of_predicate != t_index:
                valid_index.append(number_of_predicate)
            number_of_predicate += 1
            
    # Compute whether all boolean variable in the body are zero
    print("Valid predicare index are")
    print(valid_index)
    template = {}
    number_of_predicate = 0
    for i in flag_predicates:
        template[i] = []
        for j in range(len(flag_predicates[i])):
            if flag_predicates[i][j] == 1 and number_of_predicate != t_index:
                template[i].append(j)
            number_of_predicate += 1 
            
    print("Valid template")
    print(template)
    return valid_index, template

def show_process(step, all_process, current_process, start_time, note = ''):
    '''
    Print the process and the running time and the left time.
    Parameters: 
    step: the interval to print info 
    all_process: all number of substitutions
    current_process: current number substitution
    start_time: the initial start time.
    '''
    process = float(current_process*100/all_process)
    step_time = float((time.time() - start_time)/60.0)
    rest_running_time = float(((all_process - current_process) / step) * step_time)
    if rest_running_time > 60:
        print(note + ' Percent: %.2f'%process,'%','Current step: %d'%int(current_process),'Total steps: %d'%int(all_process),'Remain running time: %.2f (hours).'%float(rest_running_time/60), end='\r')
    else:
        print(note + ' Percent: %.2f'%process,'%','Current step: %d'%int(current_process),'Total steps: %d'%int(all_process),'Remain running time: %.2f (minutes).'%rest_running_time, end='\r')
    return time.time()
    
    
    
def make_all_predicate(relation, relation_variable_permutations):
    '''
    Generate all first-order features in the dataset.
    '''
    a_p_a_v = []
    for relation_name in relation: #
        current_permination = relation_variable_permutations[relation_name]
        for variable_pair in current_permination:
            a_p_a_v.append(variable_pair)
    return a_p_a_v

def make_small_data(all_substitution, relation, relation_variable_permutations, template, pro, t_relation, train_label,original_data_path, actual_n_substitutation):
    '''
    Generate trainable data when datasets are samll.
    - ALL substitutions based on the all objects;
    - Generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
    '''
    times_substitution = 0
    valid_substitution = 0
    start_time = time.time()
    for single_substitution in all_substitution: 
        relation_index = 0
        # The first constrain mentioned in the paper: if all of the input are 0, then the labels should not be one 
        wrong_flag = True  
        
        all_body_boolean_value = {} # used to record all boolean values of the body of the logic program
            
        for relation_name in relation: #
            y_one_data = []
            data = relation[relation_name] # Read all relational data 
            current_permination = relation_variable_permutations[relation_name]
            for variable_pair in current_permination:
                string_tuple = []
                # begin to save variable arrgement 
                for m in variable_pair:
                    string_tuple.append(single_substitution[m])
                string_tuple = tuple(string_tuple) # like (sue, dinana)...
                if string_tuple in data:
                    # build the current symbolic relational data 
                    current_fact = relation_name+'('+string_tuple[0] +',' + string_tuple[1]+')'
                    prob_value = pro[current_fact]
                    y_one_data.append(prob_value)
                else:
                    y_one_data.append(0)
            
            all_body_boolean_value[relation_index]=y_one_data
            relation_index += 1   
        times_substitution += 1

        # Compute whether all boolean variable in the body are zero
        check_ind = 0
        for i in all_body_boolean_value:
            label_list=[]
            for acq in template[check_ind]:     #Check the tempalte data
                label_list.append( all_body_boolean_value[i][acq])
            if 1 in label_list:
                # ! change to false to check whether accuracy of neural predicate are improved
                wrong_flag = False 
                break
            check_ind += 1
        # target predicate is ancester(x,y) -> ancester(0,1) in the embedding point of view 
        target_tuple = []
        # append the subtitution value corresponding to the variable x
        target_tuple.append(single_substitution[0]) 
        # append the subtitution value corresponding to the variable y
        target_tuple.append(single_substitution[1]) 
        # The symbolic format data correspinding to the variables x and y
        target_tuple = tuple(target_tuple) 

        if target_tuple in relation[t_relation] and wrong_flag == True:
            continue
        
        # TODO STep 1: Add all non-target predicate into train dataset  
        for i in all_body_boolean_value:
            train_label[i].append(all_body_boolean_value[i])

        # TODO Step 2: Add the target predicare firsr-order feature into the train data
        if target_tuple in relation[t_relation]:
            current_fact = t_relation+'('+target_tuple[0] +',' + target_tuple[1]+')'
            current_fact_pro = pro[current_fact]
            train_label[relation_index].append([current_fact_pro])
        else:
            train_label[relation_index].append([0])            
        valid_substitution += 1
        if times_substitution % 5000 == 0:
            start_time = show_process(5000, actual_n_substitutation, times_substitution, start_time, note='[Make Trainable data]')
            
    train_label[len(relation)+1] = train_label[len(relation)]
    
    print("👉 The generated data is:")
    print(len(train_label), len(train_label[0]), len(train_label[1]), len(train_label[2]) )

    final_x = []
    for i in range(len(train_label) - 1):
        final_x.append(train_label[i])

    new_train_label = train_label[len(train_label) - 1]


    x_file = original_data_path + t_relation+'/x_train.dt' 
    y_file = original_data_path + t_relation+'/y_train.dt' 


    with open(x_file, 'wb') as xf:
        pickle.dump(final_x,xf)
        xf.close()
    with open(y_file, 'wb') as yf:
        pickle.dump(new_train_label,yf)
        yf.close()
    return valid_substitution, times_substitution

    
    
def make_large_data(all_substitution, relation, relation_variable_permutations, template, pro, t_relation,original_data_path, actual_n_substitutation, batch_size, buffer_size):
    '''
    Generate large datasets
    - ALL substitutions based on the all objects;
    - Generate all trainable data in the formate of: x-y: [[S(x),S(y),S(z)]...][[1,1,0,0]^|number of predicates|...]
    '''
    datasets_path = original_data_path + t_relation+'/data.tfr' 
    writer = tf.io.TFRecordWriter(datasets_path)
    times_substitution = 0 
    valid_substitution = 0
    start_time = time.time()
    initial_list = []
    while times_substitution < actual_n_substitutation:
        batch_substitution, initial_list = get_random_batch_elements(initial_list, all_substitution, batch_size, buffer_size)
        for single_substitution in batch_substitution: 
            times_substitution += 1
            relation_index = 0
            # The first constrain mentioned in the paper: if all of the input are 0, then the labels should not be one 
            wrong_flag = True  
            
            all_body_boolean_value = {} # used to record all boolean values of the body of the logic program
                
            for relation_name in relation: #
                y_one_data = []
                data = relation[relation_name] # Read all relational data 
                current_permination = relation_variable_permutations[relation_name]
                for variable_pair in current_permination:
                    string_tuple = []
                    for m in variable_pair:
                        string_tuple.append(single_substitution[m])
                    string_tuple = tuple(string_tuple) # like (sue, dinana)...
                    if string_tuple in data:
                        # build the current symbolic relational data 
                        current_fact = relation_name+'('+string_tuple[0] +',' + string_tuple[1]+')'
                        prob_value = pro[current_fact]
                        y_one_data.append(prob_value)
                    else:
                        y_one_data.append(0)
                
                all_body_boolean_value[relation_index]=y_one_data
                relation_index += 1   
            

            # Compute whether all boolean variable in the body are zero
            check_ind = 0
            for i in all_body_boolean_value:
                label_list=[]
                for acq in template[check_ind]:     #Check the tempalte data
                    label_list.append( all_body_boolean_value[i][acq])
                if 1 in label_list:
                    # ! change to false to check whether accuracy of neural predicate are improved
                    wrong_flag = False 
                    break
                check_ind += 1
            # target predicate is ancester(x,y) -> ancester(0,1) in the embedding point of view 
            target_tuple = []
            # append the subtitution value corresponding to the variable x
            target_tuple.append(single_substitution[0]) 
            # append the subtitution value corresponding to the variable y
            target_tuple.append(single_substitution[1]) 
            # The symbolic format data correspinding to the variables x and y
            target_tuple = tuple(target_tuple) 

            if target_tuple in relation[t_relation] and wrong_flag == True:
                continue
            
            body = []
            head = []
            
            # TODO STep 1: Add all non-target predicate into train dataset  
            for i in all_body_boolean_value:
                for j in all_body_boolean_value[i]:
                    body.append(np.float64(j))
            
            # TODO Step 2: Add the target predicare firsr-order feature into the train data
            if target_tuple in relation[t_relation]:
                current_fact = t_relation+'('+target_tuple[0] +',' + target_tuple[1]+')'
                current_fact_pro = pro[current_fact]
                head.append([np.float64(current_fact_pro)])
                body.append(np.float64(current_fact_pro))
            else:
                head.append([np.float64(0)])
                body.append(np.float64(0))
            
            tf_example = first_order_features_example(body,head)
            writer.write(tf_example.SerializeToString())
                
            valid_substitution += 1
            if times_substitution % 5000 == 0:
                start_time = show_process(5000, actual_n_substitutation, times_substitution, start_time, note='[Make trainable data in Large mode]')
            
    writer.close()
    return valid_substitution, times_substitution

            
def main(dataset, t_relation = '', path_name = '' , original_data_path= '', variable_depth=1, large = False, sub_per = 0.02, buffer_size = 9000, random_size = 5000):

    relation_name = []
    variable_number = variable_depth + 2 
    
    # Check if separate train.pl and facts.pl files exist
    train_path = original_data_path + 'train.pl'
    facts_path = original_data_path + 'facts.pl'
    
    if os.path.exists(train_path) and os.path.exists(facts_path):
        # Use new classifier that separates background knowledge from target edges
        logging.info('Found train.pl and facts.pl - using separate file loading')
        variable_objects, relation_name, relation , arity_relation, target_arity = classifier_with_separate_files(
            facts_path, train_path, variable_number, original_data_path+t_relation, t_relation)
    else:
        # Fall back to original single-file classifier
        logging.info('Using original single-file loading from .nl file')
        variable_objects, relation_name, relation , arity_relation, target_arity = classifier(
            original_data_path+t_relation +'.nl', variable_number, original_data_path+t_relation, t_relation)
    
    logging.info('Begin generating data with %s as head predicate.'%t_relation)

    # Assemb all the countries 
    variable_class = {}     # cityLocIn(x,y)  x->countries y->region z->subregion, cities, and regions. 
    # Then possible predicate include: locatedIn[c_2,c_2],locatedIn[c_1,c_2], locatedIn[c_1,c_3], locatedIn[c_2,c_3] neighbor[c_1,c_1]
    for i in variable_objects:
        variable_class[i] = list(variable_objects[i])
        
    ALL_objects = []
    for i in variable_class:
        l = variable_class[i]
        for j in l:
            if j not in ALL_objects:
                ALL_objects.append(j)

        
    entities_dic_name_number = {}  #sue:0 
    
    for i in ALL_objects:
        entities_dic_name_number[i] = ALL_objects.index(i) 
    print('👉 The second dictionary is', entities_dic_name_number)

    print("👉 All relation in the dataset")
    print(relation)
    
    # train the neural predicates
    variable_value = variable_class.values()
    # Compute the number of all substitutation
    all_substitution_number = 1
    for i in variable_value:
        all_substitution_number *= len(i)
    if large == False:
        all_substitution = list(itertools.product(*variable_value))
    else:
        all_substitution = itertools.product(*variable_value)
    
    if large == True:
        actual_n_substitutation = int(all_substitution_number * sub_per)
    else:
        actual_n_substitutation = all_substitution_number
    
    
    # The possible predicate correspodnnig each predicates 
    relation_variable_permutations = {}
    for i in arity_relation:
        if arity_relation[i] == 1:
            empty_list = []
            for variable_index in range(variable_number):
                empty_list.append((variable_index, variable_index))
            relation_variable_permutations[i] = empty_list
        else:
            relation_variable_permutations[i] = list(itertools.permutations(range(variable_number), 2 ))

    # The index of the target predicate 
    find_index = 0
    for i in range(len(relation_name)):
        for j in relation_variable_permutations[relation_name[i]]:
            if arity_relation[t_relation] == 1 and j == (0,0) and i == relation_name.index(t_relation):
                t_index = find_index
                target_predicate = t_relation+"(X,X)"
            elif arity_relation[t_relation] == 2 and j == (0,1) and i == relation_name.index(t_relation):
                t_index = find_index
                target_predicate = t_relation+"(X,Y)"
            find_index += 1
                
    print('The index of target predicate are:')
    print(t_index)

    print('👉 all variables')
    print(relation_variable_permutations)
    with open(path_name+'data/' + t_relation+'/relation_variable.dt','wb') as f:
        pickle.dump(relation_variable_permutations, f)
        f.close()
    with open(path_name+'data/' + t_relation+'/relation_variable.txt','w') as f:
        print(relation_variable_permutations, file=f)
        f.close()    
    

    train_label = {}
    for i in range(len(relation)):
        train_label[i] = []

    # Used to store the Boolean label for the taget relational predicate
    target_predicate_index = len(relation)  
    train_label[target_predicate_index] = []
        
    print("👉 Pre-operation on the label dataset:", train_label)
    

    try:
        with open(path_name + 'data/'+t_relation+'/valid_index.dt', "rb") as f:
            res = pickle.load(f)
            valid_index = res['valid_index']
            template = res['template']
            print(res)
            f.close()
    except FileNotFoundError:
        if large == True:
            valid_index, template = get_all_valid_predicate_subsample(all_substitution, relation, relation_variable_permutations, t_relation, t_index, actual_n_substitutation, buffer_size, random_size)
        else:
            valid_index, template= get_all_valide_predicate(all_substitution=all_substitution,relation = relation, all_variable_permination=relation_variable_permutations, t_relation = t_relation, t_index = t_index, number_all_sub=actual_n_substitutation )
        
        if valid_index == -1 and template == -1:
            return -1,-1,-1
        res = {}
        res['valid_index'] = valid_index
        res['template'] = template
        with open(path_name + 'data/'+t_relation+'/valid_index.dt', 'wb') as f:
            pickle.dump(res, f)
            f.close()
        with open(path_name + 'data/'+t_relation+'/valid_index.txt','w') as f:
            print(str(res), file = f)
            f.close()
        print("Save template succeess")

    if large == True:
        all_substitution = itertools.product(*variable_value)

    # open the probabilistic table 
    with open(path_name+'data/'+ t_relation+'/pro.dt',"rb") as f:
        pro = pickle.load(f)
        f.close()
    
    # Make trainable data
    if large == True: 
        valid_substitution, times_substitution =  make_large_data(all_substitution, relation, relation_variable_permutations, template, pro, t_relation, original_data_path, actual_n_substitutation, random_size, buffer_size)
    else:
        valid_substitution, times_substitution = make_small_data(all_substitution, relation, relation_variable_permutations, template, pro, t_relation, train_label, original_data_path, actual_n_substitutation)

    # Print some meta-information into datasets
    with open(original_data_path + t_relation+'/number_valid_substitution.txt', 'w') as vf:
        vf.write(str(valid_substitution))
        logging.info(str(valid_substitution)+' (valid)/(all) '+str(times_substitution))
        vf.close()
    
    # Save all variable permutations into the corresponding file
    print("👉 Begin to save all variable permutations in all predicates...")
    a_p_a_v = make_all_predicate(relation, relation_variable_permutations)
    with open(original_data_path + t_relation+'/predicate.txt', 'w') as vf:
        vf.write(str(a_p_a_v))
        vf.close()
    with open(original_data_path + t_relation+'/predicate.list', 'wb') as vf:
        pickle.dump((a_p_a_v), vf)
        vf.close()
    print("👉 Save success")
    
    
    all_obj_file = original_data_path + t_relation+'/all_ent.dt'
    with open(all_obj_file, 'wb') as f:
        pickle.dump(ALL_objects,f)
        f.close()    
        
    # return all the objects in the task 
    return ALL_objects , target_arity, target_predicate


    


def gen(dataset, target_relation_name, variable_depth, large, sub_per, buffer_size, random_size):
    original_data_path = 'deepDFOL/'+dataset+'/data/'
    path_name = 'deepDFOL/'+dataset+'/'
    _, target_arity, head_pre = main(dataset, t_relation=target_relation_name, path_name=path_name,original_data_path=original_data_path, variable_depth=variable_depth, large = large, sub_per=sub_per, buffer_size=buffer_size, random_size=random_size)
    if target_arity == -1 and head_pre == -1:
        return -1, -1
    print("-------Generating Data Success!-------")
    return target_arity, head_pre


if __name__ == "__main__":

    args = sys.argv
    dataset = args[1] 
    target_relation_name = args[2]
    print(dataset, target_relation_name)
    original_data_path = 'deepDFOL/'+dataset+'/data/'
    path_name = 'deepDFOL/'+dataset+'/'
    # target_relation_name = 'locatedIn'
    # data_index = 'S1'

    # classifier(original_data_path+dataset+'.nl', 3)


    ALL_objects = main(dataset,t_relation= target_relation_name, path_name = path_name, original_data_path = original_data_path)
    print("👉 The numbet of all objects in the task:") # 276
    print(len(ALL_objects))