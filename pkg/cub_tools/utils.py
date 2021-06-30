import pickle
import torch
import json
from itertools import tee

def save_model_dict(model, PATH):
    torch.save(model.state_dict(), PATH)

def save_model_full(model, PATH):
    torch.save(model, PATH)
    
def save_pickle(pkl_object, fname):
    '''
    Utility function to save a python object to pickle file.

    E C Morris
    ed.morris@malvernpanalytical.com
    Malvern Panalytical 
    (c) 2020
    '''
    pkl_file = open(fname, 'wb')
    pickle.dump(pkl_object, pkl_file)
    pkl_file.close()
    

def unpickle(fname):
    '''
    Utility function to load a python object from pickle file.

    E C Morris
    ed.morris@malvernpanalytical.com
    Malvern Panalytical 
    (c) 2020
    '''
    file = open(fname,'rb')
    obj = pickle.load(file)
    file.close()
    return obj


def load_json(json_fname, verbose=False):
    '''
    Function to load data from JSON file into an object and return that object.
    Exits file nicely.
    
    E C Morris
    ed.morris@malvernpanalytical.com
    Malvern Panalytical 
    (c) 2020
    '''
    with open(json_fname) as json_file:
        data = json.load(json_file)
        if verbose:
            print('Loaded JSON file: ', json_fname)
            
            # If data object is dictionary, report some basic structure.
            if type(data) == dict:
                print('Imported JSON object is dictionary')
                print('Keys contained within file::')
                for i, key in enumerate(data.keys()):
                    print(i,'::',key)
            
            # If data object is list, report first few entires to list.
            elif type(data) == list:
                print('Imported JSON object is list')
                print('Looping over first few entires')
                if len(data) > 5:
                    n_entries=5
                else:
                    n_entries = len(data)
                for i in np.arange(0,n_entries,1):
                    print(data[i])
    json_file.close()
    return data


def save_json(json_dict, json_fname, ensure_ascii=True):
    '''
    Save a dictionary to JSON file.

    ensure_ascii forces JSON file to convert to ASCII.

    E C Morris
    ed.morris@malvernpanalytical.com
    Malvern Panalytical 
    (c) 2020
    '''
    with open(json_fname, 'w') as json_file:
        json.dump(json_dict, json_file, ensure_ascii=ensure_ascii)
        json_file.close()

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return a, b