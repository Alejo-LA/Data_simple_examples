"""
Basic loading functions
"""

import json
# import ast
import pickle
import pandas as pd


####### JSON FILES
def json_loader(filepath):
    """ This function loads a json file as a python dictionary"""
    with open(filepath, 'r') as file:
        my_dict = json.load(file)
    return my_dict


####### TXT FILES
def txt_saver(p, name):
    """ This function saves a list/float/int as a txt"""
    with open("Data/Dynamic_Data/" + name + ".txt", "wb") as fp:
        pickle.dump(p, fp)


def txt_loader(path):
    """ This function retrieves a list/float/int saved in a txt"""
    with open(path, "rb") as fp:
        p = pickle.load(fp)
    return p


####### PKL FILES
def pkl_load(filepath):
    """
    This function loads a pkl file
    Input:
        - file path: str
    Output:
        - df: Pandas DataFrame with the pkl's content
    """
    with open(filepath, 'rb') as f:
        f = pickle.load(f)
    return f


def pkl_saver(p, path):
    """ This function saves a list/float/int as a txt"""
    with open(path + ".pkl", "wb") as fp:
        pickle.dump(p, fp)


####### CSV FILES
def load_csv(filepath):
    """
    This function to loads csv format
    Input:
        - file path: a string with the path to a csv file
    Output:
        - df: Pandas DataFrame with the csv's content
    """
    with open(filepath, 'rb') as f:
        df = pd.read_csv(f)
    return df
