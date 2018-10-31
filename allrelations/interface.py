"""
Simplified API for extraction of relations.
"""

import json
import pandas as pd
import os

from allrelations.skapi import preprocess_dataset, all_1_to_1, all_n_to_1
from allrelations.visualization import render_relations


def extract_1_to_1(dataset, saveto, model, use_resp_data=False):
    # read the CSV file of the dataset
    data = pd.read_csv(dataset)

    # separate dataset into concepts and user information
    concepts, respdata = preprocess_dataset(data)

    if not use_resp_data:
        respdata = None  # ignore respondent data if required

    if isinstance(model, str):
        model = [model]  # single model is being tried

    # extract all 1 to 1 relations
    relations = all_1_to_1(concepts, prefix=respdata, models_subset=model)

    # name of dataset without .csv at the end
    dfname = os.path.basename(dataset)
    if dfname.endswith('.csv'):
        dfname = dfname[:-4]

    # output name: dataset file name + model name [+ userfeatures] .json
    result_name = dfname + '_' + "-".join(model)
    if use_resp_data:
        result_name += '_respdata'

    if not os.path.exists(saveto):
        os.mkdir(saveto)

    result_path = os.path.join(saveto, result_name + '.json')
    json.dump(relations, open(result_path, 'w'), indent=2)
    render_relations(result_path)


def extract_n_to_1(dataset, saveto, model, use_resp_data=False, discount=0.8, max_iter=32, optimizer=None):
    # read the CSV file of the dataset
    data = pd.read_csv(dataset)

    # separate dataset into concepts and user information
    concepts, respdata = preprocess_dataset(data)

    if not use_resp_data:
        respdata = None  # ignore respondent data if required

    if isinstance(model, str):
        model = [model]  # single model is being tried

    # extract all 1 to 1 relations
    relations = all_n_to_1(concepts,
                           prefix=respdata, models_subset=model,
                           discount=discount, max_iter=max_iter, optimizer=optimizer)

    # name of dataset without .csv at the end
    dfname = os.path.basename(dataset)
    if dfname.endswith('.csv'):
        dfname = dfname[:-4]

    # output name: dataset file name + model name [+ userfeatures] .json
    result_name = dfname + '_' + "-".join(model)
    if use_resp_data:
        result_name += '_respdata'

    if not os.path.exists(saveto):
        os.mkdir(saveto)

    result_path = os.path.join(saveto, result_name + '.json')
    json.dump(relations, open(result_path, 'w'), indent=2)
    render_relations(result_path)