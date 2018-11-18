"""
Script to reproduce results in submission to ECIS 2018.
"""

import os
from allrelations.interface import extract_n_to_1

dataset_path = os.path.join('datasets', 'wiki4he', 'wiki.csv')

for model in ['gbrt']:
    for use_resp_data in [False]:
        for discount in [0.2, 0.5, 0.8, 1.0]:
            results_path = os.path.join('experimental_results', 'nto1_2.', 'wiki4he_%s' % discount)
            extract_n_to_1(dataset_path, results_path, model, use_resp_data, max_iter=64, discount=discount)