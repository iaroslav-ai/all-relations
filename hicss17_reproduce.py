"""
Script to reproduce results in
"Maass, W. & Shcherbatyi, I. Data-Driven, Statistical Learning Method for
Inductive Confirmation of Structural Models, Hawaii International Conference
on System Sciences (HICSS), 2017."
Notice: in this script, r^2 is used instead of IRG. For original values in IRG,
see the experimental results folder, wiki subfolder, in v1 folder there.
"""

import os
from allrelations.interface import extract_1_to_1

dataset_path = os.path.join('datasets', 'wiki4he_1', 'wiki.csv')
results_path = os.path.join('experimental_results', 'wiki4he_1')

for model in ['lasso', 'tree', 'knn', 'gbrt', 'ann']:
    extract_1_to_1(dataset_path, results_path, model)