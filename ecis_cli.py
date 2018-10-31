"""Script used for experimentaion with ECIS submission.

Usage:
  ecis_cli.py --help
  ecis_cli.py --model=<class> --discount=<float> --max_iter=<int> [--userfeatures]

Options:
  --model=<class>        A class of models to use. Supported classes
                         are ['lasso', 'ann', 'knn', 'gbrt', 'tree'].
                         The options correspond to the following
                         models:
                         'lasso' stands for Lasso Regression,
                         'ann' stands for Artificial Neural Network,
                         'knn' stands for k Nearest Neighbors,
                         'gbrt' is Gradient Boosting Regression Trees,
                         'tree' is Regression Decision tree model.
                         Warning: training ANNs takes considerably
                         more time than for other models.
  --discount=<float>     Fraction of accuracy with all inputs to maintain.
                         Controls the complexity of the relations.
  --max_iter=<int>       Max iterations to run for.
  --userfeatures         Whether to use additional user features, specified
                         in the dataset, in addition to the features of the
                         concepts themselves.
  -h --help              Show this screen.
  --version              Show version.

"""


from allrelations.interface import extract_n_to_1
from docopt import docopt
import os

if __name__ == "__main__":
    arguments = docopt(__doc__, version='Nov 2018')

    dataset_path = os.path.join('datasets', 'wiki4he', 'wiki.csv')
    print(arguments)
    model = str(arguments['--model'])
    use_resp_data = arguments['--userfeatures']
    discount = float(arguments['--discount'])
    max_iter = int(arguments['--max_iter'])

    results_path = os.path.join('experimental_results', 'nto1', 'wiki4he_%s' % discount)
    extract_n_to_1(dataset_path, results_path, model, use_resp_data, max_iter=max_iter, discount=discount)