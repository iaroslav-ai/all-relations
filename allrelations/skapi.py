"""
Scikit-Learn based api for estimation of all 1 -> 1 or n -> 1
relations in the dataset.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin

from string import digits


def preprocess_dataset(dataframe, missing_values =('', ' ', '?', 'NaN')):
    """
    Converts pandas dataframe representing a dataset
    to a set of concepts, as well as to additional
    respondent information if such information is given.

    Parameters
    ----------
    dataframe: DataFrame, contains the dataset to be processed.
        The names of the columns should be given in the
        dataset, as they define the set of concepts,
        and what is considered a feature. In particular,
        every column should be named as follows:
        [concept]_[id],
        where concept denotes the name of the concept,
        and id denotes a particular indicator for a concept.
        Underscores are not allowed in name or id, and can
        break the program if present.
        User features are specified using two special names
        for concepts:
        - respnum: numerical feature describing respondent
        - respcat: categorical feature describing respondent
        Example dataset is given below:
        respnum_age | respcat_edu | Q1_a | Q1_b | Q2_a |
        ------------|-------------|------|------|------|
        25          | College     | 1    | 2    | 5    |
        34          | University  | 2    | 3    | 4    |

    missing_values: tuple of string
        Describes possibilities for missing values. If in your
        dataset missing values are represented different from
        the defaults, specify your own values.

    Returns
    -------
    result: concepts, respdata
        - concepts: dictionary of numpy arrays, where every
        numpy array contains indicator values for a single
        concept.
        - respdata: dictionary with additional data about
        respondent. Such data is a numpy array with
        numerical values of respondent features, and one
        hot encoded values of the categorical values.
    """

    concepts = {}
    respdata = []

    for cname in dataframe.columns:
        if not '_' in cname:
            raise ValueError(
                'Expected dataset with column names in the format'
                '[concept]_[indicator], but got %s' % cname
            )

        concept = cname.split("_")[0]
        column = dataframe[cname]

        if concept == 'respcat':
            # one hot encode the categories
            respdata.append(pd.get_dummies(column).values)
            continue

        for empty in missing_values:
            # in sklearn, np.nan is missing value
            column = column.replace(empty, np.nan)

        column = column.values  # convert to numpy array

        if concept == 'respnum':
            respdata.append(column)
            continue

        if concept in concepts:
            concepts[concept].append(column)
        else:
            concepts[concept] = [column]

    concepts = {k: np.column_stack(v).astype('float') for k, v in concepts.items()}
    respdata = np.column_stack(respdata) if respdata else None
    return concepts, respdata


def pandas_to_concepts(data):
    """
    Converts pandas dataframe to set of all concepts.

    Parameters
    ----------
    data: DataFrame, contains the dataset

    Returns
    -------
    result: dict with all concepts
    """

    result = {}

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Parameter data should be of type DataFrame.")

    for c in data.columns:
        x = data[c]

        # select only the name of concept here
        concept = c
        while concept[-1] in digits:
            concept = concept[:-1]

        if not concept in result:
            result[concept] = np.array(x)[:, np.newaxis]
        else:
            result[concept] = np.column_stack([result[concept], x])

    return result


def make_regressor_grid(model_subset=None):
    """
    Generate the necessary estimator model class for grid search.

    Parameters
    ----------

    * model_subset [string, default=None]
        Whether to use a named model_subset of model classes for fitting.
        Feasible options are:
        - None: use all the models available
        - 'linear': use only linear models

    Returns
    -------
    model: GridSearchCV instance, estimator class that can be applied
        to features to learn the relationship.
    """
    estimator = Pipeline([
        ('imputer', Imputer()),
        ('scale', StandardScaler()),
        ('model', GradientBoostingRegressor()),
    ])

    # search spaces for different model classes
    model_choices = {
        "lasso": {
            'model': [Lasso()],
            'model__alpha': [10 ** i for i in np.linspace(-6, 6, 11)],
        },
        "ann": {
            'model': [MLPRegressor(solver='adam')],
            'model__hidden_layer_sizes': [[n_neurons for _ in range(n_layers)] for n_neurons in [128, 64, 32, 16] for n_layers in [2, 1]],
            'model__max_iter': [32, 16, 8],
            'model__learning_rate_init': [0.0001, 0.001]
        },
        "knn": {
            'model': [KNeighborsRegressor()],
            'model__n_neighbors': range(1, 100, 5),
        },
        "gbrt": {
            'model': [GradientBoostingRegressor()],
            'model__n_estimators': [2 ** i for i in range(1, 9)],
            'model__learning_rate': [2 ** i for i in range(-10, 0)],
        },
        "tree" : {
            'model': [DecisionTreeRegressor()],
            'model__max_depth': range(1, 20),
            'model__min_samples_split': [2 ** i for i in range(-20, -1)],
        }
    }

    # user can specify subset of models to be used
    if model_subset is None:
        choices = model_choices.keys()
    else:
        choices = model_subset

    spaces = [model_choices[k] for k in choices]

    # this class search over all parameter spaces for parameter
    # combination which yields the best validation loss
    model = GridSearchCV(
        estimator=estimator,
        param_grid=spaces, # knn, gbrt, dectree
        n_jobs=-1,
        verbose=0,
    )

    return model


def mapping_power(X, Y, models_subset=None):
    """
    Evaluate the strength of relation from X to Y.

    Parameters
    ----------

    * models_subset [string, default=None]
        See same argument of the make_regressor function.
    * X [np.ndarray, shape=(n_samples, n_features)]
        Array of input concept observations. Missing values
        are denoted with nan's.

    Returns
    -------
    model: GridSearchCV instance, estimator class that can be applied
        to features to learn the relationship.
    """
    # evaluate all the models in cross - validation fashion
    y_true, y_pred = [], []

    # iterate over all columns
    for y in Y.T:
        I = ~np.isnan(y) # select rows where outputs are not missing

        yp = cross_val_predict(make_regressor_grid(models_subset), X[I], y[I])

        y_true.append(y[I])
        y_pred.append(yp)

    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)

    # calculate bootstrap on rmsea
    #n_iter, p = 1000000, 0.000001
    #print(bs.bootstrap((yt-yp)**2, stat_func=bs_stats.mean, alpha=p, num_iterations=n_iter, iteration_batch_size=10000))
    #print(bs.bootstrap((yt-np.mean(yt))**2, stat_func=bs_stats.mean, alpha=p, num_iterations=n_iter, iteration_batch_size=10000))

    # compare the cross - validation predictions for all columns
    score = r2_score(
        yt,
        yp
    )

    return score


def all_1_to_1(concepts, prefix=None, models_subset=None):
    """
    Finds all one to one relations within the set of concepts.

    Parameters
    ----------
    concepts : dict, where every element is a numpy array of shape
        [n_samples, n_features']. Training data, where n_samples in
        the number of samples of records describing particular
        concept. n_features' can be different for different concepts.

    prefix : array-like, shape = [n_samples, n_features]
        Features that apply to every concept.

    models_subset: string or None
        Whether to use a subset of models for estimation of mapping
        power. For feasible options, see the similar parameter of
        the `mapping_power` function.


    Returns
    -------
    result : array of [set a, set b, float]
        Returns estimate of test accuracy for how single concept in
        set b can be estimated from single concept in set a. All
        combinations of single concepts are considered. The value
        of float represents how well the concept can be estimated.

    """

    names = concepts.keys()
    result = []

    for A in tqdm(names):
        for B in names:

            if A == B:
                continue

            print(A, "->", B)

            X = concepts[A]

            if prefix is not None:
                X = np.column_stack([prefix, X])

            Y = concepts[B]
            Y = Y.astype('float')

            local_result = [[A], [B]]

            # get score for estimation of non - missing values
            score = mapping_power(X, Y, models_subset=models_subset)
            local_result.append(score)

            result.append(local_result)


    # sort from highest weight to the lowest weight
    result.sort(reverse=True, key=lambda x: x[-1])

    return result


def concept_subset(concepts, names, prefix = None):
    selection = [concepts[n] for n in names]

    if prefix is not None:
        selection += prefix

    result = np.column_stack(selection)
    return result


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """This transformer simply selects a subset of columns from
    the dataset. Useful for feature selection."""
    def __init__(self, index_set, active=True):
        self.index_set = index_set
        self.active = active

    def transform(self, X, y=None):
        if self.active:
            return X[:, self.index_set]
        else:
            return X[:, 0:0] # returns an empty column

    def fit(self, X, y=None, **fit_params):
        return self

from sklearn.pipeline import FeatureUnion
from skopt.space import Categorical, Real, Integer
from skopt.utils import point_asdict, point_aslist
from searchgrid import set_grid, build_param_grid
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def minimize_inputs(concepts, target, prefix=None, models_subset=None, discount=0.95, max_iter=32, optimizer=None):
    feature_selection_steps = []  # feature extraction steps
    pipeline_steps = []  # overall data science pipeline steps

    X = {c:v for c,v in concepts.items() if c != target}
    y = concepts[target]

    idx = 0
    X_num = []
    N_concepts = len(X)

    for concept in X:
        Xcol = X[concept]
        M = Xcol.shape[-1]  # number of columns in concept

        # extend the total dataset
        X_num.append(Xcol)

        # make a selector
        selector_indicies = tuple(range(idx, idx+M))
        selector = ColumnsSelector(selector_indicies)  # this class can select subset of features
        # set the desired options
        selector = set_grid(selector, active=[Categorical([False, True])])
        feature_selection_steps.append((concept, selector))
        idx += M

    # make a proper dataset
    X_num = np.column_stack(X_num)

    # drop empty values in target
    I = ~np.any(np.isnan(y), axis=-1)
    X_num = X_num[I]
    y = y[I]

    X_train, X_test, y_train, y_test = train_test_split(X_num, y)

    # this will extract the features, if necessary
    features = FeatureUnion(feature_selection_steps)

    pipeline_steps.append(('features', features))
    pipeline_steps.append(('impute', SimpleImputer()))
    pipeline_steps.append(('scale', StandardScaler()))

    estimator = set_grid(
        GradientBoostingRegressor(),
        loss=[Categorical(['ls', 'lad', 'huber'])],
        learning_rate=[Real(1e-4, 1.0, 'log-uniform')],
        n_estimators=[Integer(32, 512)],
        max_depth=[Integer(1, 5)]
    )

    pipeline_steps.append(('model', estimator))

    def used_concepts(params):
        selection = []
        for p in params:
            if "features" in p:
                if params[p]:
                    selection.append(p.split("__")[1])

        return selection

    # objective to be minimized; depends on whether all features are used or only subset
    class objective:
        def __init__(self, select_subset, base_score, model, search_spaces):
            self.select_subset = select_subset
            self.base_score = base_score
            self.model = model
            self.search_spaces = search_spaces

        def __call__(self, p):
            # task: maximize the score
            score = 0.0
            params = point_asdict(self.search_spaces, p)

            try:

                self.model.set_params(**params)
                all_scores = []
                for i in range(y.shape[-1]):
                    scores = cross_val_score(self.model, X_train, y_train[:, i], n_jobs=1, cv=5)
                    all_scores.append(scores)

                avg_score = np.mean(all_scores)

                if self.select_subset:
                    if avg_score > self.base_score * discount:
                        # count how many concepts have been used
                        N_selection = len(used_concepts(params))

                        # if same size of concept subset is used, prefer the one with higher score
                        score = (N_concepts - N_selection + avg_score) / N_concepts
                else:
                    score = avg_score

            except BaseException as ex:
                pass

            return -score

    # First experiment: select hyperaprameters only, with subset of features
    model = Pipeline(pipeline_steps[1:])
    search_spaces = {k:v[0] for k, v in build_param_grid(model).items()}
    search_dims = point_aslist(search_spaces, search_spaces)

    obj = objective(False, None, model, search_spaces)
    solution = optimizer(obj, search_dims, n_calls=max_iter, n_random_starts=min(max_iter, 10))

    # Now minimize subset of concepts used
    model = Pipeline(pipeline_steps)
    search_spaces = {k:v[0] for k, v in build_param_grid(model).items()}
    search_dims = point_aslist(search_spaces, search_spaces)

    obj.base_score = -solution.fun
    obj.select_subset = True
    obj.model = model
    obj.search_spaces = search_spaces

    solution = optimizer(obj, search_dims, n_calls=max_iter, n_random_starts=min(max_iter, 10))
    best_params_ = point_asdict(search_spaces, solution.x)
    model.set_params(**best_params_)

    weights = []
    for i in range(y.shape[-1]):
        model.fit(X_train, y_train[:, i])
        weight = model.score(X_test, y_test[:, i])
        weights.append(weight)

    selection = used_concepts(best_params_)
    weight = np.mean(weights)

    return weight, selection


def all_n_to_1(concepts, prefix=None, models_subset=None, discount=0.95, max_iter=32, optimizer=None):
    """
    Finds all one to one relations within the set of concepts.

    Parameters
    ----------
    concepts : dict, where every element is a numpy array of shape
        [n_samples, n_features']. Training data, where n_samples in
        the number of samples of records describing particular
        concept. n_features' can be different for different concepts.

    prefix : array-like, shape = [n_samples, n_features]
        Features that apply to every concept.

    models_subset: string or None
        Whether to use a subset of models for estimation of mapping
        power. For feasible options, see the similar parameter of
        the `mapping_power` function.

    discount : float
        Fraction of r^2 with all concepts that should be preserved
        with the subset of concepts.

    max_iter : int
        Number of iterations used in black box optimization algorithm.

    optimizer : callable or None
        Optimizer to use for optimization of set of relations and parameters

    Returns
    -------
    result : array of [set a, set b, float]
        Returns estimate of test accuracy for how single concept in
        set b can be estimated from minimum set of all concepts (set a)
        such that the total accuracy is not worse than discount*100%
        compared to when all concepts are used.
        The value of float represents how well the concept can be estimated -
        0.0 is worst, 1.0 is best.

    """
    from skopt import gp_minimize

    if optimizer is None:
        optimizer = gp_minimize

    names = set(concepts.keys())
    full_results = []

    for B in tqdm(names):
        # first try with all concepts
        weight, found_concepts = minimize_inputs(concepts, B, discount=discount, max_iter=max_iter, optimizer=optimizer)
        full_results.append([found_concepts, [B], weight])

    return full_results
