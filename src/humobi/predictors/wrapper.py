import numpy as np
import pandas as pd
from src.humobi.misc.utils import to_labels
from tqdm import tqdm

tqdm.pandas()
from src.humobi.predictors.markov import MarkovChain
from src.humobi.predictors.sparse import Sparse, Sparse_old
from sklearn.model_selection import TimeSeriesSplit
import itertools
from itertools import repeat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import concurrent.futures as cf
import warnings


def iterate_random_values(S, n_check):
    """
    Takes a random combination of items of a given size.

    Args:
        S: The dictionary of items
        n_check: Size of combination

    Returns:
        A random combination
    """
    keys, values = zip(*S.items())
    combs = [dict(zip(keys, row)) for row in itertools.product(*values)]
    combs = np.random.choice(combs, n_check)
    return combs


class Splitter():
    """
    This is a Splitter class responsible for data preparation for shallow machine learning-based models from sklearn library.

    Args:
        trajectories_frame: TrajectoriesFrame class object
        split_ratio: The ratio of training data
        horizon: Window size - determines how many previous symbols are considered by the model when predicting
        n_splits: The number of splits for cross-validation process
    """

    def __init__(self, split_ratio, horizon=1, n_splits=1):
        """
        Class initialisation. Calls data splitting routine, hence after initialisation data is already prepared.
        """
        self._train_ratio = 1 - split_ratio
        if horizon < 1:
            raise ValueError("Horizon value has to be a positive integer")
        self._horizon = horizon
        self._n_splits = n_splits
        self.cv_data = []

    @property
    def test_ratio(self):
        return self._train_ratio

    @property
    def horizon(self):
        return self._horizon

    def _test_split(self, X, Y):
        """
        Splits test data into training and testing sets. Uses windowed data.

        Args:
            X: Windowed features
            Y: Windowed targets

        Returns:
            Training and testing sets of data
        """
        train_frame_X = X.groupby(level=0).progress_apply(lambda x: x.iloc[:round(len(x) * self.test_ratio)]).droplevel(
            1)
        train_frame_Y = Y.groupby(level=0).progress_apply(lambda x: x.iloc[:round(len(x) * self.test_ratio)]).droplevel(
            1)
        test_frame_X = X.groupby(level=0).progress_apply(lambda x: x.iloc[round(len(x) * self.test_ratio):]).droplevel(
            1)
        test_frame_Y = Y.groupby(level=0).progress_apply(lambda x: x.iloc[round(len(x) * self.test_ratio):]).droplevel(
            1)
        return train_frame_X, train_frame_Y, test_frame_X, test_frame_Y

    def _cv_split(self, frame_X, frame_Y, n_splits=5):
        """
        Splits training data into the training and validation sets using cross-validation approach.

        Args:
            frame_X: Training features
            frame_Y: Training targets
            n_splits: The number of splits to be applied
        """
        for n in range(1, n_splits + 1):
            train_set_X = frame_X.groupby(level=0).apply(
                lambda x: x.iloc[:round(x.shape[0] * (n / (n_splits + 1)))]).droplevel(0)
            train_set_Y = frame_Y.groupby(level=0).apply(
                lambda x: x.iloc[:round(x.shape[0] * (n / (n_splits + 1)))]).droplevel(0)
            val_set_X = frame_X.groupby(level=0).apply(lambda x: x.iloc[round(x.shape[0] * (n / (n_splits + 1))):round(
                x.shape[0] * ((n + 1) / (n_splits + 1)))]).droplevel(0)
            val_set_Y = frame_Y.groupby(level=0).apply(lambda x: x.iloc[round(
                x.shape[0] * (n / (n_splits + 1))):round(
                x.shape[0] * ((n + 1) / (n_splits + 1)))]).droplevel(0)
            self.cv_data.append((train_set_X, train_set_Y, val_set_X, val_set_Y))

    def _stride_data_single(self, frame):
        """
        Uses windowing algorithm to prepare time-series data from sequences to prediction. Takes labels and horizon size
        to create chunks of data.

        Args:
            frame: TrajectoriesFrame of single user

        Returns:
            Chunks of data in a DataFrame - features and targets
        """
        to_concat = []
        for uid, traj in frame.groupby(level=0):
            traj = traj.reset_index()
            for x in range(1, self.horizon + 1):
                traj['labels{}'.format(x)] = traj['labels'].shift(-x)
            traj['datetime'] = traj['datetime'].shift(-self.horizon)
            traj = traj.set_index(['user_id', 'datetime'])
            to_concat.append(traj[:-self.horizon])
        frame_ready = pd.concat(to_concat)
        frame_X = frame_ready.iloc[:, :self.horizon]
        frame_Y = frame_ready.iloc[:, -1]
        return frame_X, frame_Y

    def stride_data(self, sequence):
        """
        A wrapper function for data windowing algorithm. Calls windowing algorithm for every unique user in the dataset.
        After it splits data into three datasets - test, train and validation.
        """
        strided_X, strided_Y = self._stride_data_single(sequence['labels'])
        train_frame_X, train_frame_Y, self.test_frame_X, self.test_frame_Y = self._test_split(strided_X, strided_Y)
        self._cv_split(train_frame_X, train_frame_Y, n_splits=self._n_splits)


class SKLearnPred():
    """
    This is a wrapper for classification function from sklearn library which can be used for prediction here.

    Args:
        algorithm: An algorithm from sklearn library
        training_data: The training set from Splitter class
        test_data: The test set from Splitter class
        param_dist: Hyperparameters dictionary from which the best hyperparameters will be chosen
        search_size: The search size for hyperparameters (the number of random combinations to chechk)
        cv_size: The number of tests run of every cross validation
        parallel: Whether random search should be performed using multithreading
    """

    def __init__(self, algorithm, training_data, test_data, param_dist, search_size, cv_size=3, parallel=False):
        """
        Class initialisation.
        """
        self._parallel = parallel
        self._training_data = training_data
        self._test_data = test_data
        self._algorithm = algorithm
        self._param_dist = param_dist
        self._search_size = search_size
        self._cv_size = cv_size
        self._tuned_alg = {}

    def _user_learn(self, split, p_comb):
        """
        For multithreading processing: single-user learn algorithm.

        Args:
            args_x: training features
            args_y: training targets
            vals_x: validation features
            vals_y: validation targets

        Returns:
            user id and score board with accuracy for each hyperparameters combination
        """
        args_x, args_y, vals_x, vals_y = split
        alg_run = self._algorithm(**p_comb).fit(args_x, args_y)
        pred_run = alg_run.predict(vals_x)
        metric_val = accuracy_score(pred_run, vals_y)
        return p_comb, metric_val

    def learn(self):
        """
        Learns the sklearn algorithm using cross-validation and passed input data.
        """
        result_dic = {}
        usrs = pd.unique(self._test_data[0].index.get_level_values(0))
        for ids in tqdm(usrs, total=len(usrs)):
            usr_split = [[a.loc[ids] for a in spl] for spl in self._training_data]
            params_to_check = iterate_random_values(self._param_dist, self._search_size)
            score_board = {}
            if self._parallel:
                for p_comb in params_to_check:
                    with cf.ThreadPoolExecutor(max_workers=len(usr_split)) as executor:
                        results = list(
                            tqdm(executor.map(self._user_learn, usr_split, itertools.repeat(p_comb)),
                                 total=len(usr_split)))
                    averaged = np.mean([v[1] for v in results])
                    score_board[tuple(sorted(results[0][0].items()))] = averaged
                    if averaged == 1:
                        break
            else:
                for p_comb in params_to_check:
                    fold_avg = []
                    cnt = 0
                    for splits in usr_split:
                        cnt += 1
                        print("SPLIT: {}".format(cnt))
                        args_x, args_y, vals_x, vals_y = splits
                        alg_run = self._algorithm(**p_comb, n_jobs=-1).fit(args_x, args_y)
                        pred_run = alg_run.predict(vals_x)
                        metric_val = accuracy_score(pred_run, vals_y)
                        fold_avg.append(metric_val)
                    metric_val = np.mean(fold_avg)
                    score_board[tuple(sorted(p_comb.items()))] = metric_val
                    if metric_val == 1:
                        break
            result_dic[ids] = max(score_board.keys(), key=lambda x: score_board[x])
        for usr, params in result_dic.items():  # selects the best params and trains the algorithm on them (for each user)
            params = {k: v for k, v in params}
            concat_x = pd.concat([self._training_data[-1][0].loc[usr], self._training_data[-1][2].loc[usr]])
            concat_y = pd.concat([self._training_data[-1][1].loc[usr], self._training_data[-1][3].loc[usr]])
            self._tuned_alg[usr] = self.algorithm(**params).fit(concat_x, concat_y)

    def test(self):
        """
        Implements one and final algorithm test and saves the score.
        """
        test_x, test_y = self._test_data
        usrs = np.unique(test_x.index.get_level_values(0))
        metrics = {}
        predictions_dic = {}
        predictions_dic_proba = {}
        for ids in tqdm(usrs, total=len(usrs)):
            cur_alg = self._tuned_alg[ids]
            preds = cur_alg.predict(test_x.loc[ids])
            preds_proba = pd.DataFrame(cur_alg.predict_proba(test_x.loc[ids]))
            preds_proba.columns = cur_alg.classes_
            metric_val = accuracy_score(preds, test_y.loc[ids])
            predictions_dic[ids] = preds
            metrics[ids] = metric_val
            predictions_dic_proba[str(ids)] = preds_proba.to_dict(orient='index')
        self.scores = pd.Series(metrics)
        self.predictions = pd.concat(
            [pd.concat({k: pd.Series(v) for k, v in predictions_dic.items()}).droplevel(1), test_y.droplevel(1)],
            axis=1)
        self.predictions_proba = predictions_dic_proba

    @property
    def algorithm(self):
        return self._algorithm


class Baseline():
    """
    The baseline algorithm which assumes the next symbol be identical to the next one. Be aware that it uses information
    from the test set and thus, has advantage over other methods. Only test dataset is needed.

    Args:
        test_data: Test dataset. Do not use split data.
    """

    def __init__(self, test_data):
        """
        Class initialisation.
        """
        self._test_data = test_data
        self.prediction = []

    def predict(self):
        """
        Baseline prediction algorithm. This method only evaluates the accuracy of the method and do not return the
        predictions. Predictions are stored within the class attributes.

        Returns:
            accuracy score
        """
        self.prediction = self._test_data[0].groupby(level=0).apply(lambda x: x.iloc[:, -1]).droplevel(
            1)  # make predictions
        aligned = pd.concat([self._test_data[1], self.prediction], axis=1)  # align predictions and true labels
        acc_score = aligned.groupby(level=0).apply(
            lambda x: sum(x.iloc[:, 0] == x.iloc[:, 1]) / x.shape[0])  # quickly evaluate the score
        print("SCORE: {}".format(acc_score.mean()))
        return aligned, acc_score


class TopLoc():
    """
    This is a naive algorithm which assumes every symbol in the test dataset being the most frequently occurring symbol
    in the training data.

    Args:
        train_data: Train dataset, from the Splitter class. Do not use split data.
        test_data: Test dataset, from the Splitter class. Do not use split data.
    """

    def __init__(self, train_data, test_data):
        """
        Class initialisation.
        """
        self._train_data = train_data
        self._test_data = test_data
        self.prediction = []

    def predict(self):
        """
        Makes the prediction and returns the score. Predictions are stored within the class.

        Returns:
            Accuracy score
        """
        tr_data = pd.concat([self._train_data[0][1], self._train_data[0][3]])
        top_location = tr_data.groupby(level=0).apply(lambda x: x.groupby(x).count().idxmax())
        to_conc = {}
        for uid, vals in self._test_data[1].groupby(level=0):
            vals[vals > -1] = top_location.loc[uid]
            to_conc[uid] = vals
        self.prediction = pd.concat(to_conc).droplevel(1)
        aligned = pd.concat([self._test_data[1], self.prediction], axis=1)
        acc_score = aligned.groupby(level=0).apply(lambda x: sum(x.iloc[:, 0] == x.iloc[:, 1]) / x.shape[0])
        print("SCORE: {}".format(acc_score.mean()))
        return aligned, acc_score


def split(trajectories_frame, train_size, state_size):
    """
    Simple train-test data split for a Markov Chain.

    Args:
        trajectories_frame: TrajectoriesFrame class object
        train_size: The split ratio for training set
        state_size: The size of a window (for a Markov Chain algorithm)

    Returns:
        Split data
    """
    train_frame = trajectories_frame['labels'].groupby(level=0).progress_apply(
        lambda x: x.iloc[:round(len(x) * train_size)])
    test_frame = trajectories_frame['labels'].groupby(level=0).progress_apply(
        lambda x: x.iloc[round(len(x) * train_size) - state_size:])
    return train_frame, test_frame


def markov_wrapper(trajectories_frame, test_size=.2, state_size=2, update=False, online=True):
    """
    The wrapper, one stop shop algorithm for the Markov Chain. Splits the data, learns the model and makes predictions
    on the test set.

    Args:
        trajectories_frame: TrajectoriesFrame class object
        test_size: The training set size.
        state_size: The order of the Markov Chain
        update: Whether the model should update its beliefs based on own predictions
        averaged: Whether an averaged accuracy should be returned
        online: Whether the algorithm should make an online prediction (sees the last n symbols when predicting)

    Returns:
        Prediction scores
    """
    train_size = 1 - test_size
    train_frame, test_frame = split(trajectories_frame, train_size, state_size)  # train test split
    test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
    predictions_dic = {}
    for uid, train_values in train_frame.groupby(level=0):  # training
        try:
            if online:
                predictions_dic[uid] = MarkovChain(list(train_values.values), state_size)
            else:
                predictions_dic[uid] = MarkovChain(list(train_values.values), state_size).move_from_build(
                    test_lengths.loc[uid], update)
        except:
            continue
    results_dic = {}
    to_conc = {}
    to_conc_topk = {}
    for test_values, prediction_values in zip([g for g in test_frame.groupby(level=0)], predictions_dic):  # predicting
        uid = test_values[0]
        test_values = test_values[1].values
        if online:
            forecast = []
            topk = []
            for current_state in range(len(test_values) - state_size):
                pred = predictions_dic[uid].move(test_values[current_state:current_state + state_size])
                forecast.append(pred[0])
                topk.append({str(k): v for k, v in pred[1].items()})
            to_conc[uid] = forecast
            results_dic[uid] = sum(forecast == test_values[state_size:]) / len(forecast)
            to_conc_topk[uid] = topk
        else:
            results_dic[uid] = sum(test_values[state_size:] == predictions_dic[prediction_values]) / len(test_values)
            to_conc[uid] = predictions_dic[prediction_values]
    predictions = pd.DataFrame().from_dict(to_conc, orient='index').T.unstack().droplevel(1)
    predictions = predictions[~predictions.isna()]
    test_frame_stack = test_frame.groupby(level=0).apply(lambda x: x[state_size:]).droplevel([1, 2])
    if test_frame_stack.index.nlevels == 2:
        test_frame_stack = test_frame_stack.droplevel(1)
    aligned = pd.concat([predictions, test_frame_stack], axis=1)
    aligned.columns = ['predictions', 'y_set']
    return aligned, pd.DataFrame.from_dict(results_dic, orient='index'), to_conc_topk


def sparse_wrapper_learn(train_frame, overreach=True, reverse=False, old=False, rolls=True,
                         remove_subsets=False, reverse_overreach=False, search_size=None, jit=True, parallel=True,
                         cuda=False, truncate=0):
    if old == True and any((overreach, reverse, reverse_overreach, remove_subsets, rolls)):
        warnings.warn("When old is set to True, other parameters have no effect")
    predictions_dic = {}
    for uid, train_values in train_frame.groupby(level=0):
        if old:
            predictions_dic[uid] = Sparse_old()
            predictions_dic[uid].fit(train_values.values)
        else:
            predictions_dic[uid] = Sparse(overreach=overreach, reverse=reverse, rolls=rolls,
                                          remove_subsets=remove_subsets, reverse_overreach=reverse_overreach,
                                          search_size=search_size)
            predictions_dic[uid].fit(train_values.values, jit=jit, parallel=parallel, cuda=cuda, truncate=truncate)
    return predictions_dic


def sparse_wrapper_test(predictions_dic, test_frame, trajectories_frame, split_ratio, test_lengths,
                        length_weights=None, recency_weights=None, use_probs=False, org_recency_weights=None,
                        org_length_weights=None,
                        completeness_weights=None, uniqueness_weights=None, parallel=True):
    results_dic = {}
    forecast_dic = {}
    topk_dic = {}
    for test_values, prediction_values in zip([g for g in test_frame.groupby(level=0)], predictions_dic):  # predicting
        uid = test_values[0]
        test_values = test_values[1].values
        forecast = []
        topk = []
        split_ind = round(trajectories_frame.loc[uid].shape[0] * split_ratio)
        if parallel:
            with cf.ThreadPoolExecutor() as executor:
                contexts = [trajectories_frame.loc[uid].iloc[:split_ind + n].values for n in range(test_lengths.loc[uid])]
                preds = list(tqdm(executor.map(predictions_dic[uid].predict, contexts,
                                               repeat(length_weights),
                                               repeat(recency_weights),
                                               repeat(use_probs),
                                               repeat(org_length_weights),
                                               repeat(org_recency_weights),
                                               repeat(completeness_weights),
                                               repeat(uniqueness_weights)),
                             total=test_lengths.loc[uid]))
                forecast = [x[0] for x in preds]
                topk = [x[1] for x in preds]
        else:
            for n in tqdm(range(test_lengths.loc[uid]), total=test_lengths.loc[uid]):
                context = trajectories_frame.loc[uid].iloc[:split_ind].values
                pred = predictions_dic[uid].predict(context, length_weights=length_weights, recency_weights=recency_weights,
                                                    from_dist=use_probs,
                                                    org_length_weights=org_length_weights,
                                                    org_recency_weights=org_recency_weights,
                                                    completeness_weights=completeness_weights,
                                                    uniqueness_weights=uniqueness_weights)
                topk.append(pred[1])
                forecast.append(pred[0])
                split_ind += 1
        results_dic[uid] = sum(forecast == test_values) / len(forecast)
        forecast_dic[uid] = forecast
        topk_dic[uid] = topk
    forecast_df = pd.DataFrame().from_dict(forecast_dic, orient='index').T.unstack().droplevel(1)
    forecast_df = forecast_df[~forecast_df.isna()]
    forecast_df = pd.concat([forecast_df, test_frame.droplevel([1, 2])], axis=1)
    forecast_df.columns = ['predictions', 'y_set']
    results_dic = pd.DataFrame().from_dict(results_dic, orient='index')
    return forecast_df, results_dic, topk_dic


def sparse_wrapper(trajectories_frame, test_size=.2, state_size=0, averaged=True, length_weights=None,
                   recency_weights=None, use_probs=False,
                   overreach=True, reverse=False, old=False, rolls=True, remove_subsets=False, reverse_overreach=False,
                   search_size=None):
    split_ratio = 1 - test_size
    train_frame, test_frame = split(trajectories_frame, split_ratio, state_size)
    test_lengths = test_frame.groupby(level=0).apply(lambda x: x.shape[0])
    predictions_dic = {}
    for uid, train_values in train_frame.groupby(level=0):
        if old:
            predictions_dic[uid] = Sparse_old()
            predictions_dic[uid].fit(train_values.values)
        else:
            predictions_dic[uid] = Sparse(overreach=overreach, reverse=reverse, rolls=rolls,
                                          remove_subsets=remove_subsets, reverse_overreach=reverse_overreach,
                                          search_size=search_size)
            predictions_dic[uid].fit(train_values.values)
    results_dic = {}
    for test_values, prediction_values in zip([g for g in test_frame.groupby(level=0)], predictions_dic):  # predicting
        uid = test_values[0]
        test_values = test_values[1].values
        forecast = []
        split_ind = round(trajectories_frame.uloc(uid).shape[0] * split_ratio)
        for n in tqdm(range(test_lengths.loc[uid]), total=test_lengths.loc[uid]):
            context = trajectories_frame.uloc(uid).iloc[:split_ind].labels.values
            pred = predictions_dic[uid].predict(context, length_weights=length_weights, recency_weights=recency_weights,
                                                from_dist=use_probs, )
            forecast.append(pred)
            split_ind += 1
        results_dic[uid] = sum(forecast == test_values[state_size:]) / len(forecast)
    if averaged:
        return sum(list(results_dic.values())) / len(results_dic.values())
    else:
        return results_dic
