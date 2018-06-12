import numpy as np
from preprocessing import TraceStats
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

data_path = './../data/'
output_path = '/home/ynezri/TexStudio/incremental/img/'

#######################################################################################################################
# Error Metrics
#######################################################################################################################


class ErrorMetric:
    name = 'do_not_use'

    @staticmethod
    def error(truth, estimation):
        raise NotImplementedError


class MSE(ErrorMetric):
    name = 'MSE'

    @staticmethod
    def error(truth, estimation):
        return np.mean(np.power((estimation - truth), 2))


class RMSE(ErrorMetric):
    name = 'RMSE'

    @staticmethod
    def error(truth, estimation):
        return np.sqrt(MSE.error(truth, estimation))


class MAPE(ErrorMetric):
    name = 'MAPE'

    @staticmethod
    def error(truth, estimation):
        return 100 * np.mean(np.abs((truth - estimation) / truth))


class Cumulative(ErrorMetric):
    name = 'Cumulative'

    @staticmethod
    def error(truth, estimation):
        return sum(np.abs(truth - estimation))

#######################################################################################################################
# Time Analysis
#######################################################################################################################


def time_estimate(ts, estimators):

    time_d = {}

    for estimator in estimators:
        total_time = 0
        max_time = -float('Inf')
        min_time = float('Inf')

        for bs in ts.batch_list:
            start = perf_counter()
            estimator.estimate(bs)
            end = perf_counter()

            max_time = max(max_time, end-start)
            min_time = min(min_time, end-start)
            total_time += (end - start)

        time_d[estimator.name] = [total_time / len(ts.batch_list), max_time, min_time]

    return pd.DataFrame.from_dict(time_d, orient='index', columns=['avg_time', 'max_time', 'min_time'])


def time_fit(ts, estimators, feature_names):

    time_d = {}

    for estimator in estimators:
        total_time = 0
        max_time = -float('Inf')
        min_time = float('Inf')

        estimator.estimate(ts.batch_list[0])
        for bs in ts.batch_list:
            features = bs.get_features(feature_names)

            start = perf_counter()
            estimator.model.partial_fit(features, [bs.batch_card])
            end = perf_counter()

            max_time = max(max_time, end-start)
            min_time = min(min_time, end-start)
            total_time += (end - start)

        time_d[estimator.name] = [total_time / len(ts.batch_list), max_time, min_time]

    return pd.DataFrame.from_dict(time_d, orient='index', columns=['avg_time', 'max_time', 'min_time'])


def time_predict(ts, estimators, feature_names):

    time_d = {}
    for estimator in estimators:
        total_time = 0
        max_time = -float('Inf')
        min_time = float('Inf')

        for bs in ts.batch_list:
            features = bs.get_features(feature_names)
            estimator.model.partial_fit(features, [bs.batch_card])

            start = perf_counter()
            estimator.model.predict(features)
            end = perf_counter()

            max_time = max(max_time, end-start)
            min_time = min(min_time, end-start)
            total_time += (end - start)

        time_d[estimator.name] = [total_time / len(ts.batch_list), max_time, min_time]

    return pd.DataFrame.from_dict(time_d, orient='index', columns=['avg_time', 'max_time', 'min_time'])

#######################################################################################################################
# Evaluation Section Graphs
#######################################################################################################################

error_metrics = [RMSE, MAPE, Cumulative]


def plot_card(trace, sampling_rate, partitions, ylim, xlabel='Batch Index'):

    for i in range(len(partitions)):
        partition = partitions[i]
        ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)

        # ignore last batch
        ts.remove_last_batch()

        plt.figure()
        ts.to_df().batch_card.plot()
        plt.ylabel('Batch cardinality (D)', fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(output_path + '%s_%s_card.png' % (trace, partition))


def plot_sampling(trace, sampling_rate, partition, models, model_names, ylim, xlabel='Batch Index'):

    errors = {}
    ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
    ts.remove_last_batch()
    truth = ts.to_df().batch_card
    for i in range(len(models)):
        estimator = models[i](name=model_names[i])
        est = ts.run_estimation(estimator)
        plt.figure()
        plt.plot(truth)
        plt.plot(est)
        plt.legend(loc=1)
        plt.ylim(ylim)
        errors[estimator.name] = [error.error(truth, est) for error in error_metrics]

        plt.ylabel('Batch cardinality (D)', fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.tight_layout()
        plt.savefig(output_path + '%s_%s_sampling.png' % (trace, model_names[i]))
    return pd.DataFrame.from_dict(errors, orient='index',
                                  columns=[error.name for error in error_metrics]).reindex(model_names)


def plot_ml(trace, sampling_rate, training_rate, partition, features, models, model_names, ylim, xlabel='Batch Index'):

    errors = {}
    ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
    ts.remove_last_batch()
    truth = ts.to_df().batch_card

    for i in range(len(models)):
        estimator = models[i](name=model_names[i], features=features, training_rate=training_rate)

        est = ts.run_estimation(estimator)

        plt.figure()
        plt.plot(truth)
        plt.plot(est)
        plt.legend(loc=1)
        plt.ylim(ylim)
        errors[estimator.name] = [error.error(truth, est) for error in error_metrics]
        plt.ylabel('Batch cardinality (D)', fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.tight_layout()
        plt.savefig(output_path + '%s_%s_online_ml.png' % (trace, model_names[i]))

    return pd.DataFrame.from_dict(errors, orient='index',
                                  columns=[error.name for error in error_metrics]).reindex(model_names)


def plot_features(trace, sampling_rate, training_rate, partition, feature_sets, models, model_names, error=MAPE):

    error_d = {}

    ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
    ts.remove_last_batch()
    truth = ts.to_df().batch_card

    for i, j in product(range(len(models)), range(len(feature_sets))):
        if j == 0:
            model_name = model_names[i]
            error_d[model_name] = []
        features = feature_sets[j]
        estimator = models[i](name=model_name, features=features, training_rate=training_rate)

        est = ts.run_estimation(estimator)
        error_d[model_name].append(error.error(truth, est))

    df = pd.DataFrame.from_dict(error_d, orient='index').reindex(model_names)
    df.columns = map(str, feature_sets)
    df.plot(kind='bar')
    plt.ylabel(error.name, fontsize=15)
    plt.xticks(fontsize=15, rotation=25)
    lgd = plt.legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(output_path + '%s_features.png' % trace, bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    return df


def plot_tradeoff(trace, effective_sampling_rate, sampling_rates, partitions, features, model, model_name, error=MAPE):

    error_d = {}
    trainings = []
    labels = []

    for i, j in product(range(len(partitions)), range(len(sampling_rates))):
        if j == 0:
            partition = partitions[i]
            error_d[partition] = []
        sampling_rate = sampling_rates[j]
        training_rate = (effective_sampling_rate - sampling_rate) / (1 - sampling_rate)
        labels.append('%.4f / %.4f' % (sampling_rate, training_rate))

        ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
        ts.remove_last_batch()
        truth = ts.to_df().batch_card
        trainings.append(ts.batch_count * training_rate)

        estimator = model(name=model_name, features=features, training_rate=training_rate)

        est = ts.run_estimation(estimator)
        error_d[partition].append(error.error(truth, est))

    df = pd.DataFrame.from_dict(error_d, orient='index').reindex(partitions)
    df.columns = map(str, sampling_rates)
    df.plot(kind='bar')
    plt.ylabel(error.name, fontsize=15)
    plt.xticks(fontsize=15, rotation=25)
    lgd = plt.legend(labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                     title='sampling_rate / training_rate')
    plt.savefig(output_path + '%s_tradeoff.png' % trace, bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    return df
