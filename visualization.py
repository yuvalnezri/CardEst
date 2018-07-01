import numpy as np
from preprocessing import TraceStats
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
import pylatex as pl

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
    name = 'CE'

    @staticmethod
    def error(truth, estimation):
        return sum(np.abs(truth - estimation))


class MAE(ErrorMetric):
    name = 'MAE'

    @staticmethod
    def error(truth, estimation):
        return np.mean(np.abs(truth - estimation))


class MAXAE(ErrorMetric):
    name = 'MAXAE'

    @staticmethod
    def error(truth, estimation):
        return max(np.abs(truth - estimation))


#######################################################################################################################
# Time Analysis
#######################################################################################################################


def time_estimate(tss, estimators):

    time_d = {}
    counter = 0

    for ts in tss:
        time_d[counter] = [ts.batch_list[0].sample_size]
        for estimator in estimators:
            total_time = 0
            for bs in ts.batch_list:

                start = perf_counter()
                estimator.estimate(bs)
                end = perf_counter()

                total_time += end - start
            time_d[counter].append(total_time/ts.batch_count)
        counter += 1
    estimator_names = [estimator.name for estimator in estimators]
    df = pd.DataFrame.from_dict(time_d, orient='index',
                                columns=['Mean Sample Size'] + estimator_names)

    # format times
    df[estimator_names] = df[estimator_names].applymap('{:,.2e}'.format)
    return df


def time_fit(tss, estimators, estimator_names, feature_names):

    time_d = {}
    counter = 0

    for ts in tss:
        inited_estimators = []
        for i in range(len(estimators)):
            estimator_class = estimators[i]
            estimator_name = estimator_names[i]
            inited_estimators.append(estimator_class(estimator_name, feature_names, 1))

        time_d[counter] = [ts.batch_list[0].sample_size]

        for estimator in inited_estimators:
            estimator.estimate(ts.batch_list[0])
            total_time = 0

            for bs in ts.batch_list:

                features = bs.get_features(feature_names)

                start = perf_counter()
                estimator.model.partial_fit(features, [bs.batch_card])
                end = perf_counter()

                total_time += end-start
            time_d[counter].append(total_time/ts.batch_count)
        counter += 1

    df = pd.DataFrame.from_dict(time_d, orient='index',
                                columns=['Mean Sample Size'] + estimator_names)

    # format times
    df[estimator_names] = df[estimator_names].applymap('{:,.2e}'.format)
    return df


def time_predict(tss, estimators, estimator_names, feature_names):
    time_d = {}
    counter = 0

    for ts in tss:
        inited_estimators = []
        for i in range(len(estimators)):
            estimator_class = estimators[i]
            estimator_name = estimator_names[i]
            inited_estimators.append(estimator_class(estimator_name, feature_names, 1))

        time_d[counter] = [ts.batch_list[0].sample_size]

        for estimator in inited_estimators:
            estimator.estimate(ts.batch_list[0])
            total_time = 0

            for bs in ts.batch_list:
                features = bs.get_features(feature_names)

                start = perf_counter()
                estimator.model.predict(features)
                end = perf_counter()

                total_time += end - start

                # we don't want to predict over an untrained estimator
                estimator.model.partial_fit(features, [bs.batch_card])

            time_d[counter].append(total_time / ts.batch_count)
        counter += 1

    df = pd.DataFrame.from_dict(time_d, orient='index',
                                columns=['Mean Sample Size'] + estimator_names)

    # format times
    df[estimator_names] = df[estimator_names].applymap('{:,.2e}'.format)
    return df

#######################################################################################################################
# Evaluation Section Graphs
#######################################################################################################################

error_metrics = [RMSE, MAE, MAPE, MAXAE]


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
    plt.xticks(fontsize=15, rotation=0)
    labels = ['{' + ', '.join(feature_set) + '}' for feature_set in feature_sets]
    labels = [label.replace('f_', 'f') for label in labels]
    lgd = plt.legend(title='Feature Set', labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
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
    locs, _ = plt.xticks()
    ticks = [partition.replace('1S', '1 second').replace('S', ' seconds') for partition in partitions]
    plt.xticks(locs, ticks, fontsize=15, rotation=0)
    lgd = plt.legend(labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                     title='sampling_rate / training_rate')
    plt.savefig(output_path + '%s_tradeoff.png' % trace, bbox_extra_artists=(lgd,),
                bbox_inches='tight')
    return df


#######################################################################################################################
# Dataframe to PDF
#######################################################################################################################

def df_to_pdf(df, out_file, print_index=True, debug=False, digit_round=1, caption=None, comma_separated_columns=[]):

    if digit_round is not None:
        df = df.round(digit_round)

    for column in comma_separated_columns:
        df[column] = df[column].map('{:,}'.format)

    doc = pl.Document(documentclass='standalone', document_options='varwidth')
    doc.packages.append(pl.Package('booktabs'))

    table_columns = len(df.columns)+1 if print_index else len(df.columns)

    with doc.create(pl.MiniPage()):
        with doc.create(pl.Table(position='htbp')) as table:
            table.append(pl.Command('centering'))
            table.append(pl.NoEscape(df.to_latex(escape=False, index=print_index,
                                                 column_format='c'*table_columns)))
            if caption is not None:
                table.add_caption(caption)
    if debug:
        return doc.dumps()

    doc.generate_pdf(output_path + out_file)
