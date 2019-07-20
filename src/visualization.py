import numpy as np
from src.preprocessing import TraceStats
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
import pylatex as pl
from cycler import cycler

data_path = '../data/'
output_path = '../paper/img/'
table_path = '../paper/tbl/'
plot_format = 'pdf'

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
    """
    compare time for fit operation between estimators
    """

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
    """
    compare time of predict operation for estimators.
    """

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
    """
    plot cardinality (baseline) for trace.
    """

    for i in range(len(partitions)):
        partition = partitions[i]
        ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)

        # ignore last batch
        ts.remove_last_batch()

        plt.figure()
        truth = ts.to_df().batch_card
        truth.name = 'real'
        truth.plot()
        plt.ylabel('Batch cardinality (D)', fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(output_path + '%s_%s_card.%s' % (trace, partition, plot_format), format=plot_format)


def plot_sampling(trace, sampling_rate, partition, models, model_names, ylim, xlabel='Batch Index'):
    """
    plot results for statistical sampling based algorithms (thesis version).
    """

    errors = {}
    ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
    ts.remove_last_batch()
    truth = ts.to_df().batch_card
    truth.name = 'real'

    for i in range(len(models)):
        estimator = models[i](name=model_names[i])
        est = ts.run_estimation(estimator)
        plt.figure()
        plt.plot(truth)

        plt.plot(est)

        plt.legend(loc=1)
        plt.ylim(ylim)
        plt.ylabel('Batch cardinality (D)', fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.tight_layout()
        plt.savefig(output_path + '%s_%s_sampling.%s' % (trace, model_names[i], plot_format), format=plot_format)

        errors[estimator.name] = [error.error(truth, est) for error in error_metrics]

    return pd.DataFrame.from_dict(errors, orient='index',
                                  columns=[error.name for error in error_metrics]).reindex(model_names)


def plot_sampling_paper(trace, sampling_rate, partition, models, model_names, ylim, xlabel='Batch Index'):
    """
    plot results for statistical sampling based algorithms (paper version).
    """

    errors = {}
    ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
    ts.remove_last_batch()
    truth = ts.to_df().batch_card.interpolate()
    truth.name = 'real'
    lines = 0
    plt.figure()
    plt.rcParams.update({'legend.handlelength': 5})

    monochrome = (cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']) +
                  cycler('linestyle', ['-', '--', ':', '-.']) +
                  cycler('marker', ['o', '^', 's', 'v']))

    markers = cycler('color', 'k') * cycler('marker', ['o', '^', 's', 'v'])

    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    ax.grid(linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.plot(truth, linewidth=2, markevery=[0, -1],
             markersize=7, markeredgecolor='k', markerfacecolor='k')

    for i in range(len(models)):
        estimator = models[i](name=model_names[i])
        est = ts.run_estimation(estimator).interpolate()
        plt.plot(est, linewidth=2, markevery=([0, -1]),
                 markersize=7, markeredgecolor='k', markerfacecolor='k')

    plt.legend(loc=1)

    # re-plot markers
    ax.set_prop_cycle(markers)
    plt.plot(truth, linestyle='', markevery=((lines * 10) % 50, 50),
             markersize=7, markeredgecolor='k', markerfacecolor='k')

    for i in range(len(models)):
        lines += 1
        estimator = models[i](name=model_names[i])
        est = ts.run_estimation(estimator).interpolate()
        plt.plot(est, markevery=((lines * 10) % 50, 50),  linestyle='',
                 markersize=7, markeredgecolor='k', markerfacecolor='k')
        errors[estimator.name] = [error.error(truth, est) for error in error_metrics]

    plt.ylim(ylim)
    plt.ylabel('Batch cardinality (D)', fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path + '%s_sampling_paper.%s' % (trace, plot_format), format=plot_format)
    return pd.DataFrame.from_dict(errors, orient='index',
                                  columns=[error.name for error in error_metrics]).reindex(model_names)


def plot_ml(trace, sampling_rate, training_rate, partition, features, models, model_names, ylim, xlabel='Batch Index'):
    """
    plot results for online ML algorithms (thesis version).
    """
    errors = {}
    ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
    ts.remove_last_batch()
    truth = ts.to_df().batch_card
    truth.name = 'real'

    for i in range(len(models)):
        estimator = models[i](name=model_names[i], features=features, training_rate=training_rate)

        est = ts.run_estimation(estimator)

        plt.figure()
        plt.plot(truth)

        plt.plot(est)

        plt.legend(loc=1)
        plt.ylim(ylim)
        plt.ylabel('Batch cardinality (D)', fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.tight_layout()
        plt.savefig(output_path + '%s_%s_online_ml.%s' % (trace, model_names[i], plot_format), format=plot_format)

        errors[estimator.name] = [error.error(truth, est) for error in error_metrics]

    return pd.DataFrame.from_dict(errors, orient='index',
                                  columns=[error.name for error in error_metrics]).reindex(model_names)


def plot_ml_paper(trace, sampling_rate, training_rate, partition, features, models,
                  model_names, ylim, xlabel='Batch Index'):
    """
    plot results for online ML algorithms (paper version).
    """
    errors = {}
    ts = TraceStats.load(data_path + trace + '_' + partition + '_' + '%.4f.pickle' % sampling_rate)
    ts.remove_last_batch()
    truth = ts.to_df().batch_card
    truth.name = 'real'
    lines = 0

    plt.rcParams.update({'legend.handlelength': 5})

    monochrome = (cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']) +
                  cycler('linestyle', ['-', '--', ':', '-.']) +
                  cycler('marker', ['o', '^', 's', 'v']))

    plt.figure()
    ax = plt.gca()
    ax.set_prop_cycle(monochrome)
    ax.grid(linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.plot(truth, markevery=[0, -1], linewidth=2,
             markersize=7, markeredgecolor='k', markerfacecolor='k')

    for i in range(len(models)):
        estimator = models[i](name=model_names[i], features=features, training_rate=training_rate)
        est = ts.run_estimation(estimator)
        plt.plot(est, markevery=[0, -1],  linewidth=2,
                 markersize=7, markeredgecolor='k', markerfacecolor='k')
        errors[estimator.name] = [error.error(truth, est) for error in error_metrics]

    plt.legend(loc=1)

    # re-plot markers
    plt.plot(truth, markevery=((lines * 10) % 50, 50), linestyle='',
             markersize=7, markeredgecolor='k', markerfacecolor='k')

    for i in range(len(models)):
        lines += 1
        estimator = models[i](name=model_names[i], features=features, training_rate=training_rate)
        est = ts.run_estimation(estimator)
        plt.plot(est, markevery=((lines * 10) % 50, 50),  linestyle='',
                 markersize=7, markeredgecolor='k', markerfacecolor='k')
        errors[estimator.name] = [error.error(truth, est) for error in error_metrics]

    plt.ylim(ylim)
    plt.ylabel('Batch cardinality (D)', fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path + '%s_online_ml_paper.%s' % (trace, plot_format), format=plot_format)

    return pd.DataFrame.from_dict(errors, orient='index',
                                  columns=[error.name for error in error_metrics]).reindex(model_names)


def plot_features(trace, sampling_rate, training_rate, partition, feature_sets, models, model_names, error=MAPE,
                  legend='outside', ylim=None):
    """
    plot comparison between different feature sets.
    """

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

    plt.rcParams.update({'legend.handlelength': 3})
    plt.rcParams.update({'legend.handleheight': 1})
    ax = plt.gca()
    bars = ax.patches
    patterns = ['///', '--', '...', '\///', 'xxx', '\\\\']
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    if ylim is not None:
        plt.ylim(ylim)

    if legend == 'outside':
        lgd = plt.legend(title='Feature Set', labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(output_path + '%s_features.%s' % (trace, plot_format), bbox_extra_artists=(lgd,),
                bbox_inches='tight', format=plot_format)
    if legend == 'inside':
        plt.legend(title='Feature Set', labels=labels)
        plt.savefig(output_path + '%s_features.%s' % (trace, plot_format))
    return df


def plot_tradeoff(trace, effective_sampling_rate, sampling_rates, partitions, features, model, model_name, error=MAPE,
                  legend='ouside', ylim=None):
    """
    plot sampling rate / training rate tradeoff.
    """
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

    plt.rcParams.update({'legend.handlelength': 3})
    plt.rcParams.update({'legend.handleheight': 1})
    ax = plt.gca()
    bars = ax.patches
    patterns = ['///', '--', '...', '\///', 'xxx', '\\\\']
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    if ylim is not None:
        plt.ylim(ylim)

    if legend == 'outside':
        lgd = plt.legend(labels=labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                         title='sampling_rate / training_rate')
        plt.savefig(output_path + '%s_tradeoff.%s' % (trace, plot_format), bbox_extra_artists=(lgd,),
                    bbox_inches='tight', format=plot_format)
    if legend == 'inside':
        lgd = plt.legend(labels=labels, title='sampling_rate / training_rate')
        plt.savefig(output_path + '%s_tradeoff.%s' % (trace, plot_format), format=plot_format)

    return df


#######################################################################################################################
# Dataframe to PDF
#######################################################################################################################

def df_to_pdf(df, out_file, print_index=True, debug=False, digit_round=1, caption=None, comma_separated_columns=[],
              gen_latex='False'):
    """
    convert data frame to pdf/latex. used to create tables for paper/thesis.
    """
    if digit_round is not None:
        df = df.round(digit_round)

    for column in comma_separated_columns:
        df[column] = df[column].map('{:,}'.format)

    table_columns = len(df.columns)+1 if print_index else len(df.columns)

    if gen_latex:
        with open(table_path + out_file + '.tex', 'w') as f:
            f.write(df.to_latex(escape=False, index=print_index, column_format='c'*table_columns))
        return

    doc = pl.Document(documentclass='standalone', document_options='varwidth')
    doc.packages.append(pl.Package('booktabs'))

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
