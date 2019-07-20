import subprocess
import datetime
import pandas as pd
import numpy as np
import pickle
from glob import glob
from collections import Counter
import multiprocessing
import time


##############################################################################
# The following code is used to process pcap/ucla traces, calculate
# true cardinality, sample, extract features and save to a dataframe.
# each pcap trace was divided to files according to batche_size using editcap.
# tshark is used to read pcaps files, multiprocessing is supported.
##############################################################################

# code referenced from the following NB: https://gist.github.com/dloss/5693316
def read_pcap(filename, fields=[], display_filter="",
              timeseries=False, strict=False):
    """
    Read PCAP file into Pandas DataFrame object.
    Uses tshark command-line tool from Wireshark.

    filename:       Name or full path of the PCAP file to read
    fields:         List of fields to include as columns
    display_filter: Additional filter to restrict frames
    strict:         Only include frames that contain all given fields
                    (Default: false)
    timeseries:     Create DatetimeIndex from frame.time_epoch
                    (Default: false)

    Syntax for fields and display_filter is specified in
    Wireshark's Display Filter Reference:

      http://www.wireshark.org/docs/dfref/
    """
    if timeseries:
        fields = ["frame.time_epoch"] + fields
    fieldspec = " ".join("-e %s" % f for f in fields)

    display_filters = fields if strict else []
    if display_filter:
        display_filters.append(display_filter)
    filterspec = "-Y '%s'" % " and ".join(f for f in display_filters)

    options = "-r %s -n -T fields -Eheader=y" % filename
    cmd = "tshark %s %s %s" % (options, filterspec, fieldspec)

    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE)
    if timeseries:
        df = pd.read_csv(proc.stdout,
			sep='\t',
			index_col="frame.time_epoch",
			parse_dates=True,
			date_parser=datetime.datetime.fromtimestamp,
			low_memory=False)
    else:
        df = pd.read_csv(proc.stdout, sep='\t', low_memory=False)
    return df


def ucla_to_df(filename):
    '''
    used to read a single TCP ucla trace file to dataframe.
    '''
    df = pd.read_csv(filename,
                       names=['ip.src', 'ip.dst', 'port.src', 'port.dst', 'frame.len', 'flag'],
                       sep=' ', usecols=range(1, 7), low_memory=False)
    df['tcp.port'] = df.apply(lambda x: '%d,%d' % (x['port.src'], x['port.dst']), axis=1)
    df = df.drop(labels=['port.src', 'port.dst'], axis=1)
    df['udp.port'] = np.nan
    df['ip.proto'] = 6
    df['frame.len'] = df['frame.len'] + 46
    df['tcp.flags.syn'] = df['flag'].apply(lambda x: 1 if x == 'S' else 0)
    df['tcp.flags.fin'] = df['flag'].apply(lambda x: 1 if x == 'F' else 0)
    df = df[["frame.len", "ip.src", "ip.dst", "udp.port", "tcp.port", "ip.proto", "tcp.flags.syn", "tcp.flags.fin"]]

    return df


def pcap_to_df(filename):
    fields = ["frame.len", "ip.src", "ip.dst", "udp.port", "tcp.port", "ip.proto", "tcp.flags.syn", "tcp.flags.fin"]

    # read file to pandas data frame
    df = read_pcap(filename, fields, strict=False)
    return df


class BatchStats:
    """
    This class holds the statisics for a single Batch.
    """
    def __init__(self):
        self.batch_size = None
        self.batch_card = None
        self.sampling_rate = None
        self.sample_size = None
        self.sample_card = None
        self.histogram = None
        self.avg_pkt_len = None
        self.syn_count = None
        self.process_batch_time = None
        self.process_sample_time = None

    @staticmethod
    def generate(probs):
        raise NotImplementedError

    @staticmethod
    def read_df(trace_df, sampling_rate):

        batch = BatchStats()

        process_batch_begin = time.perf_counter()
        packets = trace_df.apply(lambda x: hash((x['ip.src'], x['ip.dst'],
                                                 x['tcp.port'], x['udp.port'], x['ip.proto'])), axis=1)
        batch.batch_card = len(np.unique(packets))
        process_batch_end = time.perf_counter()

        # TODO: add hyperloglog calculation

        # sample batch
        batch.batch_size = len(packets)
        batch.sampling_rate = sampling_rate
        batch.sample_size = int(sampling_rate * batch.batch_size)
        sampled_indices = np.random.choice(batch.batch_size, batch.sample_size, replace=False)

        process_sample_begin = time.perf_counter()
        sampled_packets = trace_df.iloc[sampled_indices].apply(lambda x: hash((x['ip.src'], x['ip.dst'],
                                                                          x['tcp.port'], x['udp.port'],
                                                                          x['ip.proto'])), axis=1)

        # calc batch flow size histogram
        unique, counts = np.unique(sampled_packets, return_counts=True)
        batch.sample_card = len(unique)

        # calc batch flow distrubution histogram
        unique, counts = np.unique(counts, return_counts=True)

        process_sample_end = time.perf_counter()

        batch.histogram = Counter(dict(zip(unique, counts)))

        sampled_df = trace_df.iloc[sampled_indices]
        batch.avg_pkt_len = sampled_df['frame.len'].mean()
        batch.syn_count = len(sampled_df[sampled_df['tcp.flags.syn'] == 1])
        batch.fin_count = len(sampled_df[sampled_df['tcp.flags.fin'] == 1])
        batch.process_batch_time = process_batch_end - process_batch_begin
        batch.process_sample_time = process_sample_end - process_sample_begin

        return batch

    @classmethod
    def from_pcap(cls, filename, sampling_rate):
        df = pcap_to_df(filename)
        return cls.read_df(df, sampling_rate)

    @classmethod
    def from_ucla(cls, filename, sampling_rate):
        df = ucla_to_df(filename)
        return cls.read_df(df, sampling_rate)

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def get_features(self, feature_names):
        features = []

        if 'f_1' in feature_names:
            features.append(self.histogram[1])
        if 'f_2' in feature_names:
            features.append(self.histogram[2])
        if 'f_3' in feature_names:
            features.append(self.histogram[3])
        if 'avg_pkt_len' in feature_names:
            features.append(self.avg_pkt_len)
        if 'syn_count' in feature_names:
            features.append(self.syn_count)
        if 'sample_size' in feature_names:
            features.append(self.sample_size)

        return np.array([features])


class TraceStats:
    '''
    This class holds the statistics for an entire trace (list of batches).
    '''
    def __init__(self):
        self.batch_list = None
        self.batch_count = None

    @staticmethod
    def from_path(path, batch_parser, batch_count=None, multiprocess=False):

        ts = TraceStats()

        files = glob(path + '/*')
        files.sort()
        if batch_count is None:
            batch_count = len(files)
        ts.batch_count = batch_count

        files = files[:batch_count]

        parsed_files = []
        current = 1

        if multiprocess:
            p = multiprocessing.Pool()
            for parsed_file in p.imap(batch_parser, files):
                parsed_files.append(parsed_file)
                current += 1
            p.close()
        else:
            for file in files:
                parsed_files.append(batch_parser(file))
                current += 1

        ts.batch_list = parsed_files
        return ts

    def run_estimation(self, estimator):

        res = []
        for bs in self.batch_list:
            res.append(estimator.estimate(bs))

        return pd.Series(res, name=estimator.name)

    def time_estimation(self, estimator):

        times = []
        for bs in self.batch_list:
            begin = time.perf_counter()
            estimator.estimate(bs)
            end = time.perf_counter()
            times.append(end - begin)

        return pd.Series(times, name=estimator.name)

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def concat(self, trace_stats):
        new = TraceStats()
        new.batch_list = self.batch_list + trace_stats.batch_list
        new.batch_count = self.batch_count + trace_stats.batch_count
        return new

    def to_df(self):

        trace_data = []
        for batch in self.batch_list:
            batch_data = dict(batch.__dict__)
            batch_data['f_1'] = batch_data['histogram'][1]
            batch_data['f_2'] = batch_data['histogram'][2]
            batch_data['f_3'] = batch_data['histogram'][3]
            del batch_data['histogram']
            trace_data.append(batch_data)

        return pd.DataFrame(trace_data)

    def remove_last_batch(self):
        if self.batch_count == 0:
            raise IndexError()

        del self.batch_list[-1]
        self.batch_count -= 1
