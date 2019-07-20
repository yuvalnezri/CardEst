from scipy.optimize import brentq
import numpy as np
from collections import Counter


class Estimator:
    """
    abstract estimator class.
    """
    def __init__(self, name):
        self.name = name

    def estimate(self, batch_stats):
        """
        Returns an estimation over a single batch
        :param batch_stats: BatchStats data structure of a single batch.
        :return: cardinality estimation for batch_stats using Estimator.
        """
        raise NotImplementedError


class NaiveLearn(Estimator):
    """
    The most naive step function estimator, returns the cardinality value of the previous training example.
    """

    def __init__(self, name, training_rate):
        super(NaiveLearn, self).__init__(name)
        self.training_step = int(1 / training_rate)
        self.batch_counter = 0
        self.prev_estimation = None

    def estimate(self, batch_stats):
        if (self.batch_counter % self.training_step) == 0:
            self.prev_estimation = batch_stats.batch_card

        self.batch_counter += 1
        return self.prev_estimation

#######################################################################################################################
# Statistical Estimators
#######################################################################################################################


class GT(Estimator):
    """
    Good-Turing frequency estimator.
    see - I. J. Good. The population frequencies of species and the estimation of population parameters.
    Biometrika, 40(3-4), 1953.
    """

    def __init__(self, name):
        super(GT, self).__init__(name)

    def estimate(self, batch_stats):
        freq1 = batch_stats.histogram[1]
        if freq1 == batch_stats.sample_card:
            return batch_stats.sample_card
        return batch_stats.sample_card * (1 / (1 - freq1 / batch_stats.sample_size))


class GEE(Estimator):
    """
    Guaranteed Error Estimator.
    see - M. Charikar, S. Chaudhuri, R. Motwani, and V. Narasayya. Towards estimation error guarantees for distinct
    values. In Proceedings of the nineteenth ACM SIGMOD-SIGACT-SIGART. ACM, 2000.
    """
    def __init__(self, name):
        super(GEE, self).__init__(name)

    def estimate(self, batch_stats):
        freq1 = batch_stats.histogram[1]
        return np.sqrt(1 / batch_stats.sampling_rate) * freq1 + (sum(batch_stats.histogram.values()) - freq1)


class AE(Estimator):
    """
    Adaptive Estimator.
    see - M. Charikar, S. Chaudhuri, R. Motwani, and V. Narasayya. Towards estimation error guarantees for distinct
    values. In Proceedings of the nineteenth ACM SIGMOD-SIGACT-SIGART. ACM, 2000.
    """
    def __init__(self, name):
        super(AE, self).__init__(name)

    @staticmethod
    def _f(m, histogram):
        sum1 = sum(np.exp(-i) * histogram[i] for i in histogram)
        sum2 = sum(i * np.exp(-i) * histogram[i] for i in histogram)
        f1 = histogram[1]
        f2 = histogram[2]

        res = m - f1 - f2 - f1 * (sum1 + m * np.exp(-(f1 + 2 * f2) / m)) / (
        sum2 + (f1 + 2 * f2) * np.exp(-(f1 + 2 * f2) / m))
        return res

    def estimate(self, batch_stats):
        try:
            freq1 = batch_stats.histogram[1]
            freq2 = batch_stats.histogram[2]
            m = brentq(AE._f, 1, batch_stats.batch_size, args=(batch_stats.histogram,))
            ae = batch_stats.sample_card + m - freq1 - freq2
            return ae
        except ValueError as e:
            # logger.warning('brentq failed, returning sample_card as estimation.')
            return batch_stats.sample_card


class UJ2A(Estimator):
    """
    UJ2A estimator.
    see - P. J. Haas and L. Stokes. Estimating the number of classes in a finite population.
    Journal of the American Statistical Association, 93(444), 1998.
    """
    def __init__(self, name, c=50):
        super(UJ2A, self).__init__(name)
        self.c = c

    def estimate(self, batch_stats):

        f_i = Counter(batch_stats.histogram)
        reduced_elements = 0
        reduced_packets = 0
        for i in batch_stats.histogram.keys():
            if i > self.c:
                reduced_elements += 1
                reduced_packets += i * f_i[i]
                del f_i[i]

        f_1 = f_i[1]
        N = batch_stats.batch_size
        n = batch_stats.sample_size
        q = (n - reduced_packets) / N
        d = batch_stats.sample_card - reduced_elements

        uj1 = d / (1 - ((1 - q) * f_1 / n))

        gamma_square = (uj1 / n ** 2) * sum([i * (i - 1) * f_i[i] for i in f_i.keys()]) + uj1 / N - 1
        gamma_square = max(0, gamma_square)

        d_hat = uj1 / d * (d - f_1 * (1 - q) * np.log(1 - q) * gamma_square / q)

        return d_hat + reduced_elements

#######################################################################################################################
# Regressors
#######################################################################################################################


class Regressor:
    """
    Online ML Regressor base class.
    """
    def __init__(self, loss=None, d_loss=None, fit_intercept=True, tol=10 ** -3, max_iter=1000, verbose=False):
        self.w = None
        self.loss = loss
        self.d_loss = d_loss
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        """
        train for max_iter epochs.
        :param X: features vector.
        :param y: labels vector.

        """
        return self._fit_n(X, y, self.max_iter)

    def partial_fit(self, X, y):
        """
        preform one epoch over small or one training example.
        :param X: features vector.
        :param y: labels vector.

        """
        return self._fit_n(X, y, 1)

    def _fit_n(self, X, y, max_iter):
        """
        fit until max_iter epochs or convergence ((prev_loss - sum_loss) < self.tol * X.shape[0])
        :param X: features vector.
        :param y: labels vector.

        """

        if type(X) is not np.ndarray:
            X = np.asarray(X)

        if type(y) is not np.ndarray:
            y = np.asarray(y)

        if len(X.shape) != 2:
            raise ValueError('invalid X shape')

        if self.fit_intercept:
            # append constant intercept term
            ones = np.ones((X.shape[0], X.shape[1] + 1))
            ones[:, 1:] = X
            X = ones

        if self.w is None:
            self.w = np.zeros(X.shape[1])

        prev_loss = float('Inf')

        for epoch in range(max_iter):

            sum_loss = 0
            for i in range(X.shape[0]):
                self._fit(X[i, :], y[i])

                if self.loss is not None:
                    sum_loss += self.loss(X[i, :], y[i])

            if self.loss is not None:
                if (prev_loss - sum_loss) < self.tol * X.shape[0]:
                    if self.verbose:
                        print('Converged after %d epochs.' % epoch)

                    break

            prev_loss = sum_loss

        return self.w

    def _fit(self, X, y):
        """
        abstract method. One fit iteration, over one example. Usually implemented by a Regressor class.
        """
        raise NotImplementedError

    def predict(self, X):

        # append constant intercept term
        if self.fit_intercept:
            ones = np.ones((X.shape[0], X.shape[1] + 1))
            ones[:, 1:] = X
            X = ones

        return [np.dot(self.w, x) for x in X]


class SGDRegressor(Regressor):
    """
    Stochastic Gradient descent regressor.
    """

    MAX_DLOSS = 10 ** 12

    def __init__(self, learning_rate=10 ** -6, **kwargs):

        super().__init__(loss=self._squared_loss, d_loss=self._d_squared_loss, **kwargs)
        self.learning_rate = learning_rate

    def _squared_loss(self, X, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return 0.5 * (np.dot(self.w, X) - y) * (np.dot(self.w, X) - y)

    def _d_squared_loss(self, X, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return np.dot(self.w, X) - y

    def _fit(self, X, y):

        d_loss = self._d_squared_loss(X, y)

        if d_loss < -self.MAX_DLOSS:
            d_loss = -self.MAX_DLOSS
        elif d_loss > self.MAX_DLOSS:
            d_loss = self.MAX_DLOSS

        grad = d_loss * X
        self.w -= self.learning_rate * grad


class PARegressor(Regressor):
    """
    Passive Agressive regressor.
    See - K. Crammer, O. Dekel, J. Keshet, S. Shalev-Shwartz, and Y. Singer. Online passive-aggressive algorithms.
    Journal of Machine Learning Research, 7(Mar), 2006.
    """

    def __init__(self, C=1.0, epsilon=0.1, **kwargs):
        super().__init__(loss=self._epsilon_insensitive_loss, **kwargs)

        self.C = C
        self.epsilon = epsilon

    def _epsilon_insensitive_loss(self, X, y):
        loss = np.abs(np.dot(self.w, X) - y)
        if loss <= self.epsilon:
            return 0
        else:
            return loss - self.epsilon

    def _fit(self, X, y):

        sign = np.sign(y - np.dot(self.w, X))
        loss = self._epsilon_insensitive_loss(X, y)
        tau = loss / (np.linalg.norm(X) ** 2 + 1 / (2 * self.C))
        self.w += sign * tau * X


class RLSRegressor(Regressor):
    """
    Recrusive Least Squares Regressor.
    """
    def __init__(self, mu=0.99, epsilon=0.1 ):
        """
        `mu` : forgetting factor (float). It is introduced to give exponentially
          less weight to older error samples. It is usually chosen
          between 0.98 and 1.

        `epsilon` : initialisation value (float). It is usually chosen
          between 0.1 and 1.

        """
        super().__init__(loss=None, fit_intercept=False)
        self.w = None
        self.mu = mu
        self.epsilon = epsilon
        self.R = None

    def fit(self, X, y):
        # In RLS we only make one iteration
        super()._fit_n(X, y, 1)

    def _fit(self, X, y):
        if self.R is None:
            self.R = 1 / self.epsilon * np.identity(X.shape[0])

        y_hat = np.dot(self.w, X)
        error = y - y_hat
        R1 = np.dot(np.dot(np.dot(self.R, X), X.T), self.R)
        R2 = self.mu + np.dot(np.dot(X, self.R), X.T)
        self.R = 1 / self.mu * (self.R - R1 / R2)
        dw = np.dot(self.R, X.T) * error
        self.w += dw

    def reset(self):
        if self.R is not None:
            self.R = 1 / self.epsilon * np.identity(self.R.shape[0])

    def get_md(self, X):
        return np.sqrt(np.dot(X.transpose(), np.dot(self.R, X)))


class ADAMRegressor(Regressor):
    """
    ADAM Regressor.
    See - D. P. Kingma and J. Ba. Adam: A method for stochastic optimization.
    arXiv preprint arXiv:1412.6980, 2014.
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=10**-8, **kwargs):
        super().__init__(loss=self._squared_loss, d_loss=self._d_squared_loss, **kwargs)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = -1
        self.m_t = None
        self.v_t = None

    def _squared_loss(self, X, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return 0.5 * (np.dot(self.w, X) - y) * (np.dot(self.w, X) - y)

    def _d_squared_loss(self, X, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return np.dot(self.w, X) - y

    def _fit(self, X, y):

        if self.m_t is None:
            self.m_t = np.zeros(X.shape[1])
        if self.v_t is None:
            self.v_t = np.zeros(X.shape[1])

        g_t = self.d_loss(X, y) * X
        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * g_t
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * g_t * g_t
        m_cap = self.m_t / (1 - (self.beta1 ** self.t))
        v_cap = self.v_t / (1 - (self.beta2 ** self.t))
        self.w -= (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)


class NAGRegressor(Regressor):

    def __init__(self, learning_rate = 0.1, **kwargs):
        super().__init__(loss=self._squared_loss, d_loss=self._d_squared_loss, **kwargs)
        self.learning_rate = learning_rate
        self.s_i = None
        self.G_i = None
        self.N = 0
        self.t = 0

    def _fit(self, X, y):

        if self.s_i is None:
            self.s_i = np.zeros(X.shape[1])
        if self.G_i is None:
            self.G_i = np.zeros(X.shape[1])

        self.t += 1

        for i in range(len(X)):
            if X[i] > self.s_i[i]:
                self.w[i] = self.w[i]*self.s_i[i]/abs(X[i])
                self.s_i[i] = abs(self.X[i])

        self.N += np.sum((X ** 2) / (self.s_i ** 2))
        self.G_i += self._d_squared_loss(X, y) ** 2
        self.w -= self.learning_rate * np.sqrt(self.t/self.N) * self.s_i * np.sqrt(self.G_i) * self._d_squared_loss(X, y)

    def _squared_loss(self, X, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return 0.5 * (np.dot(self.w, X) - y) * (np.dot(self.w, X) - y)

    def _d_squared_loss(self, X, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return np.dot(self.w, X) - y

#######################################################################################################################
# Online ML Estimators
#######################################################################################################################


class MLEstimator(Estimator):
    """
    Abstract class for online ML estimator
    """
    def __init__(self, name, model, training_rate, features, training_delay):
        super().__init__(name)
        self.features = features
        self.model = model
        self.training_rate = training_rate
        self.training_step = int(1 / training_rate)
        self.batch_counter = 0

        if training_delay > self.training_step:
            raise ValueError('Training delay is bigger than training step')

        self.training_delay = training_delay
        self._delay_activated = False
        self._training_delay_count = training_delay

    def estimate(self, batch_stats):

        features = batch_stats.get_features(self.features)

        if self.batch_counter == 0:
            # in the first example, first perform a fit
            self.model.fit(features, [batch_stats.batch_card])

        estimation = self.model.predict(features)[0]

        if (self.batch_counter % self.training_step) == 0 and self.batch_counter != 0:
            self._delay_activated = True

        # this is used to simulate a delayed training phase
        if self._delay_activated:
            if self._training_delay_count > 0:
                self._training_delay_count -= 1
            elif self._training_delay_count == 0:
                self.model.partial_fit(features, [batch_stats.batch_card])
                self._delay_activated = False
                self._training_delay_count = self.training_delay

        self.batch_counter += 1

        return estimation


class SGD(MLEstimator):
    def __init__(self, name, features, training_rate, training_delay=1, **kwargs):
        model = SGDRegressor(**kwargs)
        super().__init__(name, model, training_rate, features, training_delay)


class PA(MLEstimator):
    def __init__(self, name, features, training_rate, training_delay=1, **kwargs):
        model = PARegressor(**kwargs)
        super().__init__(name, model, training_rate, features, training_delay)


class RLS(MLEstimator):
    def __init__(self, name, features, training_rate, training_delay=1, **kwargs):
        model = RLSRegressor(**kwargs)
        super().__init__(name, model, training_rate, features, training_delay)


class ADAM(MLEstimator):
    def __init__(self, name, features, training_rate, training_delay=1, **kwargs):
        model = ADAMRegressor(**kwargs)
        super().__init__(name, model, training_rate, features, training_delay)


class NAG(MLEstimator):
    def __init__(self, name, features, training_rate, training_delay=1, **kwargs):
        model = NAGRegressor(**kwargs)
        super().__init__(name, model, training_rate, features, training_delay)


#######################################################################################################################
# Active Online ML
#######################################################################################################################


class ActiveEstimator(Estimator):
    """
    base class for active learning estimator. It trains over batches whose
    Mahalanobis distance (https://en.wikipedia.org/wiki/Mahalanobis_distance) is larger then md_threshold.
    """

    def __init__(self, name, model, md_threshold, features, mu=0.99, epsilon=0.1, debug=False,
                 outlier_threshold=0):
        super().__init__(name)
        self.features = features
        self.model = model
        self.md_threshold = md_threshold
        self.batch_counter = 0
        self.train_counter = 0
        self.outlier_counter = 0
        self.sum_features = None
        self.count_features = 0
        self.outlier_threshold = outlier_threshold
        self.debug = debug
        self.cov_mat = None
        self.mu = mu
        self.epsilon = epsilon

        if self.debug:
            self.train_history = []
            self.md_values = []

    def estimate(self, batch_stats):

        features = batch_stats.get_features(self.features)

        if self.sum_features is None:
            self.sum_features = features
        else:
            self.sum_features += features

        self.count_features += 1

        if self.batch_counter == 0:
            # in the first example first perform a fit
            self.model.fit(features, [batch_stats.batch_card])

        estimation = self.model.predict(features)[0]

        # init cov_mat
        if self.cov_mat is None:
            self.reset_cov_mat(features)

        # if Mahalanobis distance is bigger than threshold, train.
        md = self.get_md(features)
        if self.debug:
            self.md_values.append(md)

        if md > self.md_threshold:
            self.outlier_counter += 1
        else:
            self.outlier_counter = 0

        if self.outlier_counter > self.outlier_threshold:
            self.model.partial_fit(features, [batch_stats.batch_card])
            self.train_counter += 1
            if self.debug:
                self.train_history.append(self.batch_counter)
            self.reset_cov_mat(features)
            self.sum_features = None
            self.count_features = 0
            self.outlier_counter = 0
        self.batch_counter += 1
        return estimation

    def get_md(self, features):
        return float(np.sqrt(np.dot(features.transpose(), np.dot(self.cov_mat, features))))

    def reset_cov_mat(self, features):
        self.cov_mat = 1 / self.epsilon * np.identity(features.shape[0])
        self.update_cov_mat(features)

    def get_mean(self):
        return self.sum_features/self.count_features

    def update_cov_mat(self, features):
        R1 = np.dot(np.dot(np.dot(self.cov_mat, features), features.T), self.cov_mat)
        R2 = self.mu + np.dot(np.dot(features, self.cov_mat), features.T)
        self.cov_mat = 1 / self.mu * (self.cov_mat - R1 / R2)


class RLSA(ActiveEstimator):
    """
    Recursive Least Squares active estimator.
    """
    def __init__(self, name, features, mu=0.99, epsilon=0.1, md_threshold=0.4, debug=False):
        model = RLSRegressor(mu, epsilon)
        super().__init__(name, model, md_threshold, features, mu, epsilon, debug)


class PAA(ActiveEstimator):
    """
    Passive Aggresive active estimator.
    """
    def __init__(self, name, features, mu=0.99, epsilon=0.1, md_threshold=0.4, debug=False, pa_c=1.0, pa_epsilon=0.1):
        model = PARegressor(pa_c, pa_epsilon)
        super().__init__(name, model, md_threshold, features, mu, epsilon, debug)


