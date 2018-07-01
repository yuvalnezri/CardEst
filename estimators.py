from scipy.optimize import brentq
import numpy as np
from collections import Counter


class Estimator:
    def __init__(self, name):
        self.name = name

    def estimate(self, batch_stats):
        raise NotImplementedError


#######################################################################################################################
# Statistical Estimators
#######################################################################################################################


class NaiveLearn(Estimator):
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


class GT(Estimator):
    def __init__(self, name):
        super(GT, self).__init__(name)

    def estimate(self, batch_stats):
        freq1 = batch_stats.histogram[1]
        if freq1 == batch_stats.sample_card:
            return batch_stats.sample_card
        return batch_stats.sample_card * (1 / (1 - freq1 / batch_stats.sample_size))


class GEE(Estimator):
    def __init__(self, name):
        super(GEE, self).__init__(name)

    def estimate(self, batch_stats):
        freq1 = batch_stats.histogram[1]
        return np.sqrt(1 / batch_stats.sampling_rate) * freq1 + (sum(batch_stats.histogram.values()) - freq1)


class AE(Estimator):
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
    def __init__(self, loss=None, d_loss=None, fit_intercept=True, tol=10 ** -3, max_iter=1000, verbose=False):
        self.w = None
        self.loss = loss
        self.d_loss = d_loss
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y):
        return self._fit_n(X, y, self.max_iter)

    def partial_fit(self, X, y):
        return self._fit_n(X, y, 1)

    def _fit_n(self, X, y, max_iter):

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
            for i in range(X.shape[0]):

                sum_loss = 0

                self._fit(X[i, :], y[i])

                if self.loss is not None:
                    sum_loss += self.loss(X[i, :], y[i])

            if self.loss is not None:
                if sum_loss > prev_loss - self.tol * X.shape[0]:
                    if self.verbose:
                        print('Converged after %d epochs.' % epoch)

                    break

                prev_loss = sum_loss

        return self.w

    def _fit(self, X, y):
        """
        One fit iteration, over one example.
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
    def __init__(self, mu=0.99, epsilon=0.1):
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


class ADAMRegressor(Regressor):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=10**-8, **kwargs):
        super().__init__(loss=self._squared_loss, d_loss=self._d_squared_loss, **kwargs)
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = -1
        self.m_t = None
        self.v_t = None

    def _squared_loss(self, x, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return 0.5 * (np.dot(self.w, x) - y) * (np.dot(self.w, x) - y)

    def _d_squared_loss(self, x, y):

        if self.w is None:
            raise RuntimeError('w not inited')

        return np.dot(self.w, x) - y

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


#######################################################################################################################
# Online ML Estimators
#######################################################################################################################


class MLEstimator(Estimator):
    def __init__(self, name, model, training_rate, features):
        super(MLEstimator, self).__init__(name)
        self.features = features
        self.model = model
        self.training_rate = training_rate
        self.training_step = int(1 / training_rate)
        self.batch_counter = 0

    def estimate(self, batch_stats):

        features = batch_stats.get_features(self.features)

        if self.batch_counter == 0:
            # in the first example first perform a fit
            self.model.fit(features, [batch_stats.batch_card])

        estimation = self.model.predict(features)[0]

        if (self.batch_counter % self.training_step) == 0 and self.batch_counter != 0:
            self.model.partial_fit(features, [batch_stats.batch_card])

        self.batch_counter += 1

        return estimation


class SGD(MLEstimator):
    def __init__(self, name, features, training_rate, **kwargs):
        model = SGDRegressor(**kwargs)
        super().__init__(name, model, training_rate, features)


class PA(MLEstimator):
    def __init__(self, name, features, training_rate, **kwargs):
        model = PARegressor(**kwargs)
        super().__init__(name, model, training_rate, features)


class RLS(MLEstimator):
    def __init__(self, name, features, training_rate, **kwargs):
        model = RLSRegressor(**kwargs)
        super().__init__(name, model, training_rate, features)


#######################################################################################################################
# Active Online ML Estimators
#######################################################################################################################


class ActiveEstimator(Estimator):
    def __init__(self, name, model, threshold, features):
        super().__init__(name)
        self.features = features
        self.model = model
        self.threshold = threshold
        self.batch_counter = 0
        self.train_counter = 0
        self.train_history = []

    def get_uncertainty(self, batch_stats):
        raise NotImplementedError

    def estimate(self, batch_stats):

        features = batch_stats.get_features(self.features)

        if self.batch_counter == 0:
            # in the first example first perform a fit
            self.model.fit(features, [batch_stats.batch_card])

        estimation = self.model.predict(features)[0]

        # if uncertainty is bigger than threshold, train
        if self.get_uncertainty(batch_stats) > self.threshold:
            self.model.partial_fit(features, np.asarray([batch_stats.batch_card]))
            self.train_counter += 1
            self.train_history.append(self.batch_counter)

        self.batch_counter += 1
        return estimation


class RLSA(ActiveEstimator):
    def __init__(self, name, features, threshold, mu=0.99, cr=None):
        self.mu = mu
        self.cr = cr

        # debug ############
        self.certainty = []
        self.reset = []
        ####################

        # to align with sklearn's behaviour we only insatciate the model on first estimate,
        # after number of features is known.
        model = None
        super().__init__(name, model, threshold, features)

    def estimate(self, batch_stats):
        if self.model is None:
            n = self.features(batch_stats).shape[1]
            self.model = RLSRegressor(n, self.mu)

        # add CR-RLS functionality, reset R every cr trainings
        if (self.cr is not None) and (self.batch_counter % self.cr == 0):
            self.model.R = 1 / self.model.eps * np.identity(self.model.n)

            # debug ########
            self.reset.append(self.batch_counter)
            ################

        return super().estimate(batch_stats)

    def get_uncertainty(self, batch_stats):
        x = self.features(batch_stats)
        cert = np.dot(x.flatten().transpose(), np.dot(self.model.R, x.flatten()))
        self.certainty.append(cert)
        return cert





