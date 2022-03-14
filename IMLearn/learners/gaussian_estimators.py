from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # m = X.size
        # self.mu_ = X.sum() / m
        # if self.biased_:
        #     self.var_ =(((X - self.mu_)*(X - self.mu_)).sum()) / m
        # else:
        #     self.var_ = (((X - self.mu_) * (X - self.mu_)).sum()) / (m - 1)
        # raise NotImplementedError()
        if self.biased_:
            self.var_ = X.var(ddof=0)
        else:
            self.var_ = X.var(ddof=1)
        self.mu_ = X.mean()
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        # f(x) = (1/sqrt(2*pi*var)) * exp(-()x-mu)^2) / 2*var
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        PDF = np.ndarray(X.size)
        for i in range(X.size):
            mone = np.exp((-(X[i]-(self.mu_**2))) / (2 * self.var_))
            PDF[i] = mone / np.sqrt(2 * np.pi * self.var_)
        return PDF




    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # likelihood: (1/ (2*pi*variance)^(m/2)) * exp(-1/2var * (sum(X-mu)^2)
        # log-likelihood = log ((1/ (2*pi*variance)^(m/2)) -(1/2var * sum(X-mu)^2) =
        # -n/2 * log(2pi) - n/2 * log(var) - (1/2var) * sum ((x-mu)^2)
        n = X.size
        s = ((X - mu)**2).sum()
        return (-n/2) * np.log(np.pi) - (n/2) * np.log(sigma) - (1/2*sigma) * s


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        mu should be of shape (n_features,)
        cov should be of shape (n_features, n_features)
        X should be of shape (n_samples, n_features)
        """
        #
        self.cov_ = np.cov(X, bias=False, ddof=1)
        self.mu_ = np.mean(X, axis=1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        d = X[0].size
        inverse = np.linalg.inv(self.cov_)
        return 1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov_)) * \
               np.exp((-1 / 2) * (X - self.mu_).transpose().dot(inverse).dot((X - self.mu)))



    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        raise NotImplementedError()
