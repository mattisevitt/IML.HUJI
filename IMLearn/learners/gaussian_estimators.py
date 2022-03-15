from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet
from numpy import transpose
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


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
            PDF[i] = np.exp((-(X[i]-(self.mu_))**2) / (2 * self.var_)) / np.sqrt(2 * np.pi * self.var_)
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
        self.cov_ = np.cov(X, bias=False, ddof=1, rowvar=False)
        self.mu_ = np.mean(X, axis=0)
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
        inverse = inv(self.cov_)
        return 1 / np.sqrt(((2 * np.pi) ** d) * det(self.cov_)) * \
               np.exp((-0.5) * (X - self.mu_).sum().dot(inverse).dot((X - self.mu)))


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
    m = X.shape[0]
    md = X.size
    B = 0
    for i in range(m):
        B += np.linalg.multi_dot([(X[i] - mu).transpose(), inv(cov), (X[i] - mu)])
    A = md*np.log(2*np.pi) + m*np.log(det(cov))
    return -0.5 * (A + B )


if __name__ == '__main__':
    # q1
    mu1 = 10
    sigma1 = 1
    X1 = np.random.normal(mu1, sigma1, 1000)
    UG = UnivariateGaussian()
    UG.fit(X1)
    print("expectation1: ", UG.mu_, "variance1: ", UG.var_, "\n")
    arr = np.ndarray(100)
    for i, smp in enumerate(range(10, 1001, 10)):
        temp = UnivariateGaussian()
        temp.fit(X1[:smp])
        arr[i] = np.abs(temp.mu_ - mu1)
    # q2
    px.scatter( x= np.array(list(range(10,1001, 10))), y = arr, title="question 2").update_xaxes\
        (title_text="sample size").update_yaxes(title_text="|estimated - true value of expectation|").show()
    # # q3
    px.scatter(x=X1, y=UG.pdf(X1), title="question 3").update_xaxes\
        (title_text="ordered sample values").update_yaxes(title_text="PDF").show()
    # q4
    mu2 = np.array([0,0,4,0])
    cov_matrix = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X2 = np.random.multivariate_normal(mu2, cov_matrix, 1000)
    MG = MultivariateGaussian()
    MG.fit(X2)
    print("expectation2: \n", MG.mu_, "\n", "covar matrix: \n", MG.cov_)
    # q5
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_matrix = np.array([[log_likelihood(np.array([f1[i], 0, f3[j], 0]), cov_matrix, X2)
                            for j in range(200)] for i in range(200)])
    px.imshow(log_matrix, x=f1, y=f3, labels=dict(x="f1", y="f3", color="log-likelihood"), title="question 5").show()
    # q6
    max_val = np.amax(log_matrix)
    max_location = np.where(log_matrix == max_val)
    print("max log-likelihood value: ", max_val, "f1 index: ", max_location[0][0], "f3 index: ", max_location[1][0])




