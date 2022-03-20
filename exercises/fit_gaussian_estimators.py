from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu1 = 10
    sigma1 = 1
    X1 = np.random.normal(mu1, sigma1, 1000)
    UG = UnivariateGaussian()
    UG.fit(X1)
    print("expectation1: ", UG.mu_, "variance1: ", UG.var_, "\n")
    # Question 2 - Empirically showing sample mean is consistent
    arr = np.ndarray(100)
    for i, smp in enumerate(range(10, 1001, 10)):
        temp = UnivariateGaussian()
        temp.fit(X1[:smp])
        arr[i] = np.abs(temp.mu_ - mu1)
    px.scatter(x=np.array(list(range(10, 1001, 10))), y=arr, title="question 2").update_xaxes \
        (title_text="sample size").update_yaxes(title_text="|estimated - true value of expectation|").show()
    # Question 3 - Plotting Empirical PDF of fitted model
    px.scatter(x=X1, y=UG.pdf(X1), title="question 3").update_xaxes \
        (title_text="ordered sample values").update_yaxes(title_text="PDF").show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu2 = np.array([0, 0, 4, 0])
    cov_matrix = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X2 = np.random.multivariate_normal(mu2, cov_matrix, 1000)
    MG = MultivariateGaussian()
    MG.fit(X2)
    print("expectation2: \n", MG.mu_, "\n", "covar matrix: \n", MG.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_matrix = np.array([[MultivariateGaussian.log_likelihood(np.array([f1[i], 0, f3[j], 0]), cov_matrix, X2)
                            for j in range(200)] for i in range(200)])
    px.imshow(log_matrix, x=f1, y=f3, labels=dict(x="f1", y="f3", color="log-likelihood"), title="question 5").show()
    # Question 6 - Maximum likelihood
    max_val = np.amax(log_matrix)
    max_location = np.where(log_matrix == max_val)
    print("max log-likelihood value: ", max_val,"f1 value: ", f1[max_location[0][0]],
          "f3 value:  ", f3[max_location[1][0]])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
