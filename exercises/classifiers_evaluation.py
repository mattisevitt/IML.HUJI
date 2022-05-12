from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics.loss_functions import accuracy


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    def current_callback(fit: Perceptron, x: np.ndarray, y_: int):
        return fit.loss(X,y)

    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # Fit Perceptron and record loss in each fit iteration
        model = Perceptron(callback=current_callback)
        model.fit(X,y)
        # Plot figure of loss as function of fitting iteration
        px.scatter(x=np.arange(len(model.callback_values)),y=model.callback_values, title=n).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # Fit models and predict over training set
        gaussian_model = GaussianNaiveBayes()
        gaussian_model.fit(X,y)
        gauss_prediction = gaussian_model.predict(X)
        lda_model = LDA()
        lda_model.fit(X,y)
        lda_prediction = lda_model.predict(X)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        symbols = np.array(["circle", "square", "triangle-down"])
        fig = make_subplots(rows=1, cols=2,  subplot_titles=(f"Gaussian classifier  Accuracy: {accuracy(y, gauss_prediction)}\n",
                                                             f"LDA classifier  Accuracy: {accuracy(y, lda_prediction)}\n"))

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker=dict(color=gauss_prediction, symbol=symbols[y])), row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker=dict(color=lda_prediction, symbol=symbols[y])), row=1, col=2)
        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=gaussian_model.mu_[:, 0], y=gaussian_model.mu_[:, 1],
                                 mode="markers", marker=dict(color="black", symbol="x")), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_model.mu_[:, 0], y=lda_model.mu_[:, 1],
                                 mode="markers", marker=dict(color="black", symbol="x")), row=1, col=2)
        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(gaussian_model.classes_.size):
            fig.add_trace(get_ellipse(gaussian_model.mu_[i], np.diag(gaussian_model.vars_[i])), row=1, col=1)
        for i in range(lda_model.classes_.size):
            fig.add_trace(get_ellipse(lda_model.mu_[i], lda_model.cov_), row=1, col=2)
        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    # X = np.array([[1,1],[1,2],[2,3],[2,4],[3,3],[3,4]])
    # y = np.array([0,0,1,1,1,1])
    # model = GaussianNaiveBayes()
    # model.fit(X,y)
    # print(model.vars_)
