import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost_ensemble = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    adaboost_ensemble.fit(train_X, train_y)
    amount_of_learners = np.arange(1, n_learners+1)
    training_error_vec = np.array([adaboost_ensemble.partial_loss(train_X, train_y, k) for k in range(1, n_learners+1)])
    test_error_vec = np.array([adaboost_ensemble.partial_loss(test_X, test_y, k) for k in range(1, n_learners+1)])
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=amount_of_learners, y=training_error_vec, name="training error"))
    fig1.add_trace(go.Scatter(x=amount_of_learners, y=test_error_vec, name="test error"))
    fig1.show()



    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    # T = [2, 3, 4, 8]
    names = ["5 Iterations", "50 Iterations", "100 Iterations", "250 Iterations"]
    symbols = np.array(["square","circle", "x"])
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in names],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda x:adaboost_ensemble.partial_predict(x,t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig2.show()
    # Question 3: Decision surface of best performing ensemble
    lowest_test_error = np.argmin(np.array(test_error_vec))
    print(f"Lowest Test Error Achieved With {lowest_test_error} learners")
    best_accuracy = accuracy(test_y, adaboost_ensemble.partial_predict(test_X, lowest_test_error))
    fig3 = go.Figure()
    fig3.add_traces([decision_surface(lambda x:adaboost_ensemble.partial_predict(x,lowest_test_error), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))])
    fig3.update_layout(title=f"Ensemble Size: {lowest_test_error} \n Accuracy: {best_accuracy}")
    fig3.show()


    # Question 4: Decision surface with weighted samples
    d_t = adaboost_ensemble.D_[n_learners-1]/np.max(adaboost_ensemble.D_[n_learners-1]) * 10
    fig4 = go.Figure()
    fig4.add_traces([decision_surface(lambda x: adaboost_ensemble.partial_predict(x, n_learners), lims[0],
                                      lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=train_y.astype(int), symbol=symbols[train_y.astype(int)],
                                            colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1), size=d_t))])
    fig4.show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
