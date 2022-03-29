from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import IMLearn.utils.utils as utils
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from IMLearn.learners.regressors.linear_regression import LinearRegression

pio.templates.default = "simple_white"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    houses_df = pd.read_csv(filename)
    # cleaning bad data
    empty_index = np.where(pd.isnull(houses_df))[0]
    houses_df.drop(empty_index, axis=0, inplace=True)
    bad_index = houses_df[(houses_df['price'] <= 0) | (houses_df['sqft_lot15'] <= 0)].index
    houses_df.drop(bad_index, axis=0, inplace=True)
    # combined renovation date and year built to more recent, and converted to age
    houses_df['yr_built'].replace(2015 - np.maximum(houses_df['yr_built'], houses_df['yr_renovated']), inplace=True)
    houses_df.rename(columns={}, inplace=True)
    # there is a large correlation between month sold and price
    houses_df['date'] = houses_df['date'].str[0:6]
    # lat in (47.1559,47.776) long in (-121.315, -122.519), there is a large correlation between location and price
    # so I divided location into 9 square areas
    houses_df['lat'] = pd.qcut(houses_df['lat'], 3, labels=['1','2','3'])
    houses_df['long'] = pd.qcut(houses_df['long'], 3, labels=['a','b','c'])
    houses_df['lat'] = houses_df['lat'].astype(str) + houses_df['long'].astype(str)
    houses_df['zipcode'] = pd.cut(houses_df['zipcode'], 10, labels=[1,2,3,4,5,6,7,8,9,10])
    # yes or no basement
    houses_df['sqft_basement'] = (houses_df['sqft_basement'] > 0).astype(int)
    # floors and conditions are not linear
    houses_df.rename(columns={'lat': 'area', 'yr_built':'age', 'sqft_basement': 'basement Y/N'}, inplace=True)
    price = houses_df['price']
    # houses_df = pd.get_dummies(houses_df, columns=['zipcode','date', 'area','floors','condition', 'view'])
    houses_df = pd.get_dummies(houses_df, columns=['zipcode', 'date', 'area'])
    # combining waterfront and view
    # houses_df.drop(['id', "yr_renovated", 'long',
    #                 'view_0.0','price'], axis=1, inplace=True)
    houses_df.drop(['id', "yr_renovated", 'long',
                     'price'], axis=1, inplace=True)
    return [houses_df, price]



def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    calc = lambda t, s: ((np.cov(t, s)) / (np.std(t) * np.std(s)))[0][1]
    corr = np.array([calc(X[col], y) for col in X.columns])
    graphs = np.array([[X[col], y] for col in X.columns])
    fig = make_subplots(rows=60, cols=1)
    print(graphs.shape)
    for i,g in enumerate(graphs):
        print(i)
        px.scatter(x=g[0], y=[g[1]]).show()

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("..\datasets\house_prices.csv")
    # print(X.head(), "\n\n\n")
    # print(X.columns.size)

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X,y, "exercise_2_plots")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = utils.split_train_test(X,y)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    n = train_X.shape[0]
    model = LinearRegression()
    per_los = np.zeros([91,2])
    std_los = np.zeros(91)
    for k,i in enumerate(range(10, 101)):
        cur_los = np.zeros(10)
        for j in range(10):
            idx = np.random.choice(n, size=int(n * (i/100)))
            x_sample = train_X.iloc[idx]
            y_sample = train_y.iloc[idx]
            model.fit(x_sample, y_sample)
            cur_los[j] = model.loss(x_sample, y_sample)
        std_los[k] = 2 * np.std(cur_los)
        per_los[k][0] = cur_los.mean()
        per_los[k][1] = i
    # fig = px.line(per_los)
    per_los=per_los.transpose()

    # fig = go.Figure([go.Scatter(x=per_los[1], y=per_los[0], mode='lines')])
    fig = go.Figure([go.Scatter(x=per_los[1],
                                y=per_los[0],
                                mode='lines'),
                     go.Scatter(x=per_los[1],
                                y=per_los[0]+std_los,
                                mode='lines',
                                marker=dict(color='#444'),
                                line=dict(width=1),
                                showlegend=False),
                     go.Scatter(x=per_los[1],
                                y=per_los[0] - std_los,
                                mode='lines',
                                marker=dict(color='#444'),
                                line=dict(width=1),
                                showlegend=False,
                                fillcolor='rgba(68, 68, 68, 0.3)',
                                fill='tonexty')])
    fig.show()







