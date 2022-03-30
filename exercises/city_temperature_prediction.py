import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"
pio.templates.default = "simple_white"
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
MONTHS = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
COUNTRIES = np.array(['Israel', 'Jordan', 'South Africa', 'The Netherlands'])
def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    cities_df = pd.read_csv(filename, parse_dates=['Date'])
    cities_df['DayOfYear'] = cities_df['Date'].dt.dayofyear
    bad_index = cities_df[(cities_df['Temp'] < -20)].index
    cities_df.drop(bad_index, axis=0, inplace=True)
    # Temp = cities_df['Temp']
    # cities_df.drop(['Temp'], axis=1, inplace=True)
    return cities_df
if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("..\datasets\City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    not_israel_idx = df[(df['Country'] != 'Israel')].index
    israel_df = df.drop(not_israel_idx, axis=0)
    israel_df['Year'] = israel_df['Year'].astype(str)
    fig1 = px.scatter(israel_df, x='DayOfYear', y='Temp', color='Year')
    fig1.show()
    israel_df_grouped = israel_df.groupby('Month')
    fig2 = go.Figure()
    for month in MONTHS:
        fig2.add_trace(go.Bar(x=[month], y=[np.std(israel_df_grouped.get_group(month)['Temp'])]))
    fig2.update_xaxes(title_text='months', type='category')
    fig2.update_yaxes(title_text='standard deviation of daily temperatures')
    fig2.show()
    # Question 3 - Exploring differences between countries
    df = df.groupby(['Country', 'Month'])
    new_df = pd.DataFrame()
    fig3 = go.Figure()
    for country in COUNTRIES:
        average_temp = np.zeros(12)
        dev_temp = np.zeros(12)
        for month in MONTHS:
            average_temp[month-1] = df.get_group((country,month))['Temp'].mean()
            dev_temp[month-1] = np.std(df.get_group((country,month))['Temp'])
        fig3.add_trace(go.Scatter(x=MONTHS, y=average_temp, error_y=dict(type='data', array=dev_temp, visible=True), name=country))
    fig3.update_xaxes(title_text='months', type='category')
    fig3.update_yaxes(title_text='Average Temperature')
    fig3.show()

    # Question 4 - Fitting model for different values of `k`

    X_train, y_train, X_test, y_test = split_train_test(israel_df['DayOfYear'], israel_df['Temp'], train_proportion=.75)
    fig4 = go.Figure()
    for k in np.arange(1,11):
        model4 = PolynomialFitting(k)
        model4.fit(X_train, y_train)
        cur_los = model4.loss(X_test, y_test)
        print("for value k={0} the loss is {1}".format(k, round(cur_los, 2)))
        fig4.add_trace(go.Bar(x=[k], y=[round(cur_los, 2)]))
    fig4.update_xaxes(title_text='k', type='category')
    fig4.update_yaxes(title_text='loss')
    fig4.show()
    # Question 5 - Evaluating fitted model on different countries
    model5 = PolynomialFitting(5)
    model5.fit(israel_df['DayOfYear'], israel_df['Temp'])


