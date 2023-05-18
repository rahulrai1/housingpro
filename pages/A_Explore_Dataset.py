import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import os
import tarfile
import urllib.request
from itertools import combinations

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

feature_lookup = {
    'BOROUGH': '**BOROUGH**',
    'NEIGHBORHOOD': '**NEIGHBORHOOD**',
    'BUILDING CLASS CATEGORY': '**BUILDING CLASS CATEGORY**',
    'TOTAL UNITS': '**TOTAL UNITS**',
    'SALE PRICE': '**SALE PRICE**',
    'SALE DATE': '**SALE DATE**',
    'AVERAGE PRICE': '**AVERAGE PRICE**',

}
#############################################

st.markdown("## AI-Powered Housing Price Prediction Platform")

st.markdown('## Explore Dataset')

#############################################

st.markdown('### Import Dataset')


def load_dataset(data):
    df = pd.read_csv(data, low_memory=False)
    return df


# Checkpoint 1
def compute_correlation(X, features):
    correlation = X[features].corr()
    cor_summary_statements = []

    # Add code here
    for i, j in combinations(features, 2):
        corr_value = correlation.loc[i, j]
        magnitude = "strongly" if abs(corr_value) > 0.5 else "weakly"
        direction = "positively" if corr_value > 0 else "negatively"
        statement = f"Features {i} and {j} are {magnitude} {direction} correlated: {corr_value:.2f}"
        cor_summary_statements.append('- ' + statement)  # added '- ' +

    return correlation, cor_summary_statements


# Helper Function
def user_input_features(df, chart_type, x=None, y=None):
    side_bar_data = {}
    select_columns = []
    if x is not None:
        select_columns.append(x)
    if y is not None:
        select_columns.append(y)
    if x is None and y is None:
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        unique_key = f"{chart_type}_{feature}_{idx}"
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=unique_key
            )
        except Exception as e:
            print(e)
            f = (float(df[str(feature)].min()), float(df[str(feature)].max()))
        side_bar_data[feature] = f
    return side_bar_data


# Helper Function
def display_features(df, feature_lookup):
    numeric_columns = list(df.columns)
    # for idx, col in enumerate(df.columns):
    for idx, col in enumerate(numeric_columns):
        if col in feature_lookup:
            st.markdown('Feature %d - %s' % (idx, feature_lookup[col]))
        else:
            st.markdown('Feature %d - %s' % (idx, col))


# Helper Function
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedir(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Helper function
@st.cache_data
def convert_df(df):
    """
    Cache the conversion to prevent computation on every rerun

    Input: 
        - df: pandas dataframe
    Output: 
        - Save file to local file system
    """
    return df.to_csv().encode('utf-8')


###################### FETCH DATASET #######################
df = None
col1, col2 = st.columns(2)

with(col1):
    data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'], key="fileUploader")

with(col2):
    data_path = st.text_input("Enter dataset URL",
                              "",
                              key="dataset_url", )
    if (data_path):
        fetch_housing_data()
        data = os.path.join(HOUSING_PATH, "housing.csv")
        st.write("You entered: ", data_path)

if data is not None:
    df = load_dataset(data)

    st.session_state['house_df'] = df

if df is not None:
    ###################### EXPLORE DATASET #######################
    st.markdown('### Explore Dataset Features')

    # Display feature names and descriptions (from feature_lookup)
    display_features(df, feature_lookup)

    st.dataframe(df.describe())
    X = df
    ###################### VISUALIZE DATASET #######################
    # st.markdown('### Visualize Features')
    #
    numeric_columns = list(df.select_dtypes(['float', 'int']).columns)

    # Specify Input Parameters
    # st.sidebar.header('Specify Input Parameters')
    #
    # # Collect user plot selection
    # st.sidebar.header('Select type of chart')
    # chart_select = st.sidebar.selectbox(
    #     label='Type of chart',
    #     options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    # )

    # Draw plots
    # if chart_select == 'Scatterplots':
    #     x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
    #     y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
    #     side_bar_data = user_input_features(df, chart_select, x_values, y_values)
    #     plot = px.scatter(data_frame=df, x=x_values, y=y_values, range_x=side_bar_data[x_values],
    #                       range_y=side_bar_data[y_values])
    #     st.write(plot)
    # else:
    #     x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
    #     y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
    #     side_bar_data = user_input_features(df, chart_select, x_values, y_values)
    #
    #     if chart_select == 'Histogram':
    #         plot = px.histogram(data_frame=df[df[x_values].between(*side_bar_data[x_values])], x=x_values)
    #         st.write(plot)
    #     elif chart_select == 'Lineplots':
    #         plot = px.line(
    #             df[df[x_values].between(*side_bar_data[x_values]) & df[y_values].between(*side_bar_data[y_values])],
    #             x=x_values, y=y_values)
    #         st.write(plot)
    #     elif chart_select == 'Boxplot':
    #         plot = px.box(
    #             df[df[x_values].between(*side_bar_data[x_values]) & df[y_values].between(*side_bar_data[y_values])],
    #             x=x_values, y=y_values)
    #         st.write(plot)

    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### Looking for Correlations")

    # Collect features for correlation analysis using multiselect
    select_features_for_correlation = st.multiselect(
        'Select secondary features for visualize of correlation analysis (up to 4 recommended)',
        X.numeric_columns,
    )

    if select_features_for_correlation:
        correlation, cor_summary_statements = compute_correlation(X, select_features_for_correlation)

        # Display summary statements
        st.markdown("#### Correlation Summary")
        for summary in cor_summary_statements:
            st.markdown(f"- {summary}")

        # Display correlation table
        st.markdown("#### Correlation Table")
        st.dataframe(correlation)

        # Display scatter matrix plot
        try:
            fig = scatter_matrix(df[select_features_for_correlation], figsize=(12, 8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)

    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',
    )

    st.markdown('#### Continue to Preprocess Data')
