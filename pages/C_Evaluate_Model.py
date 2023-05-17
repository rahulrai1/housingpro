import time
import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import matplotlib.pyplot as plt  # pip install matplotlib
import streamlit as st  # pip install streamlit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pages.B_Preprocess_Data import remove_nans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import random
import plotly.express as px
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly
from prophet.serialize import model_to_json, model_from_json

random.seed(10)
#############################################
st.markdown("## AI-Powered Housing Price Prediction Platform")

#############################################

st.title('Train and Test Model')


#############################################

def restore_dataset():
    df = None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'], key="fileUploader1")
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df
df_boroughs = pd.read_csv('Borough_names.csv', header=0)
df_bronx = pd.read_csv('bronx_neighborhoods.csv', header=0)
df_brooklyn = pd.read_csv('brooklyn_neighborhoods.csv', header=0)
df_manhattan = pd.read_csv('manhattan_neighborhoods.csv', header=0)
df_queens = pd.read_csv('queens_neighborhoods.csv', header=0)
df_staten_island = pd.read_csv('staten_island_neighborhoods.csv', header=0)

#df__main_file=pd.read_csv('NYC_Citywide_Annualized_Calendar_Sales_Update.csv', header=0)
df_houses=pd.read_csv('House_categories.csv', header=0)
houses=df_houses['BUILDING CLASS CATEGORY'].tolist()
boroughs_list=df_boroughs['BOROUGH_NAME'].tolist()
neighborhood_file_dict={
    'BRONX': df_bronx,
    'BROOKLYN': df_brooklyn,
    'MANHATTAN': df_manhattan,
    'QUEENS': df_queens,
    'STATEN ISLAND': df_staten_island
}


# Complete this helper function from HW1
def split_dataset(X, y, number, random_state=45):
    X_train = []
    X_val = []
    y_train = []
    y_val = []

    try:
        # Calculate test_size from the percentage input
        test_size = number / 100

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_percentage = len(X_train) / (len(X_train) + len(X_val)) * 100
        test_percentage = len(X_val) / (len(X_train) + len(X_val)) * 100

        # Print dataset split result
        st.markdown(
            'The training dataset contains sorted by time  {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} observations ({3:.2f}%).'
            .format(
                len(X_train),
                train_percentage,
                len(X_val),
                test_percentage))
        # Save state of train and test splits in st.session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_val'] = X_val
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val
    except:
        print('Exception thrown; testing test size to 0')
    return X_train, X_val, y_train, y_val

#Train the dataset
def train_model(X):    
    df_models = pd.DataFrame(
    columns=['Borough', 'Neighborhood', 'Housing Category', 'model']
    )
    #for borough in boroughs_list:
    
    for borough, df_vals in neighborhood_file_dict.items():
        print(borough+df_vals)
        for neighborhood in df_vals['NEIGHBORHOOD'].tolist():
            for house in houses:
                #st.write(house)
                df_new=X[['AVERAGE PRICE', 'SALE DATE']][(X['BOROUGH']==borough)&(X['NEIGHBORHOOD']==neighborhood)&(X['BUILDING CLASS CATEGORY']==house)]
                prophet_df=(df_new.rename(columns={"SALE DATE": "ds", "AVERAGE PRICE": "y"}))
                #st.write(prophet_df.head())
                try:    
                    model = Prophet()
                    model.fit(prophet_df)
                    file_name="models/"+borough+"_"+neighborhood+"_"+house+".json"
                    with open(file_name, 'w') as fout:
                        fout.write(model_to_json(model))  # Save model
                except:
                    print(borough+" "+neighborhood+" "+house+" does not have enough data to run time series model.")
                df_model={'Borough':borough,'Neighborhood': neighborhood, 'Housing Category': house, 'model': model}                
                df_models = pd.concat([df_models, pd.DataFrame([df_model])], ignore_index=True)
                print(borough+" "+neighborhood+" "+house+" done.")
                #globals()[borough+neighborhood+house]=make_forecast()
    st.session_state['model_list'] = df_models
    return df_models



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
df = restore_dataset()
if df is not None:
    # Display dataframe as table
    st.dataframe(df.describe())

    # Select variable to predict
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=list(df.columns),
        key='feature_selectbox',
        index=6
    )

    st.session_state['target'] = feature_predict_select

    # Select input features
    feature_input_select = st.multiselect(
        label='Select Input features for model',
        options=[f for f in list(df.columns) if f != feature_predict_select],
        key='feature_multiselect'
    )

    st.session_state['feature'] = feature_input_select

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    df = remove_nans(df)
    X = df.loc[:, df.columns.isin(feature_input_select)]
    Y = df.loc[:, df.columns.isin([feature_predict_select])]

    # Split train/test
    st.markdown('## Configure parameters for time-series evaluation')
    st.markdown(
        '### Enter the percentage of test period to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    # Compute the percentage of test and training data
    X_train, X_val, y_train, y_val = split_dataset(X, Y, number)

    # regression_methods_options = ['Multiple Linear Regression',
    #                               'Polynomial Regression', 'Ridge Regression', 'Lasso Regression']

    # # Multiple Linear Regression
    # if (regression_methods_options[0] in regression_model_select):
    #     st.markdown('#### ' + regression_methods_options[0])
    #     if st.button('Train Multiple Linear Regression Model'):
    #         train_multiple_regression(
    #             X_train, y_train, regression_methods_options)
    #
    #     if regression_methods_options[0] not in st.session_state:
    #         st.write('Multiple Linear Regression Model is untrained')
    #     else:
    #         st.write('Multiple Linear Regression Model trained')

    # Store dataset in st.session_state
    # Add code here ...

    if st.button("Train All Data"):
        train_model(df)
    
    if 'model_list' in st.session_state:
        st.markdown("Data has been trained and back tested.")
        models_df=st.session_state['model_list']
    st.markdown('### Evaluate the model')
    borough_name=st.selectbox('Select a borough:', df_boroughs)
    neighborhood_name = st.selectbox('Select a neighborhood:', neighborhood_file_dict[borough_name])
    df_building=df[['BUILDING CLASS CATEGORY']][(df['BOROUGH']==borough_name)&(df['NEIGHBORHOOD']==neighborhood_name)]
    #st.write(df_building.head())
    building_class=st.selectbox('Select the house type:', df_building['BUILDING CLASS CATEGORY'].unique())
    file_name="models/"+borough_name+"_"+neighborhood_name+"_"+building_class+".json"
    with open(file_name, 'r') as fin:
        model_ext = model_from_json(fin.read())
    if model_ext:
        if st.button("Evaluate model"):
            #model_cell=models_df[['model']][(models_df['Borough']==borough_name)&(models_df['Neighborhood']==neighborhood_name)&(models_df['Housing Category']==building_class)]
            #model_cell=models_df[['model']][(models_df['Borough']=='MANHATTAN')&(models_df['Neighborhood']=='FLATIRON')&(models_df['Housing Category']=='CONDOS - ELEVATOR APARTMENTS')]
            #st.write(model_cell)
            #model_ext=model_cell.iloc[0]['model']
            #st.write(model_ext)
            future = model_ext.make_future_dataframe(periods=180)
            future['cap'] = 2000000
            future['floor'] = 100000
            forecast=model_ext.predict(future)
            #df_last=df[['BOROUGH', 'NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'SALE DATE', 'AVERAGE PRICE', 'SALE PRICE', 'TOTAL UNITS']][(df['BOROUGH']==borough_name)&(df['NEIGHBORHOOD']==neighborhood_name)&(df['BUILDING CLASS CATEGORY']==building_class)]
            #st.write(df_last.tail(20))
            df_cv=cross_validation(model_ext,initial='730 days', period='60 days', horizon = '365 days', parallel="processes")
            df_p=performance_metrics(df_cv)
            fig = plot_cross_validation_metric(df_cv, metric='mape')
            fig1=plot_plotly(model_ext, forecast)
            st.write(fig1)
            st.write(fig)
            #st.write(df_p.head())
                        

    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(df)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Train: Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',
    )

    st.write('Continue to Test Model')
