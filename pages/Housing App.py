import streamlit as st
import pandas as pd                     
import numpy as np
from pages.B_Preprocess_Data import remove_nans
import random
from sklearn.preprocessing import OrdinalEncoder
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objs as go
from prophet.plot import plot_plotly
from prophet.serialize import model_to_json, model_from_json

random.seed(10)
#############################################

st.markdown("## AI-Powered Housing Price Prediction Platform")

#############################################

#############################################

st.title('Deploy Application')

#############################################

enc = OrdinalEncoder()
df_boroughs = pd.read_csv('Borough_names.csv', header=0)
df_bronx = pd.read_csv('bronx_neighborhoods.csv', header=0)
df_brooklyn = pd.read_csv('brooklyn_neighborhoods.csv', header=0)
df_manhattan = pd.read_csv('manhattan_neighborhoods.csv', header=0)
df_queens = pd.read_csv('queens_neighborhoods.csv', header=0)
df_staten_island = pd.read_csv('staten_island_neighborhoods.csv', header=0)

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

# Checkpoint 11
def deploy_model(df):
    """
    Deploy trained regression model trained on 
    Input: 
        - df: pandas dataframe with trained regression model including
            number of bedrooms, number of bathrooms, desired city, proximity to water features
    Output: 
        - house_price: predicted house price
    """
    house_price=None
    model=None

    # Add code here

    st.write('deploy_model not implemented yet.')
    return house_price

# Helper Function
def is_valid_input(input):
    """
    Check if the input string is a valid integer or float.

    Input: 
        - input: string, char, or input from a user
    Output: 
        - True if valid input; otherwise False
    """
    try:
        num = float(input)
        return True
    except ValueError:
        return False
    
# Helper Function
def decode_integer(original_df, decode_df, feature_name):
    """
    Decode integer integer encoded feature

    Input: 
        - original_df: pandas dataframe with feature to decode
        - decode_df: dataframe with feature to decode 
        - feature: feature to decode
    Output: 
        - decode_df: Pandas dataframe with decoded feature
    """
    original_dataset[[feature_name]]= enc.fit_transform(original_dataset[[feature_name]])
    decode_df[[feature_name]]= enc.inverse_transform(st.session_state['X_train'][[feature_name]])
    return decode_df

###################### FETCH DATASET #######################

df = None
def restore_dataset(): 
    df = None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'], key="fileUploader2")
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
    else:
        df = st.session_state['house_df']
    
    if df is None:
        st.write("Please upload a dataset to continue.")
    
    return df
df=restore_dataset()
# Restore model from st.session_state[model_name]
# Fill in code below
#if 'data' in st.session_state:
#    # df = ... Add code here: read in data and store in st.session_state
#    st.write('Data import not implemented yet.')
#else:
#    st.write('### The Housing Price Application is under construction. Coming to you soon.')

###################### Deploy App #######################
if 'model_list' in st.session_state:    
    models=st.session_state['model_list']
    st.markdown("##### Model has been deployed")
#else:
#    st.markdown("##Please go back and train the model.")
if df is not None:
    df.dropna(inplace=True)
    st.markdown('## Interested in investing in New York City?')
    st.markdown('#### Predict future housing prices in New York based on neighborhood information, property type, and market trends.')

    #User Selection
    borough_name=st.selectbox('Select a borough:', df_boroughs)
    neighborhood_name = st.selectbox('Select a neighborhood:', neighborhood_file_dict[borough_name])
    df_building=df[['BUILDING CLASS CATEGORY']][(df['BOROUGH']==borough_name)&(df['NEIGHBORHOOD']==neighborhood_name)]
    house=st.selectbox('Select the house types you want to check price forecast for:', df_building['BUILDING CLASS CATEGORY'].unique())

    if st.button('Predict'):
        file_name="models/"+borough_name+"_"+neighborhood_name+"_"+house+".json"
        with open(file_name, 'r') as fin:
            model_ext = model_from_json(fin.read())  # Load model

        #model_cell=models[['model']][(models['Borough']==borough_name)&(models['Neighborhood']==neighborhood_name)&(models['Housing Category']==house)]
        #model_ext=model_cell.iloc[0]['model']
        future = model_ext.make_future_dataframe(periods=1900)
        future['cap'] = 2000000
        future['floor'] = 80000
        forecast=model_ext.predict(future)
        forecast=forecast.dropna()
        forecast['Year']=pd.DatetimeIndex(forecast['ds']).year
        forecast['Month']=pd.DatetimeIndex(forecast['ds']).month
        #st.write(df.tail())
        df['Year']=pd.DatetimeIndex(df['SALE DATE']).year
        df['Month']=pd.DatetimeIndex(df['SALE DATE']).month
        dec_2022_price_df=df[['AVERAGE PRICE']][(df['Year']==2022)&(df['BOROUGH']==borough_name)&(df['NEIGHBORHOOD']==neighborhood_name)&(df['BUILDING CLASS CATEGORY']==house)]
        dec_2024_price_df=forecast[['yhat']][(forecast['Year']==2023)]
        dec_2027_price_df=forecast[['yhat']][(forecast['Year']==2024)]
        #st.write(dec_2032_price_df.tail())
        #st.write(dec_2022_price_df)
        #st.write(forecast.tail())
        average_dec_2022_price=int(dec_2022_price_df['AVERAGE PRICE'].mean())
        average_dec_2024_price=int(dec_2024_price_df.mean())
        average_dec_2027_price=int(dec_2027_price_df.mean())
        returns_2024=round(((average_dec_2024_price-average_dec_2022_price)/average_dec_2022_price)*100,2)
        returns_2027=round(((average_dec_2027_price-average_dec_2022_price)/average_dec_2022_price)*100,2)
        #st.write(forecast.head())
        st.write("A "+house+" property purchased in December 2022 for $"+str(average_dec_2022_price)+" would be worth:")
        st.write("$"+str(average_dec_2024_price)+" in December 2024, yielding "+str(returns_2024)+"% in returns,")
        st.write("and")
        st.write("$"+str(average_dec_2027_price)+" in December 2027, yielding "+str(returns_2027)+"% in returns.")
        subset_df=df[['SALE DATE']][(df['BOROUGH']==borough_name)&(df['NEIGHBORHOOD']==neighborhood_name)&(df['BUILDING CLASS CATEGORY']==house)]
        dec_2022_index=subset_df.shape[0]
        
        #forecast_actual=forecast['ds','yhat'].iloc[:dec_2022_index]
        #forecast_predicted=forecast['ds','yhat'].iloc[dec_2022_index:]
        fig = model_ext.plot_components(forecast)
        fig1=plot_plotly(model_ext, forecast)
        st.write(fig1)
        st.write(fig)
        #fig = ff.create_distplot(forecast_actual)
        #fig, ax= plt.subplots()


        #ax=forecast.iloc[:dec_2022_index,:].plot(ls="-", color="b")
        #forecast.iloc[dec_2022_index:,:].plot(ls="--", color="r", ax=ax)
        #fig, ay = plt.subplots()
        #st.write(plt.show())
    
