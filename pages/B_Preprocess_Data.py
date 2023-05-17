import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import streamlit as st  # pip install streamlit
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from pages.A_Explore_Dataset import user_input_features

#############################################
st.markdown("## AI-Powered Housing Price Prediction Platform")

st.markdown('# Preprocess Dataset')


#############################################

def restore_dataset():  # ADDED THIS AND COMMENTED OUT ALL OF ABOVE at 1:49am
    df = None
    if 'house_df' not in st.session_state:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df


# Checkpoint 2
def summarize_missing_data(df, top_n=3):
    missing_values = df.isna().sum()
    num_categories = (missing_values > 0).sum()
    average_per_category = missing_values.mean()
    total_missing_values = missing_values.sum()
    top_missing_categories = missing_values.nlargest(top_n).index.tolist()

    out_dict = {
        'num_categories': num_categories,
        'average_per_category': average_per_category,
        'total_missing_values': total_missing_values,
        'top_missing_categories': top_missing_categories
    }
    return out_dict


# Remove features function
def remove_features(X, removed_features):
    X = X.drop(removed_features, axis=1)
    return X


# Checkpoint 3
def remove_nans(df):
    df = df.dropna()
    return df


# Checkpoint 4
def remove_outliers(df, feature):
    df = df.dropna()
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    return dataset, lower_bound, upper_bound


# Checkpoint 5
def one_hot_encode_feature(df, feature):
    df_copy = df.copy()
    encoded_df = pd.get_dummies(df_copy, columns=[feature], prefix=[feature])

    st.write('Feature {} has been one-hot encoded.'.format(feature))
    return encoded_df


def integer_encode_feature(df, feature):
    df_copy = df.copy()
    encoder = OrdinalEncoder()
    df_copy[feature] = encoder.fit_transform(df_copy[[feature]])

    st.write('Feature {} has been integer encoded.'.format(feature))
    return df_copy


# Checkpoint 6
def create_feature(df, math_select, math_feature_select, new_feature_name):
    df_copy = df.copy()
    if math_select == 'square root':
        df_copy[new_feature_name] = np.sqrt(df_copy[math_feature_select[0]])
    elif math_select == 'ceil':
        df_copy[new_feature_name] = np.ceil(df_copy[math_feature_select[0]])
    elif math_select == 'floor':
        df_copy[new_feature_name] = np.floor(df_copy[math_feature_select[0]])
    elif math_select == 'add':
        df_copy[new_feature_name] = df_copy[math_feature_select[0]] + df_copy[math_feature_select[1]]
    elif math_select == 'subtract':
        df_copy[new_feature_name] = df_copy[math_feature_select[0]] - df_copy[math_feature_select[1]]
    elif math_select == 'multiply':
        df_copy[new_feature_name] = df_copy[math_feature_select[0]] * df_copy[math_feature_select[1]]
    elif math_select == 'divide':
        df_copy[new_feature_name] = df_copy[math_feature_select[0]] / df_copy[math_feature_select[1]]

    return df_copy


def filter_numeric_columns(df):
    return df.select_dtypes(include=[np.number])


def impute_dataset(X, impute_method):
    if impute_method == 'Zero':
        numeric_X = filter_numeric_columns(X)
        for col in numeric_X.columns:
            X[col] = X[col].astype(float)
            X[col].fillna(0, inplace=True)
    elif impute_method == 'Mean':
        numeric_X = filter_numeric_columns(X)
        mean_values = numeric_X.mean()
        for col in mean_values.index:
            X[col].fillna(mean_values[col], inplace=True)
    elif impute_method == 'Median':
        numeric_X = filter_numeric_columns(X)
        median_values = numeric_X.median()
        for col in median_values.index:
            X[col].fillna(median_values[col], inplace=True)

    return X


# Complete this helper function from HW1
def remove_features(X, removed_features):
    return X.drop(removed_features, axis=1)


# Complete this helper function from HW1
def compute_descriptive_stats(X, stats_feature_select, stats_select):
    output_str = ''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    for f in stats_feature_select:
        output_str = str(f)
        for s in stats_select:
            if (s == 'Mean'):
                mean = round(X[f].mean(), 2)
                output_str = output_str + ' mean: {0:.2f}    |'.format(mean)
                out_dict['mean'] = mean
            elif (s == 'Median'):
                median = round(X[f].median(), 2)
                output_str = output_str + ' median: {0:.2f}    |'.format(median)
                out_dict['median'] = median
            elif (s == 'Max'):
                max = round(X[f].max(), 2)
                output_str = output_str + ' max: {0:.2f}    |'.format(max)
                out_dict['max'] = max
            elif (s == 'Min'):
                min = round(X[f].min(), 2)
                output_str = output_str + ' min: {0:.2f}    |'.format(min)
                out_dict['min'] = min
        st.write(output_str)
    return output_str, out_dict


# Helper function
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


###################### FETCH DATASET #######################
df = restore_dataset()

if df is not None:
    X = df
    Y = df.loc[:, df.columns.isin(['AVERAGE PRICE'])]
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.dataframe(df)

    # Show summary of missing values
    missing_data_summary = summarize_missing_data(df)
    st.markdown('### Summary of Missing Data')
    st.write(f"Number of Categories with Missing Values: {missing_data_summary['num_categories']}")
    st.write(f"Average Missing Values per Category: {missing_data_summary['average_per_category']}")
    st.write(f"Total Missing Values: {missing_data_summary['total_missing_values']}")
    st.write(f"Top Missing Categories: {', '.join(missing_data_summary['top_missing_categories'])}")

    ############################################# MAIN BODY #############################################

    # Remove feature
    st.markdown('### Remove irrelevant/useless features')
    removed_features = st.multiselect(
        'Select features',
        X.columns,
    )

    X = remove_features(X, removed_features)

    # Display updated dataframe
    st.dataframe(X)

    numeric_columns = list(X.select_dtypes(include='number').columns)

     # Create New Features
    st.markdown('## Create New Features')
    st.markdown(
        'Create new features by selecting two features below and selecting a mathematical operator to combine them.')
    math_select = st.selectbox(
        'Select a mathematical operation',
        ['add', 'subtract', 'multiply', 'divide', 'square root', 'ceil', 'floor'],
    )

    if (math_select):
        if (math_select == 'square root' or math_select == 'ceil' or math_select == 'floor'):
            math_feature_select = st.multiselect(
                'Select features for feature creation',
                numeric_columns,
            )
            sqrt = np.sqrt(df[math_feature_select])
            if (math_feature_select):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    if (new_feature_name):
                        X = create_feature(
                            X, math_select, math_feature_select, new_feature_name)
                        st.write(X)
        else:
            math_feature_select1 = st.selectbox(
                'Select feature 1 for feature creation',
                numeric_columns,
            )
            math_feature_select2 = st.selectbox(
                'Select feature 2 for feature creation',
                numeric_columns,
            )
            if (math_feature_select1 and math_feature_select2):
                new_feature_name = st.text_input('Enter new feature name')
                if (st.button('Create new feature')):
                    X = create_feature(X, math_select, [
                        math_feature_select1, math_feature_select2], new_feature_name)
                    st.write(X)
    # Handle NaNs
    remove_nan_col, impute_col = st.columns(2)

    with (remove_nan_col):
        # Remove Nans
        st.markdown('### Remove Nans')
        if st.button('Remove Nans'):
            X = remove_nans(X)
            st.write('Nans Removed')
        else:
            st.write('Dataset might contain Nans')

    with (impute_col):
        # Clean dataset
        st.markdown('### Impute data')
        st.markdown('Transform missing values to 0, mean, or median')

        # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
        impute_method = st.selectbox(
            'Select cleaning method',
            ('None', 'Zero', 'Mean', 'Median')
        )

        X = impute_dataset(X, impute_method)

    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')
    st.dataframe(X)

    # Remove outliers
    st.markdown("### Inspect Features and Remove Outliers")
    feature_to_inspect = st.selectbox("Select a feature", numeric_columns)
    chart_type = st.selectbox("Select a chart type", ["Scatterplot", "Lineplot", "Histogram", "Boxplot"])

    if chart_type == "Scatterplot":
        fig = px.scatter(df, x=feature_to_inspect, y="SALE PRICE")
    elif chart_type == "Lineplot":
        fig = px.line(df, x=feature_to_inspect, y="SALE PRICE")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=feature_to_inspect)
    elif chart_type == "Boxplot":
        fig = px.box(df, x=feature_to_inspect)

    st.plotly_chart(fig)

    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label='Type of chart',
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot'],
        key='sidebar_chart'
    )

    # Add code here

    st.markdown('### Inspect Features for outliers')
    outlier_feature_select = None
    outlier_feature_select = st.selectbox(
        'Select a feature for outlier removal',
        numeric_columns,
    )
    if (outlier_feature_select and st.button('Remove Outliers')):
        X, lower_bound, upper_bound = remove_outliers(
            X, outlier_feature_select)
        st.write('Outliers for feature {} are lower than {} and higher than {}'.format(
            outlier_feature_select, lower_bound, upper_bound))
        st.write(X)

    # Handling Text and Categorical Attributes
    st.markdown('### Handling Text and Categorical Attributes')
    string_columns = list(X.select_dtypes(['object']).columns)

    int_col, one_hot_col = st.columns(2)

    # Perform Integer Encoding
    with (int_col):
        text_feature_select_int = st.selectbox(
            'Select text features for Integer encoding',
            string_columns,
        )
        if (text_feature_select_int and st.button('Integer Encode feature')):
            if 'integer_encode' not in st.session_state:
                st.session_state['integer_encode'] = {}
            if text_feature_select_int not in st.session_state['integer_encode']:
                st.session_state['integer_encode'][text_feature_select_int] = True
            else:
                st.session_state['integer_encode'][text_feature_select_int] = True
            X = integer_encode_feature(X, text_feature_select_int)

    # Perform One-hot Encoding
    with (one_hot_col):
        text_feature_select_onehot = st.selectbox(
            'Select text features for One-hot encoding',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('One-hot Encode feature')):
            if 'one_hot_encode' not in st.session_state:
                st.session_state['one_hot_encode'] = {}
            if text_feature_select_onehot not in st.session_state['one_hot_encode']:
                st.session_state['one_hot_encode'][text_feature_select_onehot] = True
            else:
                st.session_state['one_hot_encode'][text_feature_select_onehot] = True
            X = one_hot_encode_feature(X, text_feature_select_onehot)

    # Show updated dataset
    st.write(X)



    # Descriptive Statistics
    st.markdown('### Summary of Descriptive Statistics')

    stats_numeric_columns = list(X.select_dtypes(['float', 'int']).columns)
    stats_feature_select = st.multiselect(
        'Select features for statistics',
        stats_numeric_columns,
    )
    # Select statistic to compute
    if (stats_feature_select):
        stats_select = st.multiselect(
            'Select statistics to display',
            ['Mean', 'Median', 'Max', 'Min']
        )

        # Compute Descriptive Statistics including mean, median, min, max
        display_stats, _ = compute_descriptive_stats(X, stats_feature_select, stats_select)

    st.session_state['house_df'] = X

    ###################### DOWNLOAD DATASET #######################
    st.markdown('### Download the dataset')

    csv = convert_df(X)

    st.write('Saving dataset to paml_dataset.csv')
    st.download_button(
        label="Preprocess: Download data as CSV",
        data=csv,
        file_name='paml_dataset.csv',
        mime='text/csv',
    )

    st.write('Continue to Train Model')
