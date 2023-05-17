import streamlit as st  # pip install streamlit

st.markdown("### AI-Powered Housing Price Prediction Platform")

#############################################
#############################################

st.markdown("## Team Members:")
st.markdown("Vrinda Lohia, Steven Abd El Hamid, Simarpreet Arora, Saransh Srivastava, Rahul Rai")

st.markdown("## Executive Summary")
st.markdown(
    "Our AI-powered housing price prediction platform aims to provide accurate and reliable price forecasts for properties in the New York City metropolitan area. The New York City metropolitan housing market is complex and dynamic, making it difficult for individuals and businesses to accurately predict property values. Inaccurate pricing can lead to missed opportunities or financial losses for homebuyers, investors, and real estate professionals alike. By leveraging historical housing data and advanced machine learning algorithms, the platform will help homebuyers, real estate investors, and professionals make informed decisions based on data-driven insights.")

st.markdown("## Solution")
st.markdown(
    "Our platform will use historical housing data, including property features, neighborhood information, and market trends, to create accurate predictions for housing prices. By employing machine learning algorithms, the platform will continuously learn from new data, refining its predictions over time. Users will have access to valuable insights, allowing them to make well-informed decisions in the housing market.")

st.markdown("## Business Model")
st.markdown("Our platform will generate revenue through the following product mix: \n"
            "- **Subscription Plans**: Monthly or annual subscriptions meant for institutional, commercial, and individual buyers for different tiers of access with premium features such as advanced analytics and personalized recommendations. There will be no free tier.\n"
            "- **Marketplace listing for house owners**: Empowering homeowners to review historical prices of properties sold in their neighborhood or comparable areas, as well as predict future prices, allowing them to set an informed asking price for their own home. Generating revenue primarily through commissions earned upon the successful sale of the property.\n"
            "- **API access**: Providing API access to businesses, allowing them to integrate our predictions into their systems and applications as long as they don’t use the API to sell a competing subscription plan.\n"
            "- **Custom reports and analysis**: Offering custom reports and in-depth analysis for users seeking specific insights, more visualizations, or further predictions for particular properties or neighborhoods.\n")

st.markdown("## Plan Mock-up")
st.markdown(
    "Automated tool with a simple interface that asks users to input the characteristics of the property they want to predict the price for - location (could be a neighborhood/ street/ particular building), size of the property (specified in sq ft), type of property (condominium/ house/ etc.), forecast period (time at which future price needed). The tool will use these inputs and at the back-end run AI/ ML powered algorithms to obtain a particular/ range of price given the user inputs. In the matter of a few seconds, the user will be shown the forecasted price for the inputs provided.")

st.markdown("## Datasets")
st.markdown("Publicly available datasets would be used for this project. Some of the sources are as follows: \n"
            "-  Kaggle [1], [2] \n"
            "-  NYC Open Data \n"
            "-  StreetEasy \n"
            "-  Data.gov \n"
            "Additionally, synthetic datasets can be generated in case the public datasets are small in size")

st.markdown("Click **Explore Dataset** to get started.")
