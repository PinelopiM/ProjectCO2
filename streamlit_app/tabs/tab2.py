import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


title = "Data Visualization"
sidebar_name = "Data Visualization"


def tab2_content():

    selected_viz = st.selectbox("Select Visualization", ["Correlation Analysis", "CO2 Emissions by Country", "CO2 Emissions by Car Manufacturer", 'Pairplot', 'Countries by Fuel Types'])
    if selected_viz == "Correlation Analysis":
        st.write("CO2 emissions correlate primarily with the engine capacity and power of the vehicle, as well as with its mass. Secondarily, the dimensions of the car seem to affect its levels of CO2 emissions, as shown by the correlation of  the axle width and the wheel base variables with our target variable. Weaker correlation of CO2 emission was detected with the capacity of the automobile for the off-road function. Additionally, the values of CO2 emissions seem to be negatively correlated with the the hybrid fuel source of petrol and electricity.")
        st.image("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/correlation heatmap, wo electrics.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.caption("Correlation heatmap")

    elif selected_viz == "CO2 Emissions by Country":
        st.image('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/emissionsbycountry.png', use_column_width=True)
        
    elif selected_viz == "CO2 Emissions by Car Manufacturer":
        st.write("All manufacturers under examination exhibit fleet-wide CO2 emissions exceeding the European Union target of 95g CO2/km for passenger cars during the period 2020-2024. Furthermore, our analysis revealed statistically significant variations in CO2 emissions among cars from different manufacturers. Notably, TOYOTA cars have the lowest amount of CO2 emissions, while vehicles manufactured by DAIMLER AG the highest range.")
        st.image("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/CO2 emissions of fossil fuel cars per manufacturer.png",caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.caption("CO2 emissions of fossil fuel cars per manufacturer. Only the 10 most present manufacturers are displayed here.")
    
    #elif selected_viz == "CO2 emissions by Fuel Type":
        #st.image('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/emissionsbyFT.png', use_column_width=True)

    elif selected_viz == 'Pairplot':
        st.image('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/paiplot.png')
    elif selected_viz == 'Countries by Fuel Types':
        st.image('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/Country_FT.png')

        

      