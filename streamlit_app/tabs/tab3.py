import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn import model_selection, preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import scipy.stats as stats

title = "Data Exploration"
sidebar_name = "Data Exploration"


def tab3_content():
    selected_graph = st.selectbox("Select", ["Fuel mode", "Fuel type"])
    if selected_graph == "Fuel mode":
        st.subheader("Vehicles in our dataset are distinguished in 6 classes based on their fuel modes.")
        st.write(f"**M, B, F are fossil fuels only motors:**")
        st.write(f"- M (Mono): One fossil fuel")
        st.write(f"- B (Bi): Two fossil fuels")
        st.write(f"- F (Flex): Two or more fossil fuels are kept in the same tank.")
        st.write(f"**H, P use electricity:**")
        st.write(f"- H Fuel_Type: Non-Plug-In Hybrid. The only fuel is a fossil one, although there is a battery and an electric motor")
        st.write(f"- P Fuel Type is Plug-In-Hybrid. Charged by plug-in only. But there is a fossil fuel motor as well, as a backup. ")
        st.write(f"**E is electric motor only**")
        st.divider()
        st.write(f"**Modes E and P are not comparable with the fuel-powered cars. For that reason, we will perform the analysis with vehicles only of M, B, F and H class.**")

    elif selected_graph == "Fuel type": 
        st.write("Cars fueled by different sources exhibited varying levels of CO2 emissions. Notably, vehicles powered by **e85**, a blend of ethanol and gasoline, demonstrated the least environmentally friendly profile in terms of CO2 emissions. **Diesel**, **petrol**, **liquefied petroleum gas (LPG)**, **natural gas (NG)**, and **NG-biomethane** vehicles exhibited comparable CO2 emissions. Both types of **hybrid** cars (diesel and petrol hybrids) showed significantly lower CO2 emissions. ")
        st.write("**Electric cars** were found to be the most environmentally friendly, with virtually undetectable CO2 emissions. However, excluding them from subsequent analyses is necessary to prevent skewing in the dataset.")
        st.image("CO2 emmisions per fuel type.png")
        st.caption("CO2 emissions per fuel type")




