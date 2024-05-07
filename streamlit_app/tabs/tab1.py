import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


title = "Predicting CO2 emissions of EU cars in 2019"
sidebar_name = "Table of contents"


def tab1_content():
  st.title("Predicting CO2 emissions of EU cars in 2019")
  df=pd.read_csv("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/CO2_2019_Sanjana.csv") #this is the final file after all the preprocessing
  df19 =pd.read_csv("/Users/sanjanasingh/Documents/2023/DST_course/Project/CO2_2019.csv") #this is the final file after MOST of the preprocessing that was used for visualization (eg, electric cars are still in)
  df_raw = pd.read_csv('/Users/sanjanasingh/Documents/2023/DST_course/Project/df_random.csv')

  st.write('A snapshot of the dataset of C02 emissions by cars in the EU in 2019')
  st.dataframe(df_raw.head(10))
  st.write('The dataset taken for analysis and modelling is 0.1\% of the original dataset')
  st.write('This dataset has the following rows and columns respectively:')
  st.write(df_raw.shape)
  st.write('Summary statistics of the dataset are :')
  st.dataframe(df_raw.describe())

  if st.checkbox("Show NA"):
    st.dataframe(df.isna().sum())


  st.write('Variables that are not required or have too many missing values are deleted with the result:')
  df_dropped = df_raw.drop(['year','Unnamed: 0','Date of registration','Electric range (km)','Erwltp (g/km)','Ernedc (g/km)','Ewltp (g/km)','z (Wh/km)','IT','Status','r','Vf','De','At2 (mm)','Mt','Ct','Ve', 'Tan','MMS','Man','VFN','Mp','T', 'Va','Mk','Cn'], axis=1)
  df_dropped = df_raw.iloc[:,:-1]
  st.dataframe(df_dropped.head())

  #duplicated entries:
  if st.checkbox('Show duplicates'):
    st.dataframe(df_raw.duplicated().sum())
  
  #changing the index to ID and dropping ID column
  df.index = df['ID']
  df = df.drop('ID', axis=1)

  #renaming columns to be more readable: 

  df_renamed = df_raw.rename({"Mh": "Manufacturer", "m (kg)": "Mass", "Cr": "Vehicle_Category", "W (mm)": "Wheel_Base", "At1 (mm)": "Axle_Width", "Ft": "Fuel_Type", "Fm": "Fuel_Mode", "ec (cm3)": "Eng_Capacity", "ep (KW)": "Eng_Power", "Enedc (g/km)": "CO2"}, axis=1)
  st.write('The columns are renamed to be more understandable:')
  st.dataframe(df_renamed.head())

  

