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
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import scipy.stats as stats
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import pickle

def tab6_content():
    
    df = pd.read_csv("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/CO2_2019_Sanjana.csv")
    #Drop some outliers
    df = df.drop(df[df["Axle_Width"] < 1250].index)
    df = df.drop(df[df["Axle_Width"] > 2000].index)

   #Include dummies for the top 10 manufacturers
    top_10 = df['Manufacturer'].value_counts().head(10).index.tolist()
    df_10 = df[df['Manufacturer'].isin(top_10)]
    dummies_10 = pd.get_dummies(df_10['Manufacturer'])
    df_10 = df_10.join(dummies_10)
    df_10.drop('Manufacturer', axis = 1, inplace = True)

    # dataFrame for the regression model
    df_10= df_10.replace({False: 0, True: 1})
    target = df_10.CO2
    data =df_10.drop(['Country', 'ID', "CO2",], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=87)

    # dataFrame for the classification model
    df_class = df_10.copy()
    df_class["CO2_Classes"]=pd.cut(x = df_10['CO2'], bins = [0, 90, 110, 130, 150, 200, 250], labels= ['A', 'B', 'C', 'D', "E", "F"])
    df_class.dropna(inplace = True)
    target_clf = df_class.CO2_Classes 
    data_clf =df_class.drop(["CO2_Classes", 'CO2', 'Country', 'ID'], axis=1)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(data_clf, target_clf, test_size=0.3, random_state=66)

    rf_10 = RandomForestRegressor(n_estimators=3, random_state=62)
    rfc_10 = RandomForestClassifier(n_estimators=50, random_state=72)

    
    # Train the regressor
    rf_10.fit(X_train, y_train)
    joblib.dump(rf_10, 'randomforestregressor.pkl')

    #Train the classifier
    rfc_10.fit(X_train_clf, y_train_clf)
    joblib.dump(rfc_10, 'randomforestclassifier.pkl')

    st.title('Predicting C02 emissions : Demonstration')

    st.write('### Select values of the variables to see the predicted CO2 emissions')

# Create sliders for numeric variables
    Mass = st.slider("Mass", min_value= 1000, max_value=2500, value= 1500)
    Axle_Width = st.slider("Axle Width", min_value=1300, max_value=1900, value= 1500)
    Wheel_Base = st.slider("Wheel Base", min_value=1500, max_value=4500, value= 2500)
    Eng_Capacity = st.slider("Engine Capacity", min_value= 875, max_value= 3000, value= 1500)
    Eng_Power = st.slider("Engine Power", min_value=43, max_value=400, value=100)

# Create dropdowns for categorical variables
    Manufacturer = st.selectbox("Select Manufacturer", ['OPEL AUTOMOBILE', 'TOYOTA', 'RENAULT', 'VOLKSWAGEN', 'DAIMLER AG', 'BMW AG', 'FIAT GROUP', 'SKODA', 'FORD WERKE GMBH', 'AUTOMOBILES PEUGEOT'])

# Add checkboxes for other binary variables
    Fuel_Type = st.radio("# Select fuel type:", ["diesel", "petrol", "natural gas", 'e85', 'natural gas- biomethane', 'lpg'])
    Fuel_Mode = st.radio('# Select fuel mode:', ['M', 'H', 'B', 'F'])

# Create checkboxes for binary variables
    ERT = st.checkbox("Does the car have any emissions reduction technology (ERT)?")
    Vehicle_cat_labels = st.checkbox("Is it a four wheel drive?")

    selected_variables = ['Mass', 'Axle_Width', 'Wheel_Base', 'Eng_Capacity', 'Eng_Power',"diesel", "petrol", "ng", 'e85', 'ng-biomethane', 'lpg',
    'M', 'H', 'B', 'F', 'ERT', 'Vehicle_cal_labels','OPEL AUTOMOBILE', 'TOYOTA', 'RENAULT', 'VOLKSWAGEN', 'DAIMLER AG', 'BMW AG', 'FIAT GROUP',
    'SKODA', 'FORD WERKE GMBH', 'AUTOMOBILES PEUGEOT']


    model = joblib.load('randomforestregressor.pkl')
    # Make predictions on the testing data
    clf = joblib.load('randomforestclassifier.pkl')

    user_input = pd.DataFrame({
        'Mass': [Mass], 'Wheel_Base': [Wheel_Base],
        'Axle_Width': [Axle_Width], 'Eng_Capacity': [Eng_Capacity],
        'Eng_Power': [Eng_Power], 'Vehicle_cat_labels': [1 if Vehicle_cat_labels else 0],
        'B': [1 if Fuel_Mode == 'B' else 0],
        'F': [1 if Fuel_Mode == 'F' else 0],
        'H': [1 if Fuel_Mode == 'H' else 0],
        'M': [1 if Fuel_Mode == 'M' else 0],
        'diesel': [1 if Fuel_Type == 'diesel' else 0], 'e85': [1 if Fuel_Type == 'e85' else 0],
        'lpg': [1 if Fuel_Type == 'lpg' else 0], 'ng': [1 if Fuel_Type == 'natural gas' else 0],
        'ng-biomethane': [1 if Fuel_Type == 'natural gas- biomethane' else 0], 'petrol': [1 if Fuel_Type == 'petrol' else 0],
        'AUTOMOBILES PEUGEOT': [1 if Manufacturer == 'AUTOMOBILES PEUGEOT' else 0],
        'BMW AG': [1 if Manufacturer == 'BMW AG' else 0],
        'DAIMLER AG': [1 if Manufacturer == 'DAIMLER AG' else 0],
        'FIAT GROUP': [1 if Manufacturer == 'FIAT GROUP' else 0],
        'FORD WERKE GMBH' : [1 if Manufacturer == 'FORD WERKE GMBH' else 0],
        'OPEL AUTOMOBILE': [1 if Manufacturer == 'OPEL AUTOMOBILE' else 0],
        'RENAULT': [1 if Manufacturer == 'RENAULT' else 0],
        'SKODA': [1 if Manufacturer == 'SKODA' else 0], 
        'TOYOTA': [1 if Manufacturer == 'TOYOTA' else 0], 
        'VOLKSWAGEN': [1 if Manufacturer == 'VOLKSWAGEN' else 0]})

    if st.button("Predict emissions using regressor"):
    #Make prediction
        prediction = model.predict(user_input)
        y_pred = ', '.join([f'{value:.2f}' for value in prediction])
        st.write('CO2 emissions of the car are:', y_pred, 'g/km')
        st.image("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/classes.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.caption("Reference C02 emissions and classes")
         

    elif st.button('Predict emissions class using classifier'): 
     #Make prediction for classification
        prediction = clf.predict(user_input)
        y_pred = ', '.join(map(str, prediction))
        st.write('The emissions class of this car is:', y_pred)
        st.image("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/classes.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.caption("Reference C02 emissions and classes")
        st.write('Class A (up to 90 g CO2/km), Class B (90-110 g/km), Class C (110-130 g/km), Class D (130-150 g/km), Class E (150-200 g/km), Class F (200-250 g/km)')
    


