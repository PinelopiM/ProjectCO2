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

def tab4_content():
      #Modeling
    from sklearn import ensemble
    from sklearn.ensemble import RandomForestRegressor
    import scipy.stats as stats
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import Lasso
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import joblib

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
    df_10 = df_10.replace({False: 0, True: 1})

    target = df_10.CO2
    data =df_10.drop(['Country', "CO2",], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=87)

  #Models
  #Define the models
    def prediction(regressor_type):
        if regressor_type == 'Linear Regression':
            lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/LR.pkl')
        elif regressor_type == 'Ridge Regression':
            alphas = st.selectbox('Alpha', [0.1,1,10,50,100])
            if alphas == 0.1:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RidgeCV_01.pkl')
            elif alphas == 1:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RidgeCV_1.pkl')
            elif alphas == 10:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RidgeCV_10.pkl')
            elif alphas == 50:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RidgeCV_50.pkl')
            elif alphas == 100:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RidgeCV_100.pkl')

        elif regressor_type == 'Lasso Regression':
            alphas_L = st.selectbox('Alpha', [0.1,1,10,50,100])
            if alphas_L == 0.1:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/Lasso01.pkl')
            elif alphas_L == 1:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/Lasso1.pkl')
            elif alphas_L == 10:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/Lasso10.pkl')
            elif alphas_L == 50:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/Lasso50.pkl')
            elif alphas_L == 100:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/Lasso100.pkl')


        elif regressor_type == 'Random Forest Regressor':
            n_estimators = st.selectbox('Number of estimators', [1,3,5,10,50])
            if n_estimators == 1:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RFR_1.pkl')
            elif n_estimators == 3:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RFR_3.pkl')
            elif n_estimators == 5:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RFR_5.pkl')
            elif n_estimators == 10:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RFR_10.pkl')
            elif n_estimators == 50:
                lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/RFR_50.pkl')
    
        elif regressor_type == 'XG Boost':
            num_estimators = st.selectbox('Number of Estimators', [100,300])
            learning_rate = st.selectbox('Learning Rate', [0.1,0.2])
            max_depth = st.selectbox('Max Depth', [3,5])
            if num_estimators == 100:
                if learning_rate == 0.1:
                    if max_depth == 3:
                        lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb1000103.pkl')
                    elif max_depth == 5:
                        lr =joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb1000105.pkl')
                elif learning_rate == 0.2:
                    if max_depth == 3:
                        lr = joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb1000203.pkl')
                    elif max_depth == 5:
                        lr =joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb1000205.pkl')
            elif num_estimators == 300:
                if learning_rate == 0.1:
                    if max_depth == 3:
                        lr =joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb3000103.pkl')
                    elif max_depth == 5:
                        lr =joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb3000105.pkl')
                elif learning_rate == 0.2:
                    if max_depth == 3:
                        lr=joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb3000203.pkl')
                    elif max_depth == 5:
                        lr=joblib.load('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/Pre-trained models/xgb3000205.pkl')
        return lr   


  # Set condition for model
    regressor_type = st.selectbox('Choice of the model', ['Linear Regression', 'Ridge Regression', 'Lasso Regression',"Random Forest Regressor", "XG Boost"])
    st.write('The chosen model is:', regressor_type)

 # Prediction
    lr = prediction(regressor_type)
    y_pred = lr.predict(X_test)

    def evaluate_regression(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        return r2, mse, mae
# Streamlit code for selecting regression model and displaying evaluation metrics


# Calculate evaluation metrics
    r2_score, mse, mae = evaluate_regression(y_test, y_pred)

# Display evaluation results
    st.write(f"### {regressor_type} Model Performance")
    st.write(f"R-squared: {r2_score:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Mean Absolute Error: {mae:.4f}")

# Dropdown menu for coefficients/feature importances
    display_option = st.selectbox('Select display option', ['None', 'Coefficients', 'Feature Importances'])

# Display coefficients or feature importances based on user selection
    if display_option == 'Coefficients':
        if regressor_type == 'Linear Regression' or regressor_type == 'Ridge Regression' or regressor_type == 'Lasso Regression':
            coeff = lr.coef_
            feature_names = data.columns
            coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coeff}).sort_values('Coefficient')
            st.write("### Coefficients")
            st.dataframe(coefficients_df)
    elif display_option == 'Feature Importances':
        if regressor_type == 'Random Forest Regressor' or regressor_type == 'XG Boost':
            if regressor_type == 'Random Forest Regressor':
                feature_importances = lr.feature_importances_
            elif regressor_type == 'XG Boost':
                feature_importances = lr.feature_importances_
            feature_names = data.columns
            importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values('Importance', ascending=False)
            st.write("### Feature Importances")
            st.dataframe(importances_df)


  