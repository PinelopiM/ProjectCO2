import streamlit as st

def conclusion():
    st.subheader("Conclusion")
    st.write('In this project, we worked with the EU cars emissions dataset from 2019, aiming to create a model\
             to predict the C02 emissions of a car given a number of characteristics of the car. The datasest\
            contained a number of variables,such as physical characteristics of a car, specifics of engines,\
             different fuels etc. We first simplified the dataset by deleting all variables that were missing a lot\
             of values or did not matter for C02 emissions.')
    
    st.write('We then plotted different variables against our target (C02 emissions) to visualize the relationships,\
             and it became clear that several of the variables heavily influenced C02 emissions.')
    
    st.write('In attempt to best model the CO2 emissions of cars, we took two approaches: classification and regression.\
            For classification, the continuous variable C02 emissions was split into classes and many models were trained\
            on this dataset. The best model for classification turned out to be a Random Forest Classifier with 50 estimators,\
            giving a very precise estimate of the emissions class. In the regression approach, the dataset was similarly trained\
            on a number of models and hyperparameters to select the best model. Here the best model was a Random Forest Regressor\
            with 3 estimators, which also gave an extremely good prediction with a very low error rate.')
    
    st.write('Thanks to these models, we can not only accurately predict CO2 emissions of cars, but we can say that the physical\
            characteristics of the car such as Mass play a critical role in predicting C02 emissions, and surprisingly, the\
            manufacturer of the car also has a big effect on the emissions of a car')
