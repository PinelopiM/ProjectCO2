
def tab5_content():
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

    df = pd.read_csv('/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/CO2_2019_preprocessed.csv')

    st.write("### Car Classification based on CO2 emissions")

   #Drop some outliers
    df = df.drop(df[df["Axle_Width"] < 1250].index)
    df = df.drop(df[df["Axle_Width"] > 2000].index)

   #Include dummies for the top 10 manufacturers
    top_10 = df['Manufacturer'].value_counts().head(10).index.tolist()
    df_10 = df[df['Manufacturer'].isin(top_10)]
    dummies_10 = pd.get_dummies(df_10['Manufacturer'])
    df_10 = df_10.join(dummies_10)
    df_10.drop('Manufacturer', axis = 1, inplace = True)
    df_10= df_10.replace({False: 0, True: 1})

  #Create the classes
    df_10["CO2_Classes"]=pd.cut(x = df_10['CO2_Eng_given'], bins = [0, 90, 110, 130, 150, 200, 250], labels= ['A', 'B', 'C', 'D', "E", "F"])

  #Figure to display class distribution
    st.write("Analysis of the distribution showed that the majority of the examined cars emitted CO2 within the range of 100-150 g/km. Based on that, six CO2 classes (A-F) were instantiated, with each class representing a distinct range of emissions proportional to the overall data distribution. Specifically, the breakdown of cars per category was as follows: Class A (up to 90 g CO2/km), Class B (90-110 g/km), Class C (110-130 g/km), Class D (130-150 g/km), Class E (150-200 g/km), Class F (200-250 g/km).")
    st.image("/Users/sanjanasingh/Documents/GitHub/bds_int_co2/streamlit_app/visualizations/classes.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption("Vehicle classes based on CO2 emissions.")

  #Finalize the datasets
    df_class=df_10.drop(["CO2","CO2_Eng_given","CO2_Eng_misses","Country","Vehicle_Category","Fuel_Type","Fuel_Mode", "Eng_Power_missing"], axis=1)
    df_class=df_class.dropna()

  ## Split data
    target = df_class.CO2_Classes 
    data =df_class.drop( "CO2_Classes", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=66) #Split into train and test
    X_test = X_test.to_numpy() #I got an error and that's why I added this

  #Modeling
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn import neighbors
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn import ensemble
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier, BaggingClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    import streamlit as st
    import joblib

  #Define the prediction function that will allow the loading of the pretrained model according to the parameter
    def prediction(classifier_type):
      # Model selection
      if classifier_type == 'KNN': #model selection (the selectbox for that will come later)
          n_neighbors = st.selectbox('Select n_neighbors', [3, 5, 7]) #create a selectbox with the appropriate parameter
          if n_neighbors == 3:  #for each parameter selection:
              clf = joblib.load('KNN_model_n_neighbors_3.joblib') #load the pre-trained model
              st.write(f"Selected model is K-nearest neighbours with n_neighbors=3") #display model description
          elif n_neighbors == 5: #repeat as before
              clf = joblib.load('KNN_model_n_neighbors_5.joblib')
              st.write(f"Selected model is K-nearest neighbours with n_neighbors=5")
          elif n_neighbors == 7: #repeat as before
              clf = joblib.load('KNN_model_n_neighbors_7.joblib')
              st.write(f"Selected model is K-nearest neighbours with n_neighbors=7")
          else:
              st.write("Invalid selection for n_neighbors.")
          
      elif classifier_type == 'Decision Tree': #model selection (the selectbox for that will come later)
          st.image("decision tree accuracy curves.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
          st.caption("Training anc validation accuracy curves")
          max_depth = st.selectbox('Select max_depth', [10, 17, 20]) #create a selectbox with the appropriate parameter
          if max_depth == 10: #for each parameter selection:
              clf = joblib.load('Decision Tree_model_max_depth_10.joblib') #load the pre-trained model
              st.write(f"Selected model is Decision Tree with max_depth=10") #display model description
              st.write(f"Feature importance for the current model:") #display a heading for the dataframe below
              feats = joblib.load('feature_importance_df10.joblib') #load the dataframe with the feature importances for the CURRENT model
              st.dataframe(feats) #display dataframe

          elif max_depth == 17: #repeat as before
              clf = joblib.load('Decision Tree_model_max_depth_17.joblib')
              st.write(f"Selected model is Decision Tree with max_depth=17")
              st.write(f"Feature importance for the current model:")
              feats = joblib.load('feature_importance_df17.joblib')
              st.dataframe(feats)

          elif max_depth == 20: #repeat as before
              clf = joblib.load('Decision Tree_model_max_depth_20.joblib')
              st.write(f"Selected model is Decision Tree with max_depth=20")
              st.write(f"Feature importance for the current model:")
              feats = joblib.load('feature_importance_df20.joblib')
              st.dataframe(feats)

          else:
              st.write("Invalid selection for max_depth.")
          
      elif classifier_type == 'Random Forest':
          st.image("random_forest_accuracy_curves.png.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
          st.caption("Training anc validation accuracy curves")
          n_estimators = st.selectbox('Select n_estimators', [20, 50, 100])
          if n_estimators==20:
            clf = joblib.load(f'Random Forest_model_n_estimators_20.joblib')
            st.write(f"Selected model is Random Forest with n_estimators=20")
            st.write(f"Feature importance for the current model:")
            feats = joblib.load('feature_importance_rf20.joblib')
            st.dataframe(feats)

          elif n_estimators==50:
            clf = joblib.load(f'Random Forest_model_n_estimators_50.joblib')
            st.write(f"Selected model is Random Forest with n_estimators=50")
            st.write(f"Feature importance for the current model:")
            feats = joblib.load('feature_importance_rf50.joblib')
            st.dataframe(feats)

          elif n_estimators==100:
            clf = joblib.load(f'Random Forest_model_n_estimators_100.joblib')
            st.write(f"Selected model is Random Forest with n_estimators=100")
            st.write(f"Feature importance for the current model:")
            feats = joblib.load('feature_importance_rf100.joblib')
            st.dataframe(feats)

      elif classifier_type == 'Voting Classifier':
          base_models = st.selectbox('Select base models', ["KNN, Random Forest, Logistic Regression", "KNN, Decision Tree, Random Forest"])
          if base_models == "KNN, Random Forest, Logistic Regression" :
            clf = joblib.load(f'Voting Classifier_knn_rf_lr.joblib')
            st.write(f"Selected model is Voting Classifier with KNN, Decision Tree and Logistic Regression as base models")
          elif base_models == "KNN, Decision Tree, Random Forest":
            clf = joblib.load(f'Voting Classifier_knn_rf_dt.joblib')
            st.write(f"Selected model is Voting Classifier with KNN, Random Forest and Decision Tree as base models")
          
      elif classifier_type == 'Stacking Classifier':
          metaclassifier = st.selectbox('Select meta-classifier', ["Gradient Boosting", "Bagging", "XGBoost"])
          if metaclassifier == "Gradient Boosting" :
            clf = joblib.load(f'Stacking_Classifier_Gradient_Boosting.joblib')
            st.write(f"Selected model is Stacking Classifier with KNN, Decision Tree and Random Forest as base models and Gradient Boosting as meta-classifier")
          elif metaclassifier == "Bagging":
            clf = joblib.load(f'Stacking_Classifier_Bagging.joblib')
            st.write(f"Selected model is Stacking Classifier with KNN, Decision Tree and Random Forest as base models and Bagging as meta-classifier")
          elif metaclassifier == "XGBoost":
            clf = joblib.load(f'Stacking_Classifier_XGBoost.joblib')
            st.write(f"Selected model is Stacking Classifier with KNN, Decision Tree and Random Forest as base models and XGBoost as meta-classifier")
          
      else:
          st.write("Invalid classifier_type.")

      return clf
    
    # Set condition for model
    classifier_type = st.selectbox('Choice of the model', ['KNN', 'Decision Tree', 'Random Forest', "Voting Classifier", "Stacking Classifier"]) #create selectbox where the model can be selected
    #st.write('The chosen model is:', classifier_type) #display the model selection

    # Call the prediction function
    clf = prediction(classifier_type)

  # Check if prediction was successful
    if clf is not None:
        # Evaluation metrics
        y_pred = clf.predict(X_test)
        choice = st.selectbox('Choice of the metric', ['Accuracy', 'Classification report', 'Confusion matrix'])
        
        # Call the scores function
        if choice == 'Accuracy':
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        elif choice == 'Classification report':
            class_report = classification_report(y_test, y_pred, output_dict=True)
            class_report_df = pd.DataFrame(class_report).transpose()
            st.dataframe(class_report_df)
        elif choice == "Confusion matrix":
            con_matrix = pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predicted'])
            st.dataframe(con_matrix)
            