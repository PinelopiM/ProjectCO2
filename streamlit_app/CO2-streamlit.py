import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Libraries for modeling
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_validate, cross_val_score

#Import data
df=pd.read_csv("CO2_2019_preprocessed.csv") #this is the final file after all the preprocessing
df19=pd.read_csv("CO2_2019.csv") #this is the final file after MOST of the preprocessing that was used for visualization (eg, electric cars are still in)

#Create presentation scaffold
st.title("Examination of the technical features of cars that influence CO2 emissions")
st.sidebar.title("Table of contents")
pages=["Data Exploration", "Data Vizualization", "Modelling-Regression","Modelling-Classification","Conclusion"]
page=st.sidebar.radio("Go to", pages)

#For page 2-Data Vizualization
if page == pages[1] : 
  st.write("### Visualization")

  selected_viz = st.selectbox("Select Visualization", ["Correlation Analysis", "CO2 Emissions by Country", "CO2 Emissions by Car Manufacturer"])

  #Display the selected visualization
  if selected_viz == "Correlation Analysis":
    var_num = df19.select_dtypes(include = ['int', 'float'])
    fig_corr=plt.figure(figsize=(16,15))
    sns.heatmap(var_num.corr(), annot=True, cmap="RdBu_r", center =0)
    st.pyplot(fig_corr)

  elif selected_viz == "CO2 Emissions by Country":
    st.write("Enter text")
    
  elif selected_viz == "CO2 Emissions by Car Manufacturer":
    df19_plot_fossil=df19[(df19["Fuel_Type"] == "diesel") | (df19["Fuel_Type"] == "petrol") |(df19["Fuel_Type"] == "lpg") | (df19["Fuel_Type"] == "ng")]
    value_counts = df19_plot_fossil["Manufacturer"].value_counts()
    most_frequent_Mk = value_counts.nlargest(10).index
    for_graph_Mk = df19_plot_fossil[df19_plot_fossil["Manufacturer"].isin(most_frequent_Mk)]

    fig_manu=plt.figure(figsize= (15,10))
    sns.barplot(x="Manufacturer", y="CO2", hue= "Fuel_Type", data=for_graph_Mk, palette="colorblind")
    plt.legend(loc='upper right')
    plt.xlabel('Manufacturer')
    plt.ylabel('CO2 emissions (g/km)')
    plt.title("CO2 emissions of fossil fuel cars per manufacturer")
    st.pyplot(fig_manu)

    fig_fleet=plt.figure(figsize= (15,10))
    plt.axhline(y=95, color='r', linestyle='--', label='European target for 2020-2024')
    sns.boxplot(x="Manufacturer", y="CO2", data=for_graph_Mk, palette="colorblind", showmeans=True)
    plt.legend(loc='upper right')
    plt.xlabel('Manufacturer')
    plt.ylabel('CO2 emissions (g/km)')
    plt.title("Fleet CO2 emissions")
    st.pyplot(fig_fleet)





#For page 4-Classification
if page == pages[3] : 
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
  st.write("Classes based on vehicle CO2 emission")
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
  axes[0].hist(df_10["CO2_Eng_given"], bins=20, rwidth=0.6, density=True)
  axes[0].set_title('Distribution of CO2 values')
  axes[0].set_xlabel('CO2 (g/km)')
  axes[0].set_ylabel('Frequency')

  sns.countplot(x='CO2_Classes', ax=axes[1], data=df_10)
  axes[1].set_title('Number of cars per CO2 class')
  axes[1].set_xlabel('Classes')
  axes[1].set_ylabel('Count')
  st.pyplot(fig)

  #Finalize the datasets
  df_class=df_10.drop(["CO2","CO2_Eng_given","CO2_Eng_misses","Country","Vehicle_Category","Fuel_Type","Fuel_Mode", "Eng_Power_missing"], axis=1)
  df_class=df_class.dropna()

  ## Split data
  target = df_class.CO2_Classes 
  data =df_class.drop( "CO2_Classes", axis=1)
  X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=66) #Split into train and test

  #Modeling
  from sklearn.metrics import classification_report, accuracy_score
  from sklearn import neighbors
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.tree import DecisionTreeClassifier, plot_tree
  from sklearn import ensemble
  from sklearn.ensemble import VotingClassifier, RandomForestClassifier


  #Models
  #Define the models
  def prediction(classifier, params):
    if classifier == 'K-nearest neighbours':
      clf = KNeighborsClassifier(**params)
    elif classifier == 'Decision Tree':
      clf = DecisionTreeClassifier(**params)
    elif classifier == 'Random Forest':
      clf = RandomForestClassifier(**params)
    elif classifier == 'Voting Classifier':
      clf = VotingClassifier(**params)
    elif classifier == 'Stacking Classifier':
      clf = StackingClassifier(**params)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

  # Set condition for model
  choice = st.selectbox('Choice of the model', ['K-nearest neighbours', 'Decision Tree', 'Random Forest',"Voting Classifier", "Stacking Classifier"])
  st.write('The chosen model is:', choice)

  # Parameters
  if choice == 'K-nearest neighbours':
    n_neighbors = st.selectbox('Select n_neighbors', [3, 5, 7])
    metric = st.selectbox('Select distance metric', ['minkowski', 'manhattan'])
  elif choice == 'Decision Tree':
    max_depth = st.selectbox('Select max_depth for Decision Trees', [5, 10, 17, 20])
  elif choice == 'Random Forest':
    n_estimators = st.selectbox('Select n_estimators for Random Forest', [10, 20, 50, 100, 150])

 # Prediction
  params = locals().get(f"{choice.lower().replace(' ', '_')}_params", {})  # Get the correct parameter dictionary
  clf, y_pred = prediction(choice, params)

  #Evaluation metrics
  def scores(clf, choice):
    y_pred=clf.predict(X_test)
    if choice == 'Accuracy':
      return accuracy_score(y_test, y_pred)
    elif choice == 'Classificatin report':
      class_report= classification_report(y_test, y_pred,output_dict=True)
      class_report_df = pd.DataFrame(class_report).transpose()
      return class_report_df
    elif choice == "Confusion matrix":
      con_matrix=pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predicted'])
      return con_matrix

    
  # Evaluation metrics display  
  #clf = prediction(choice)
  display = st.radio('What do you want to show?', ('Accuracy', "Classification report", 'Confusion matrix'))
  if display == 'Accuracy':
    st.write(scores(clf, display))
  elif display == 'Classification report':
    st.dataframe(scores(clf, display))
  elif display == 'Confusion matrix':
    st.dataframe(scores(clf, display))
