Project Name: Examination of the technical features of cars that influence CO2 emissions

Authors: Christian Erdmann, Pinelopi Moutesidi, Sanjana Singh

Scope of the project:
The project seeks to conduct a comparative analysis of CO2 emissions originating from private vehicles, pinpoint technical characteristics of cars influencing their CO2 emissions, and forecast CO2 emissions from vehicles based on their design. 

Data Selection:
Datasets containing information on CO2 emissions, various technical characteristics of European vehicles, and details about their manufacturers were acquired from the website of the European Environment Agency (https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b?activeAccordion=1086728). The analysis was performed with the dataset from the year 2019.
Due to limitations in computational power, only a random fraction of 1/100 of the dataset was chosen as the working dataset for subsequent analysis, with a 
final size of 154997 entries. These data can be found in the file under the name "CO2_2019.csv" in the folder "Data".

Analysis:
The analysis was performed at two steps. Firstly, several data preprocessing steps were performed; the code for this can be found in the Jupyter Source file with the name "CO2_data preprocessing.ipynb" and the resulting data can be found in the file "CO2_2019_preprocessed.csv" in the folder "Data". These data can be directly introduced to the second step of the analysis, which is the application of Machine Learning algorithms. The code for this part of the analysis can be found in the file "CO2 project_Machine Learning models.ipynb".

Note: This project was completed as part of the requirements for certification by DataScientest. The code provided in this repository represents only a portion of the total project code output, primarily authored by Pinelopi Moutesidi.
