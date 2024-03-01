### Forecasting Model for COVID-19 
  The analysis explores trends in COVID-19 data, focusing on total cases, total deaths, and hospitalization rates. The dataset covers a period from 2020–2023. 

  The main goal of this project is to use machine learning models that, given a COVID-19 patient's current symptom, status, and medical history, will predict whether the patient will be admitted into the ICU, with the help of exploratory data analysis and basic visualization and statistical techniques. The models will explore trends in COVID-19 patients that were hospitalized in order to predict the the number of COVID-19 patients in intensive care units using ‘ICU_patients’ as the target variable.

  In this project, we implemented a K-Means Clustering model, a Random Forest model, and a Linear Regression model. We implemented several models to analyze which features are significant and to predict the number of COVID-19 patients in intensive care units. Analyzing the correlation between features and our target variable allows us to determine which variables may be significant in predicting the number of patients in the ICU. Later on, we also assessed and compared each model's accuracy to determine which model is best suited for the data. 

#### K-Means Clustering
  To gain insights into potential subgroups within the data, we implemented a K-Means Clustering Model. Three clusters were identified based on the features of total cases, total deaths, and hospitalization rates. 

#### Random Forest Model
  The Random Forest Model revealed the features that are significant in predicting the number of COVID-19 ICU patients were the total number of cases and the number of new cases.

#### Linear Regression Model
  The objective of the Linear Regression model is to analyze current trends in the dataset in order to predict the number of COVID-19 patients in intensive care units using ‘ICU_patients’ as the target variable. Analyzing the correlation between features and our target variable allows us to determine which variables may be significant in predicting the number of patients in the ICU. 
  We found that the top three variables that were strongly correlated with ‘ICU_patients’ were 'hosp_patients', 'weekly_hosp_admissions', and '7day_avg_new_deaths'. The feature with the strongest positive correlation of 0.94 was ‘hosp_patients’, we can hypothesize an increase in the number of COVID-19 patients in the hospital results in an increase in the number of patients in the ICU. We can hypothesize a number of situations in which an increase in hospital patients correlates with an increase in ICU patients. 




