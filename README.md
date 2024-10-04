
# Heart Disease Prediction using Python and Machine Learning
![Gemini_Generated_Image_aun8e2aun8e2aun8](https://github.com/user-attachments/assets/49df4c01-2c17-47ac-8a2f-18d67f29931e)


Overview of the Heart Disease Prediction Project using Machine Learning and Python.


# Introduction
Heart disease is one of the leading causes of death globally. This project aims to predict the likelihood of heart disease based on various health-related features such as age, sex, cholesterol levels, and more. By leveraging machine learning algorithms, this project offers a predictive model that helps to identify potential heart disease patients before clinical diagnosis.
![Screenshot 2024-10-05 011951](https://github.com/user-attachments/assets/1f73b0da-799e-4473-8ad1-211887c48df4)

This project utilizes Python's powerful libraries such as scikit-learn, Pandas, and Matplotlib to build, train, and evaluate machine learning models.

# Features

Feature importance plot highlighting the most significant features influencing heart disease prediction.

Data Preprocessing: Cleans the dataset by handling missing values, normalizing data, and encoding categorical features.
![Screenshot 2024-10-05 012614](https://github.com/user-attachments/assets/1c8183bb-72da-4d2f-aa47-c46c756c6230)
![Screenshot 2024-10-05 012743](https://github.com/user-attachments/assets/bd7dff50-dffd-469f-946d-7dd87736b6dc)


Multiple ML Models: Implements Logistic Regression, Random Forest, and Support Vector Machine (SVM) to predict heart disease.
Hyperparameter Tuning: Optimizes model performance using GridSearchCV and RandomizedSearchCV.
Model Evaluation: Evaluates models based on accuracy, precision, recall, and F1-score using confusion matrix and classification reports.
# Visualization:
 Displays data insights and model performance using various plots like ROC curves and feature importance graphs.
 ![Screenshot 2024-10-05 012933](https://github.com/user-attachments/assets/d34ca478-d49a-43cc-ae58-3b08959963fb)
![Screenshot 2024-10-05 013008](https://github.com/user-attachments/assets/826143c2-af4e-4fa1-97f0-bbed47f10412)

# Technologies Used
Python: Programming language for data processing, analysis, and model building.
Pandas & NumPy: Data manipulation and analysis.
Matplotlib & Seaborn: Data visualization.
Scikit-learn: Machine learning model training and evaluation.
Jupyter Notebook: For interactive coding and visualizations.
# Dataset Overview
The dataset contains health-related metrics used to predict heart disease risk. Key features include:
![Screenshot 2024-10-05 013238](https://github.com/user-attachments/assets/105732f4-33c3-4a33-95eb-a5b9bdfa18d6)


Age: Age of the patient.
Sex: Male or female.
Chest pain type: Describes the type of chest pain (4 values).
Resting blood pressure: Blood pressure at rest (in mm Hg).
Cholesterol: Serum cholesterol level in mg/dL.
Max heart rate achieved: Maximum heart rate during a stress test.
Exercise-induced angina: Whether exercise causes chest pain.
![Screenshot 2024-10-05 011951](https://github.com/user-attachments/assets/f279597d-0aee-45a8-b3c1-888b7ea3b356)

Other key features: Thalach, Oldpeak, Slope, and more.
Modeling and Machine Learning
Data Preprocessing

# Handling Missing Values: 
![Screenshot 2024-10-05 013455](https://github.com/user-attachments/assets/a9eea7dc-a61b-4f97-a0d9-ac9daf27d576)

Cleaned and imputed missing data.
Scaling: Applied MinMaxScaler to normalize the data between 0 and 1.
Encoding: One-hot encoding was applied for categorical variables like chest pain type, rest ECG results, etc.
Modeling


Confusion matrix showing the performance of the logistic regression model.

# Logistic Regression:
 A basic classification model used as a baseline.
Random Forest: A more advanced model for improved prediction accuracy.
Support Vector Machine (SVM): Utilized for a more complex, non-linear classification.


ROC curve comparing the performance of different machine learning models.
![Screenshot 2024-10-05 013939](https://github.com/user-attachments/assets/824ccf0d-7001-44b2-8555-3c6fd816dadb)


Hyperparameter Tuning: Tuned the models using cross-validation techniques like GridSearchCV and RandomizedSearchCV for optimal performance.
Model Evaluation

Accuracy Score: Model accuracy on the test dataset.
Precision & Recall: Measures the relevance of true positives.
![Screenshot 2024-10-05 013805](https://github.com/user-attachments/assets/3039245f-0f85-4a4e-a9cd-3e7500bf559f)

F1-Score: Harmonic mean of precision and recall.
ROC-AUC Curve: Measures the trade-off between true positive rate and false positive rate.
# Key Insights
Key Features Influencing Heart Disease: Age, cholesterol, and maximum heart rate (thalach) are among the most significant features contributing to heart disease prediction.


Bar chart showing feature importance based on Random Forest Classifier.

#Model Performance: 
Random Forest outperformed other models, providing higher accuracy and better precision-recall balance.
![Screenshot 2024-10-05 013939](https://github.com/user-attachments/assets/824ccf0d-7001-44b2-8555-3c6fd816dadb)


Comparison of model accuracy across Logistic Regression, Random Forest, and SVM.
![Screenshot 2024-10-05 014214](https://github.com/user-attachments/assets/c40da57a-cdcd-4f2d-a84d-5c865db032b0)


Trade-Off Between Precision and Recall: Depending on whether false negatives or false positives are more critical, different models provide varying balances between precision and recall.


