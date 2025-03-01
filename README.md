# Fraud_Detection
End to end ml pipeline which can deployed as API

## Overview
This project focuses on building an **end-to-end machine learning pipeline** for fraud detection. The goal is to classify transactions as fraudulent or legitimate using a selected machine learning algorithm.

## Model and Framework
- Utilized a **user-selected machine learning algorithm** for training the fraud detection model.  
- Designed a modular pipeline for **data preprocessing, feature engineering, model training, and evaluation**.

## API Deployment
- Developed a **Flask application** to serve the trained model as an API.  
- The API can be used for real-time fraud detection by making HTTP requests.

## Pipeline Workflow
1. **Data Preprocessing** – Handling missing values, encoding categorical features, and feature scaling.  
2. **Model Training** – Training and tuning the user-selected ML algorithm.  
3. **Evaluation** – Assessing model performance using classification metrics.  
4. **Deployment** – Serving the model via a Flask-based API.  

## Dependencies
- Python 3.x  
- Scikit-learn, Pandas, NumPy  
- Flask for API deployment  
- Requests for API testing  

## Future Enhancements
- Implement model monitoring and logging.  
- Deploy the API using cloud services like AWS, GCP, or Azure.  
- Improve model performance with ensemble techniques.  
