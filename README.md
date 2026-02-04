#  Forecasting Cryptocurrency Prices Using Bi-LSTM

This repository contains the implementation code and datasets for a **final year project** focused on forecasting **Bitcoin prices** using a **Bi-Directional LSTM (Bi-LSTM)** model.  
The repository includes data preprocessing outputs, feature-engineered datasets, reusable model code, and Jupyter notebooks used for experimentation and evaluation.

---

##  Repository Structure

### **feature_datasets/**

Contains feature-engineered datasets generated after applying lag features, rolling statistics, volatility measures, volume-based features, and target variables.  
These datasets are used directly as inputs for model training and evaluation.

---

### **preprocessed_datasets/**

Contains cleaned and preprocessed cryptocurrency price datasets.  
The data in this folder has been sorted chronologically, formatted consistently, and prepared for feature engineering.

---

### **Feature_engineering.py**

Python module responsible for feature engineering.

This file:

- Generates lagged closing price features  
- Computes moving averages and rolling volatility  
- Creates high–low spread and volume-based features  
- Defines regression targets for 7, 30, and 365-day horizons  
- Creates a binary directional target (up/down)  
- Removes rows affected by rolling window and shift operations  

This module is designed to be reusable across cryptocurrencies.

---

### **Bi_LSTM.py**

Reusable Python module defining the basic **Bi-Directional LSTM** model architecture.

This file:

- Implements the core Bi-LSTM network structure as described in the project  
- Contains only the model definition and related utilities  
- Does not include data splitting or scaling logic  
- Is intended to be imported into notebooks for training and evaluation  

---

### **Bitcoin_model.ipynb**

Jupyter Notebook dedicated to training and evaluating the Bi-LSTM model on Bitcoin data.

This notebook includes:

- Sequence creation for time-series modelling (30-day input windows)  
- MinMax scaling applied after train–test splitting  
- Two time-aware data splitting strategies:
  - Chronological 70/30 split  
  - Year-wise 70/30 split  
- Model training using the Bi-LSTM architecture  
- Price prediction using both split strategies  
- Regression evaluation using metrics such as RMSE, MAE, MSE, and R²  
- Visual comparisons of actual vs predicted prices  

---

### **Final Year Project.ipynb**

Main experimentation notebook used for end-to-end workflow development.

This notebook covers:

- Data loading and preprocessing  
- Exploratory data analysis (EDA)  
- Feature engineering integration  
- Sequence preparation for Bi-LSTM input  
- Model training and evaluation experiments  
- Analysis and validation of results  

This notebook acts as the central workspace for developing and testing the project pipeline.

---

##  Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook  
