# Weather-Prediction

**Machine Learning Models Used**

**1.Decision Tree Regressor**
A simple tree-based model that splits the data into branches based on feature thresholds. It is easy to interpret and performs well on small to medium datasets. However, it can overfit if not pruned or tuned properly.

**2.Random Forest Regressor**
An ensemble model that builds multiple decision trees and averages their predictions to improve accuracy and reduce overfitting. It handles missing data and noisy features well, making it suitable for weather prediction.

**3.Ridge Regression**
A regularized linear regression model that adds a penalty to large coefficients, reducing overfitting and multicollinearity. It performs well when the relationship between features and output is mostly linear.

**4. XGBoost Regressor**
An optimized gradient boosting technique that builds trees sequentially to minimize error. It is known for its speed and performance on structured data and often outperforms other models in regression tasks.

**5.LSTM (Long Short-Term Memory)**
A type of Recurrent Neural Network (RNN) designed for time series data. LSTM can learn long-term dependencies and patterns in sequential weather data, making it ideal for temperature forecasting over time.

**Steps to be performed:**
1.  Dataset: Used historical weather data from ten_year_data.csv.Data covers the range from January 1, 2015 to January 31, 2025.
2.  Data Preprocessing:
    Converted the DATE column to datetime and set it as index.
    Removed columns with more than 5% missing values.
    Dropped non-numeric fields like station and name.
    Forward-filled missing values for continuity.

3. Prediction Target:
Main goal: Predict average temperature (TAVG) using past weather features.
Additional targets (if used): TMAX, TMIN.

4. Exploratory Data Analysis (EDA):Visualized trends using Line plots,Lag plots and Seasonal decomposition.Explored correlation between features.

5. Machine Learning Models: Decision Tree Regressor, Random Forest Regressor,Ridge Regression,XGBoost Regressor,LSTM (Long Short-Term Memory) – for time series modeling using TensorFlow/Keras
6. Evaluation Metrics:Mean Absolute Error (MAE),Mean Squared Error (MSE),Root Mean Squared Error (RMSE),R² Score, Mean Absolute Percentage Error (MAPE)

**How to run:**
1.Place ten_year_data.csv in the same directory as the notebook.

2.Open the Jupyter Notebook:weather_prediction_minor_project.ipynb

3.Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow statsmodels

