# Air Pollution Forecasting Project
## Overview
This project aims to predict Air Quality Index (AQI) values based on historical data using machine learning techniques, specifically Long Short-Term Memory (LSTM) networks. The project analyzes air quality data, applies feature engineering techniques, and builds a forecasting model to predict future AQI values. Additionally, traditional models like ARIMA are used for comparison with the LSTM approach.
## Dataset
The dataset used in this project contains historical air quality data, including various pollution parameters like CO (Carbon Monoxide) concentration. The dataset is stored in an Excel file (`air_quality_dataset.xlsx`) and includes timestamps, pollutant concentrations and other environmental features.
## Features
- Time-based features such as hour, day, and month
- Moving averages and rolling statistics for better trend analysis
- Lag features for time-series forecasting
## Libraries Used
- **Pandas**: For data manipulation and processing
- **Numpy**: For numerical operations and handling missing values
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For data scaling and evaluation metrics
- **TensorFlow & Keras**: For building and training the LSTM model
- **Statsmodels**: For implementing the ARIMA model
## Steps to Run the Project
### 1. Clone the Repository
To get started with the project, first clone this repository to your local machine:
```bash
git clone <repository_url>
```
### 2. Install Required Libraries
Ensure that you have Python 3.x installed. Then, install the required libraries using pip:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file contains all necessary dependencies for the project.
### 3. Prepare the Dataset
Ensure that you have the `air_quality_dataset.xlsx` file in the project directory. This dataset is used for training and testing the model.
### 4. Run the Script
To run the project, execute the Python script (`main.py` or the respective script you use for training and testing):
```bash
python main.py
```
This will:
- Load and preprocess the data
- Split the data into training and testing sets
- Apply scaling to the features
- Train an LSTM model and evaluate it
- Compare the LSTM model with the ARIMA model
- Visualize the results, including actual vs predicted AQI values
### 5. Visualize the Results
The script will generate various plots:
- **Actual vs Predicted AQI**: A comparison of the real AQI values with the predicted values.
- **Rolling Mean Trends**: To highlight the smoothed trends of AQI.
- **Residuals**: To visualize the errors made by the model.
### 6. Model Evaluation
After training, the script will output model performance metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
These metrics will help evaluate the model's prediction accuracy.
## Conclusion
The project demonstrates the use of LSTM for time-series forecasting of air quality. While the LSTM model shows promise, future improvements such as hyperparameter tuning, additional feature engineering, or other machine learning techniques could help increase the accuracy of AQI predictions.
