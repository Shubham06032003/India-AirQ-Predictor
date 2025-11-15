# AQI Prediction Dashboard

An AI-powered web application that predicts Air Quality Index using machine learning.

## Overview

This project predicts air quality levels across Indian cities using a Random Forest machine learning model. The interactive dashboard helps users understand air pollution by analyzing pollutant data and providing real-time AQI predictions with health recommendations.

## What It Does

The application allows users to:
- Select a city from major Indian metros
- Input or adjust pollutant concentration levels
- Get instant AQI predictions with 90.9% accuracy
- View historical air quality trends
- Understand health impacts through color-coded categories

## Technologies Used

- Python 3.8 - Core programming language
- Streamlit - Web application framework
- Scikit-learn - Machine learning library for Random Forest model
- Pandas - Data manipulation and analysis
- NumPy - Numerical computations
- Plotly - Interactive data visualizations

## Project Structure

The project contains:
- app folder with main.py - The Streamlit web interface
- src folder with model.py - Model training and preprocessing
- data folder with city_day.csv - Historical AQI dataset
- model folder - Saved trained model and preprocessing files
- requirements.txt - List of Python dependencies

## How It Works

### Data Processing
The system loads historical AQI data containing 29,000+ records from 2015-2020. Missing values are filled using median values. City names are converted to numerical codes. All features are normalized to ensure fair contribution to predictions.

### Machine Learning Model
Uses Random Forest Regressor algorithm with 50 decision trees and maximum depth of 10. The model learns from 14 features including city, year, and 12 different pollutants. Training achieves 90.9% R-squared score with mean absolute error of 20.73.

### Prediction Process
User selects their city and adjusts sliders for pollutant levels or uses preset average values. The model processes these inputs through the trained algorithm and outputs predicted AQI value. Results display with color-coded health categories ranging from Good to Hazardous.

## Model Performance

The trained model achieves:
- R-squared Score: 90.9 percent
- Mean Absolute Error: 20.73 AQI points
- Root Mean Squared Error: 40.82
- Average prediction error: 12.56 percent of mean AQI

Most important features for prediction:
- PM2.5 particulate matter contributes 49.2 percent
- Carbon monoxide contributes 36.9 percent  
- PM10 particulate matter contributes 3.6 percent
- Other pollutants contribute less than 2 percent each

## Dataset Information

The dataset contains air quality measurements from major Indian cities including Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Kolkata, Lucknow, Ahmedabad, and others.

Time period covered: 2015 to 2020
Total records: Over 29,000 observations
Measurements per record: 14 features

Pollutants tracked:
- PM2.5 and PM10 - Fine and coarse particulate matter
- NO, NO2, NOx - Nitrogen oxides and compounds
- NH3 - Ammonia
- CO - Carbon monoxide
- SO2 - Sulfur dioxide
- O3 - Ground-level ozone
- Benzene, Toluene, Xylene - Volatile organic compounds

## Installation

To run this project locally:

Create a virtual environment and activate it
Install all required packages from requirements.txt using pip
Navigate to the app directory
Run the Streamlit application using the main.py file
Access the dashboard in your web browser on localhost port 8501

## Using the Dashboard

### Making Predictions
Open the Prediction tab in the interface. Select your city from the dropdown menu. Adjust pollutant level sliders or keep default average values. Click the Predict AQI button. View the predicted AQI value with health category and emoji indicator. Read the color-coded health recommendation.

### Viewing Historical Data
Switch to the Historical Data tab. Select a city to analyze. View interactive line charts showing AQI trends over time. Check statistics including average, maximum, and minimum AQI values for the selected city.

## Health Categories

The application classifies AQI into six categories:

Good (0 to 50): Air quality is satisfactory with minimal health impact
Moderate (51 to 100): Acceptable quality but unusually sensitive people should limit outdoor exertion
Unhealthy (101 to 200): Everyone may experience health effects, sensitive groups more seriously affected
Very Unhealthy (201 to 300): Health alert, everyone may experience serious health effects
Hazardous (301 to 500): Emergency conditions, entire population likely affected

## Training Your Own Model

To retrain the model with updated data:

Navigate to the src directory
Run the model.py script
The script will load and clean data, encode categorical variables, train the Random Forest model, evaluate performance, and save the trained model and preprocessing objects

Training output includes feature importance rankings and accuracy metrics.

## Future Improvements

Potential enhancements include:
- Integration with real-time air quality APIs
- Addition of weather data for improved predictions
- Expansion to include global cities
- Mobile application version
- Email or SMS alerts for hazardous AQI levels
- GPS-based automatic location detection
- Forecasting future AQI trends
- Comparison tools for multiple cities

## Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline from data to deployment
- Data preprocessing and feature engineering techniques
- Random Forest algorithm implementation and optimization
- Web application development with Streamlit
- Interactive data visualization with Plotly
- Model serialization and deployment
- User interface design for ML applications

## Acknowledgments

This project was built as part of learning AI and machine learning concepts. The goal is to raise awareness about air quality and help people make informed decisions about outdoor activities based on pollution levels.

Dataset source: Historical air quality monitoring data from Indian cities
Built using open-source tools and libraries
Inspired by environmental health awareness initiatives

## Technical Details

The Random Forest model uses:
- 50 estimators (decision trees)
- Maximum tree depth of 10 levels
- Minimum 5 samples required to split a node
- Minimum 2 samples required at leaf nodes
- Parallel processing enabled for faster training

Data preprocessing includes:
- Median imputation for missing numerical values
- Label encoding for city names
- Standard scaling for feature normalization
- Train-test split ratio of 80-20

## Contact and Support

For questions about this project or suggestions for improvements, please feel free to reach out or open an issue in the repository.

Built with Python and passion for clean air.
