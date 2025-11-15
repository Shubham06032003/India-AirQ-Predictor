import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


def model_creation(data):
    X = data.select_dtypes(include=['number']).drop(columns=['AQI'], axis=1)
    y = data['AQI']
    feature_names = X.columns.tolist()
    print(f'\nFeatures ({len(feature_names)}):')
    for i, col in enumerate(feature_names, 1):
        print(f'  {i}. {col}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit scaler ONLY on training data, then transform both
    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)  # Use transform, not fit_transform

    model = RandomForestRegressor(n_estimators=50,      # Reduced from 100
        max_depth=10,         # Reduced from 20
        min_samples_split=5,  # Prevent overfitting
        min_samples_leaf=2,   # Speed up
        n_jobs=-1,            # Use all cores
        random_state=42,
        verbose=0)
    model.fit(X_train_scaler, y_train)
    print(f'Model Accuracy : {model.score(X_test_scaler, y_test)}')
    return model, scaler, feature_names


def clean_data():
    data = pd.read_csv('../data/city_day.csv')
    data.fillna(data.median(numeric_only=True), inplace=True)
    data.dropna(subset=['AQI'], inplace=True)
    data.dropna(inplace=True)
    data.Date = pd.to_datetime(data.Date)
    data.Date = data.Date.dt.year.astype(int)


    # ADD LABELENCODER (before dropping anything)
    le_city = LabelEncoder()
    data['City_encoded'] = le_city.fit_transform(data['City'])

    mapping = data[['City', 'City_encoded']].drop_duplicates().sort_values('City_encoded')
    print(mapping.to_string(index=False))
    print(f"\nTotal cities: {len(le_city.classes_)}")

    # Now drop City and AQI_Bucket
    data = data.drop(columns=['City', 'AQI_Bucket'], axis=1)

    return data, le_city  # Return encoder too!


def main():
    data, le_city = clean_data()  # Get encoder back
    model, scaler, feature_names = model_creation(data)

    # Save model
    with open('../model/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save scaler
    with open('../model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # SAVE CITY ENCODER (important for Streamlit!)
    with open('../model/city_encoder.pkl', 'wb') as f:
        pickle.dump(le_city, f)

    with open('../model/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    # print(f"Cities encoded: {list(le_city.classes_)}")


if __name__ == '__main__':
    main()
