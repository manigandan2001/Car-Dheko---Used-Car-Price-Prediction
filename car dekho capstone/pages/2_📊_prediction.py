import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st

import requests  # pip install requests
import streamlit as st  # pip install streamlit
from streamlit_lottie import st_lottie  # pip install streamlit-lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://lottie.host/2ab59e2d-5d12-4e5d-886f-f06fae6da6f0/xVfFD3CKVA.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=None,
    width=None,
    key= "WELCOME",
)

# Load and prepare data
def load_and_prepare_data():
    file_paths = [
        '/Users/sanjay/Downloads/flattened_bangalore_cars_with_city.xlsx',
        '/Users/sanjay/Downloads/flattened_chennai_cars_with_city.xlsx',
        '/Users/sanjay/Downloads/flattened_delhi_cars_with_city.xlsx',
        '/Users/sanjay/Downloads/flattened_hyderabad_cars_with_city.xlsx',
        '/Users/sanjay/Downloads/flattened_jaipur_cars_with_city.xlsx',
        '/Users/sanjay/Downloads/flattened_kolkata_cars_with_city.xlsx'
    ]
    
    dfs = []
    for file_path in file_paths:
        df = pd.read_excel(file_path, engine='openpyxl')
        dfs.append(df)

    cars_df = pd.concat(dfs, ignore_index=True)

    columns_to_drop = ['car_links', 'it', 'owner', 'centralVariantId', 'variantName', 'priceSaving', 
                       'priceFixedText', 'trendingText_imgUrl', 'trendingText_heading', 
                       'trendingText_desc', 'heading', 'top_Registration_Year_icon', 
                       'top_Insurance_Validity_icon', 'top_Fuel_Type_icon', 'top_Seats_icon', 
                       'top_Kms_Driven_icon', 'top_RTO_value', 'top_RTO_icon', 'top_Ownership_value', 
                       'top_Ownership_icon', 'top_Engine_Displacement_icon', 'top_Transmission_icon', 
                       'top_Year_of_Manufacture_icon', 'bottomData', 'heading.1', 'commonIcon', 
                       'top_values', 'Comfort_&_Convenience_Comfort_values', 'Interior_Interior_values', 
                       'Exterior_Exterior_values', 'Safety_Safety_values', 'Entertainment_&_Communication_Entertainment_values', 
                       'heading.2', 'commonIcon.1', 'Engine_and_Transmission_Engine_Color', 
                       'Engine_and_Transmission_Engine_Engine_Type', 'Engine_and_Transmission_Engine_Displacement', 
                       'Engine_and_Transmission_Engine_Max_Power', 'Engine_and_Transmission_Engine_Max_Torque', 
                       'Engine_and_Transmission_Engine_No_of_Cylinder', 'Engine_and_Transmission_Engine_Values_per_Cylinder', 
                       'Engine_and_Transmission_Engine_Value_Configuration', 'Engine_and_Transmission_Engine_Fuel_Suppy_System', 
                       'Engine_and_Transmission_Engine_BoreX_Stroke', 'Engine_and_Transmission_Engine_Compression_Ratio', 
                       'Engine_and_Transmission_Engine_Turbo_Charger', 'Engine_and_Transmission_Engine_Super_Charger', 
                       'Dimensions_&_Capacity_Dimensions_Length', 'Dimensions_&_Capacity_Dimensions_Width', 
                       'Dimensions_&_Capacity_Dimensions_Height', 'Dimensions_&_Capacity_Dimensions_Wheel_Base', 
                       'Dimensions_&_Capacity_Dimensions_Front_Tread', 'Dimensions_&_Capacity_Dimensions_Rear_Tread', 
                       'Dimensions_&_Capacity_Dimensions_Kerb_Weight', 'Dimensions_&_Capacity_Dimensions_Gross_Weight', 
                       'Miscellaneous_Miscellaneous_Gear_Box', 'Miscellaneous_Miscellaneous_Drive_Type', 
                       'Miscellaneous_Miscellaneous_Steering_Type', 'Miscellaneous_Miscellaneous_Turning_Radius', 
                       'Miscellaneous_Miscellaneous_Front_Brake_Type', 'Miscellaneous_Miscellaneous_Rear_Brake_Type', 
                       'Miscellaneous_Miscellaneous_Top_Speed', 'Miscellaneous_Miscellaneous_Acceleration', 
                       'Miscellaneous_Miscellaneous_Tyre_Type', 'Miscellaneous_Miscellaneous_No_Door_Numbers', 
                       'Miscellaneous_Miscellaneous_Cargo_Volumn', 'top_Wheel_Size', 'Miscellaneous_Miscellaneous_Alloy_Wheel_Size', 
                       'Dimensions_&_Capacity_Dimensions_Ground_Clearance_Unladen', 'oem', 'model', 'priceActual', 
                       'top_Engine_Displacement_value', 'top_Transmission_value', 'top_Year_of_Manufacture_value', 
                       'top_Engine', 'top_Max_Power', 'top_Torque', 'Miscellaneous_Miscellaneous_Seating_Capacity', 
                       'top_Registration_Year_value', 'top_Fuel_Type_value', 'top_Seats_value', 'top_Kms_Driven_value']
    cars_df = cars_df.drop(columns=columns_to_drop)

    cars_df['top_Mileage'] = cars_df['top_Mileage'].str.extract('(\d+\.?\d*)')
    cars_df['top_Mileage'] = pd.to_numeric(cars_df['top_Mileage'])

    cars_df['km'] = cars_df['km'].str.replace(',', '')
    cars_df['km'] = pd.to_numeric(cars_df['km'])

    def clean_price(price_str):
        price_str = price_str.replace('₹', '').strip()
        if 'Lakh' in price_str:
            price_str = price_str.replace('Lakh', '').strip()
            price_value = float(price_str.replace(',', '')) * 100000
        elif 'Crore' in price_str:
            price_str = price_str.replace('Crore', '').strip()
            price_value = float(price_str.replace(',', '')) * 10000000
        else:
            price_value = float(price_str.replace(',', ''))
        return price_value

    cars_df['price'] = cars_df['price'].apply(clean_price)

    cars_df['Current_Year'] = 2024
    cars_df['No_of_Years'] = cars_df['Current_Year'] - cars_df['modelYear']
    cars_df = cars_df.drop(columns=['Current_Year'])

    cars_df['bt'].fillna(cars_df['bt'].mode()[0], inplace=True)
    cars_df['top_Seats'].fillna(cars_df['top_Seats'].mode()[0], inplace=True)
    cars_df['top_Mileage'].fillna(cars_df['top_Mileage'].median(), inplace=True)
    cars_df['top_Insurance_Validity_value'].fillna(cars_df['top_Insurance_Validity_value'].mode()[0], inplace=True)

    cars_df = pd.get_dummies(cars_df, columns=['ft', 'bt', 'transmission', 'city'], drop_first=True)

    insurance_order = ['Not Available', 'Third Party', 'Third Party insurance', 'Comprehensive', 'Zero Dep', '1', '2']
    ordinal_encoder = OrdinalEncoder(categories=[insurance_order])
    cars_df['top_Insurance_Validity_value_encoded'] = ordinal_encoder.fit_transform(cars_df[['top_Insurance_Validity_value']])
    cars_df = cars_df.drop('top_Insurance_Validity_value', axis=1)

    def handle_outliers_iqr(df, columns):
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        return df

    columns_to_check = ['km', 'price', 'top_Mileage', 'top_Seats']
    cars_df = handle_outliers_iqr(cars_df, columns_to_check)

    return cars_df, ordinal_encoder

def train_model(cars_df):
    X = cars_df.drop(columns=['price'])
    y = cars_df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def preprocess_input(data, ordinal_encoder, feature_names):
    input_df = pd.DataFrame([data])
    
    input_df = pd.get_dummies(input_df, columns=['ft', 'bt', 'transmission', 'city'], drop_first=True)

    insurance_order = ['Not Available', 'Third Party', 'Third Party insurance', 'Comprehensive', 'Zero Dep', '1', '2']
    ordinal_encoder = OrdinalEncoder(categories=[insurance_order])
    input_df['top_Insurance_Validity_value_encoded'] = ordinal_encoder.fit_transform(input_df[['top_Insurance_Validity_value']])
    input_df = input_df.drop('top_Insurance_Validity_value', axis=1)

    if 'top_Seats' not in input_df.columns:
        input_df['top_Seats'] = 0

    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[feature_names]
    return input_df

def format_price(price):
    if price >= 1_00_00_000:
        return f"{price / 1_00_00_000:.2f} Crore"
    elif price >= 1_00_000:
        return f"{price / 1_00_000:.2f} Lakh"
    elif price >= 1_000:
        return f"{price / 1_000:.2f} Thousand"
    else:
        return f"{price:.2f} Rupees"

def main():
    # Load data and train model
    cars_df, ordinal_encoder = load_and_prepare_data()
    model = train_model(cars_df)

    st.title("Car Price Prediction")
    st.write("Enter the details of the car to predict its price:")

    # Define input fields
    car_details = {
        'modelYear': st.number_input('Model Year', min_value=1980, max_value=2024, value=2020),
        'km': st.number_input('Kilometers Driven', min_value=0, value=30000),
        'top_Mileage': st.number_input('Mileage (kmpl)', min_value=0.0, value=15.0),
        'top_Seats': st.selectbox('Number of Seats', [2, 4, 5, 6, 7, 8], index=2),
        'top_Insurance_Validity_value': st.selectbox('Insurance Validity', ['Not Available', 'Third Party', 'Third Party insurance', 'Comprehensive', 'Zero Dep', '1', '2']),
        'ft': st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Electric']),
        'bt': st.selectbox('Body Type', ['Sedan', 'Hatchback', 'SUV', 'Coupe']),
        'transmission': st.selectbox('Transmission', ['Manual', 'Automatic']),
        'city': st.selectbox('City', ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata']),
        'No_of_Years': st.number_input('Age of the Car', min_value=0, value=4)
    }

    # Get feature names from training data
    feature_names = cars_df.drop(columns=['price']).columns

    if st.button("Predict Price"):
        # Preprocess input data and make prediction
        input_data = preprocess_input(car_details, ordinal_encoder, feature_names)
        predicted_price = model.predict(input_data)[0]

        # Display the prediction with larger, colorful text
        st.markdown(f"<h2 style='color: green;'>Predicted Price: ₹{predicted_price:,.2f}</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()






#cd /Users/sanjay/capstone1/capstone\ projects/car\ dekho\ capstone
