import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load data
df1 = pd.read_csv("E:/plant detection/Data1.csv")

# App layout and design
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #FF00FF, #800000);
        background-size: cover;
    }
    .stButton button {
        background-color: green;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and input fields
st.markdown("<h1 style='text-align: center; color: yellow;'>Crop Prediction with SVM</h1>", unsafe_allow_html=True)
place = st.selectbox('Select a location:', ('Select...', 'Mangalore', 'Udupi', 'Raichur', 'Gulbarga', 'Mysuru', 'Hassan', 'Kasaragodu'))
area = st.text_input("Enter the Area (in acres)")
soil = st.selectbox('Select soil type:', ('Select...', 'Alluvial', 'Loam', 'Laterite', 'Sandy', 'Red', 'Black', 'Sandy Loam','Clay'))

# Handle form submission
if st.button("Submit"):
    if place == 'Select...' or soil == 'Select...':
        st.warning("Please select a location and soil type.")
    elif area == '':
        st.warning("Please enter a valid area.")
    else:
        try:
            area = float(area)
            
            # Filter the data based on user inputs
            df_filtered = df1[(df1['Location'] == place) & (df1['Soil type'] == soil)]
            
            if df_filtered.empty:
                st.warning("No data available for the selected location and soil type.")
            else:
                # Encode categorical variables
                le = LabelEncoder()
                df_filtered['Location'] = le.fit_transform(df_filtered['Location'])
                df_filtered['Soil type'] = le.fit_transform(df_filtered['Soil type'])
                df_filtered['Irrigation'] = le.fit_transform(df_filtered['Irrigation'])
                df_filtered['Crops'] = le.fit_transform(df_filtered['Crops'])

                # Calculate yields and price per area unit
                df_filtered['yields/area'] = df_filtered['yeilds'] / df_filtered['Area']
                df_filtered['price/area'] = df_filtered['price'] / df_filtered['Area']
                
                # Calculate mean yield and price for the selected area
                mean_yield_per_area = df_filtered['yields/area'].mean()
                mean_price_per_area = df_filtered['price/area'].mean()

                estimated_yield = mean_yield_per_area * area
                estimated_price = mean_price_per_area * area

                # Drop the 'Year' column
                df_filtered = df_filtered.drop(columns='Year')

                # Prepare features and target variables
                X = df_filtered.drop(columns='Crops')
                y = df_filtered['Crops']

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Feature scaling
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # SVM Classifier
                svm_clf = SVC(kernel='linear', random_state=42)
                svm_clf.fit(X_train, y_train)
                svm_pred = svm_clf.predict(X_test)

                # Display predicted crop name for the first example in test set
                predicted_crop = le.inverse_transform([svm_pred[0]])[0]

                # Display the results
                st.markdown(f"<h3 style='color: yellow;'>Predicted Crop: {predicted_crop}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: yellow;'>Estimated Yield (for {area} acres): {estimated_yield:.2f}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: yellow;'>Estimated Price (for {area} acres): {estimated_price:.2f}</h3>", unsafe_allow_html=True)

        except ValueError:
            st.error("Please enter a valid number for the Area.")
        except Exception as e:
            st.error("An error occurred while processing the data.")
            st.write(f"Error details: {e}") 