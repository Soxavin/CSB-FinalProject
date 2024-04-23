import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Ensure that all 30 crops are included in the dataset
all_crops = ['Wheat', 'Corn', 'Soybeans', 'Rice', 'Barley', 'Cotton', 'Potatoes', 'Apples', 'Grapes', 'Tomatoes', 'Peanuts',
                 'Onions', 'Watermelon', 'Strawberries', 'Avocados', 'Mangoes', 'Pineapples', 'Papayas', 'Bananas', 'Limes', 
                 'Kiwis', 'Bell Peppers', 'Broccoli', 'Asparagus', 'Eggplants', 'Blueberries', 'Cauliflower', 'Carrots', 'Oranges', 'Lemons']

def preprocess_data(data):
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    # Ensure 'Average Temperature' and 'Average Rainfall' are treated as categorical variables
    data['Average Temperature'] = data['Average Temperature'].astype('category')
    data['Average Rainfall'] = data['Average Rainfall'].astype('category')
    
    # Encode categorical variables, excluding 'Crop Type'
    data = pd.get_dummies(data, columns=[col for col in data.columns if col != 'Crop Type'])
    
    missing_crops = []
    for crop in all_crops:
        if crop not in data['Crop Type'].unique():
            missing_crops.append({'Crop Type': crop})
    
    # Create a DataFrame for missing crops
    missing_crops_df = pd.DataFrame(missing_crops)
    
    # Concatenate original DataFrame with DataFrame for missing crops
    data = pd.concat([data, missing_crops_df], ignore_index=True)
    
    return data



def train_model(data):
    # Check if 'Crop Type' column exists
    if 'Crop Type' not in data.columns:
        st.error("Error: 'Crop Type' column not found in dataset.")
        st.write("Available columns:", data.columns)
        return None, None

    # Split the data into features and target variable
    X = data.drop('Crop Type', axis=1)
    y = data['Crop Type']
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def predict_yield(model, user_input_data):
    # Predict yield probabilities for user input data
    yield_probabilities = model.predict_proba(user_input_data)
    # Get the index of the class with the highest probability for each sample
    predictions = yield_probabilities.argmax(axis=1)
    return predictions

def get_recommendations(crop, farm_data, ideal_data):
    # Debug print: Check column names in farm conditions DataFrame
    print("Column Names in Farm Conditions DataFrame:", farm_data.columns.tolist())
    
    # Get the ideal conditions for the crop
    ideal_conditions = ideal_data[ideal_data['Crop Type'] == crop].reset_index(drop=True)
    # Get the farm data for the crop
    farm_conditions = farm_data[farm_data['Crop Type'] == crop].drop(columns=['Crop Type']).reset_index(drop=True)
    
    # Check if both farm and ideal conditions data exist for the crop
    if not ideal_conditions.empty and not farm_conditions.empty:
        # Get the ideal temperature
        ideal_temp = ideal_conditions['Average Temperature'].iloc[0]
        # Get the farm temperature
        farm_temp = farm_conditions['Average Temperature'].iloc[0]
        
        # Check if the farm temperature matches the ideal temperature
        if farm_temp == ideal_temp:
            temp_recommendation = f"The average temperature ({farm_temp}) for {crop} matches the ideal temperature."
        else:
            temp_recommendation = f"The average temperature ({farm_temp}) for {crop} does not match the ideal temperature ({ideal_temp})."
        
        # Get the ideal rainfall
        ideal_rainfall = ideal_conditions['Average Rainfall'].iloc[0]
        # Get the farm rainfall
        farm_rainfall = farm_conditions['Average Rainfall'].iloc[0]
        
        # Check if the farm rainfall matches the ideal rainfall
        if farm_rainfall == ideal_rainfall:
            rainfall_recommendation = f"The average rainfall ({farm_rainfall}) for {crop} matches the ideal rainfall."
        else:
            rainfall_recommendation = f"The average rainfall ({farm_rainfall}) for {crop} does not match the ideal rainfall ({ideal_rainfall})."
        
        # Combine temperature and rainfall recommendations
        recommendation = f"{temp_recommendation} {rainfall_recommendation}"
        
        # Return the recommendation
        return recommendation
    
    else:
        recommendation = f"Ideal conditions or farm data not available for {crop}."
        return recommendation



def categorize_yield(predictions, threshold=10):
    categorized_yield = []
    for pred in predictions:
        if pred > threshold:
            categorized_yield.append("Good Yield")
        else:
            categorized_yield.append("Bad Yield")
    return categorized_yield

def main():
    st.title("Farm Crop Recommendation System")
    st.sidebar.title("Upload Data")
    file_uploaded = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

    if file_uploaded is not None:
        st.sidebar.markdown("**Uploaded file:**")
        st.sidebar.write(file_uploaded.name)

        # Read the farm data file
        if file_uploaded.name.endswith('.xlsx'):
            farm_data = pd.read_excel(file_uploaded)
        elif file_uploaded.name.endswith('.csv'):
            farm_data = pd.read_csv(file_uploaded)
        else:
            st.error("Please upload a valid Excel or CSV file.")
            return

        st.markdown("## Raw Data")
        st.write(farm_data)

        # Read the ideal conditions data
        ideal_data_path = "FarmingRecSystem/ideal_crop_conditions.xlsx" if file_uploaded.name.endswith('.xlsx') else "FarmingRecSystem/ideal_crop_conditions.csv"
        ideal_data = pd.read_excel(ideal_data_path) if file_uploaded.name.endswith('.xlsx') else pd.read_csv(ideal_data_path)

        # Preprocess the farm data
        farm_data_processed = preprocess_data(farm_data)

        # Train the model
        model, accuracy = train_model(farm_data_processed)
        if model is None:
            return

        st.sidebar.markdown("**Model Accuracy**")
        st.sidebar.write(f"The model accuracy is: {accuracy:.2f}")

        st.markdown("## Yield Prediction")
        st.write("Predicted Yield for Each Crop:")
        user_input_data = farm_data_processed.drop('Crop Type', axis=1)
        predictions = predict_yield(model, user_input_data)

        # Ensure "Wheat" is included in the predictions DataFrame
        if 'Wheat' not in farm_data_processed['Crop Type'].values:
            predictions = [0] + list(predictions)
            farm_data_processed = pd.concat([farm_data_processed, pd.DataFrame({'Crop Type': ['Wheat']}).reset_index(drop=True)], ignore_index=True)
        
        # Display predicted yield for each crop
        predicted_yield_df = pd.DataFrame({
            'Crop': farm_data_processed['Crop Type'],
            'Predicted Yield': predictions
        })

        st.write(predicted_yield_df)

        st.markdown("## Categorizing Crops")
        st.write("Categorized Crops based on Predicted Yield:")
        categorized_yield = categorize_yield(predictions)
        categorized_df = pd.DataFrame({
            'Crop': farm_data_processed['Crop Type'],
            'Predicted Yield': predictions,
            'Categorized Yield': categorized_yield
        })
        st.write(categorized_df)

        st.markdown("## Recommendations")
        st.write("Crop Recommendations based on Farm Data:")
        for crop in farm_data_processed['Crop Type']:
            recommendation = get_recommendations(crop, farm_data_processed, ideal_data)
            st.write(recommendation)

if __name__ == "__main__":
    main()
