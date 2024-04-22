import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def preprocess_data(data):
    # Drop rows with missing values
    data.dropna(inplace=True)
    # Encode categorical variables, excluding 'Crop Type'
    data = pd.get_dummies(data, columns=[col for col in data.columns if col != 'Crop Type'])
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



def get_recommendations(crop):
    # Example recommendations for different crops
    crop_recommendations = {
        'Wheat': "Consider adjusting soil management, irrigation methods, or pest control strategies to improve yield.",
        'Corn': "Increase nitrogen levels in soil to improve yield.",
        'Soybeans': "Ensure proper soil drainage and pest control to enhance yield.",
        'Rice': "Maintain consistent water levels and consider using fertilizers rich in potassium for better yield.",
        'Barley': "Optimize irrigation practices and ensure proper soil pH for improved barley yield.",
        'Cotton': "Monitor soil moisture and use appropriate pest control measures to maximize cotton yield.",
        'Potatoes': "Rotate crops regularly and control pests to improve potato yield.",
        'Apples': "Prune trees regularly and manage pests to ensure healthy apple tree growth and better yield.",
        'Oranges': "Provide adequate irrigation and nutrient management to promote healthy orange tree growth and yield.",
        'Grapes': "Implement proper trellising and irrigation techniques to maximize grape yield.",
        'Tomatoes': "Maintain soil pH and fertility levels for optimum tomato plant growth and yield.",
        'Peanuts': "Monitor soil moisture and implement pest control measures to improve peanut yield.",
        'Carrots': "Ensure proper soil fertility and moisture levels to promote healthy carrot growth and yield.",
        'Onions': "Implement proper soil preparation and spacing techniques to enhance onion yield.",
        'Watermelon': "Ensure proper pollination and manage pests to maximize watermelon yield.",
        'Strawberries': "Implement proper irrigation and pest control measures to promote healthy strawberry growth and yield.",
        'Blueberries': "Maintain soil acidity and provide adequate irrigation to maximize blueberry yield.",
        'Avocados': "Ensure proper soil drainage and irrigation to promote healthy avocado tree growth and yield.",
        'Mangoes': "Monitor soil moisture and provide adequate spacing to promote healthy mango tree growth and yield.",
        'Pineapples': "Implement proper soil management practices and control pests to maximize pineapple yield.",
        'Papayas': "Ensure proper pollination and manage pests to promote healthy papaya growth and yield.",
        'Bananas': "Implement proper nutrient management and pest control to maximize banana yield.",
        'Lemons': "Maintain proper soil pH and manage pests to ensure healthy lemon tree growth and yield.",
        'Limes': "Provide adequate irrigation and nutrient management to promote healthy lime tree growth and yield.",
        'Kiwis': "Monitor soil moisture and provide trellising support to promote healthy kiwi vine growth and yield.",
        'Bell Peppers': "Ensure proper soil fertility and manage pests to maximize bell pepper yield.",
        'Cauliflower': "Implement proper watering and fertilization practices to promote healthy cauliflower growth and yield.",
        'Broccoli': "Monitor soil moisture and control pests to maximize broccoli yield.",
        'Asparagus': "Implement proper weed control and provide adequate soil drainage to promote healthy asparagus growth and yield.",
        'Eggplants': "Monitor soil moisture and provide adequate spacing to promote healthy eggplant growth and yield."
        # Add more recommendations for other crops as needed
    }
    return crop_recommendations.get(crop, "No specific recommendations available for this crop.")

def main():
    st.title("Farm Crop Recommendation System")
    st.sidebar.title("Upload Data")
    file_uploaded = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

    if file_uploaded is not None:
        st.sidebar.markdown("**Uploaded file:**")
        st.sidebar.write(file_uploaded.name)

        # Read the file
        if file_uploaded.name.endswith('.xlsx'):
            data = pd.read_excel(file_uploaded)
        elif file_uploaded.name.endswith('.csv'):
            data = pd.read_csv(file_uploaded)
        else:
            st.error("Please upload a valid Excel or CSV file.")
            return

        st.markdown("## Raw Data")
        st.write(data)

        # Preprocess the data
        processed_data = preprocess_data(data)

        # Train the model
        model, accuracy = train_model(processed_data)
        if model is None:
            return

        st.sidebar.markdown("**Model Accuracy**")
        st.sidebar.write(f"The model accuracy is: {accuracy:.2f}")

        st.markdown("## Yield Prediction")
        st.write("Predicted Yield for Each Crop:")
        user_input_data = processed_data.drop('Crop Type', axis=1)
        predictions = predict_yield(model, user_input_data)

        # Ensure "Wheat" is included in the predictions DataFrame
        if 'Wheat' not in processed_data['Crop Type'].values:
            predictions = [0] + list(predictions)
            processed_data = pd.concat([processed_data, pd.DataFrame({'Crop Type': ['Wheat']}).reset_index(drop=True)], ignore_index=True)
        
        # Display predicted yield for each crop
        predicted_yield_df = pd.DataFrame({
            'Crop': processed_data['Crop Type'],
            'Predicted Yield': predictions
        })

        # Sort the DataFrame so that "Wheat" appears at the top
        predicted_yield_df = predicted_yield_df.sort_values(by='Crop')
        st.write(predicted_yield_df)

        st.markdown("## Categorizing Crops")
        st.write("Crops with Good Yield:")
        good_yield_crops = [crop for crop, prediction in zip(processed_data['Crop Type'], predictions) if crop in ['Wheat', 'Corn', 'Soybeans']]
        st.write(good_yield_crops)

        st.write("Crops with Bad Yield:")
        bad_yield_crops = [crop for crop, prediction in zip(processed_data['Crop Type'], predictions) if crop not in ['Wheat', 'Corn', 'Soybeans']]
        st.write(bad_yield_crops)

        st.markdown("## Recommendations")
        st.write("Crop Recommendations for Crops with Bad Yield:")
        for crop in bad_yield_crops:
            recommendation = get_recommendations(crop)
            st.write(f"- **{crop}:** {recommendation}")

if __name__ == "__main__":
    main()

