import pandas as pd
import streamlit as st
import os
import numpy as np

@st.cache_data
# @st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    try:
        # Check if the input is a file-like object with a 'name' attribute
        if hasattr(uploaded_file, 'name'):
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                return pd.read_excel(uploaded_file)
        else:
            # Handle string path input for ideal conditions
            if isinstance(uploaded_file, str):
                if uploaded_file.endswith('.csv'):
                    return pd.read_csv(uploaded_file)
                elif uploaded_file.endswith('.xlsx'):
                    return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def calculate_predicted_yield(farm_data, ideal_crop_conditions):
    # Preparing the data for comparison
    try:
        ideal_crop_conditions.set_index('Crop', inplace=True, drop=True)
        farm_data.set_index('Crop', inplace=True, drop=True)
        farm_data = farm_data[farm_data.index.isin(ideal_crop_conditions.index)]
    except KeyError as e:
        st.error(f"Missing critical data column: {e}")
        return None

    comparable_fields = [col for col in ideal_crop_conditions.columns if col not in ['Farm ID', 'Machinery Usage']]
    similarity_scores = (farm_data[comparable_fields] == ideal_crop_conditions.loc[farm_data.index, comparable_fields]).sum(axis=1)
    total_conditions = len(comparable_fields)

    # Include machinery usage if relevant
    if 'Machinery Usage' in farm_data.columns and 'Machinery Usage' in ideal_crop_conditions.columns:
        similarity_scores += (farm_data['Machinery Usage'] == ideal_crop_conditions['Machinery Usage']).astype(int)
        total_conditions += 1

    similarity_percentage = similarity_scores / total_conditions
    yield_status = pd.cut(similarity_percentage, bins=[0, 0.5, 0.8, 1], labels=['Low', 'Moderate', 'High'], right=False)
    return pd.DataFrame({'Crop': yield_status.index, 'Predicted Yield': yield_status}).reset_index(drop=True)

def categorize_crops(predicted_yield):
    return {
        'High Yield': predicted_yield[predicted_yield['Predicted Yield'] == 'High']['Crop'].tolist(),
        'Moderate Yield': predicted_yield[predicted_yield['Predicted Yield'] == 'Moderate']['Crop'].tolist(),
        'Low Yield': predicted_yield[predicted_yield['Predicted Yield'] == 'Low']['Crop'].tolist()
    }


def get_detailed_recommendations(farm_data, ideal_crop_conditions):
    recommendations = {}
    conditions = [
        'Soil Nitrogen', 'Soil Phosphorus', 'Soil Potassium', 
        'Pest Occurrences', 'Disease Occurrences', 'Irrigation Method', 
        'Temperature', 'Rainfall', 'Machinery Usage', 'Crop Rotation', 'Pesticide Usage'
    ]

    for crop in farm_data.index:
        crop_data = farm_data.loc[crop]
        ideal_data = ideal_crop_conditions.loc[crop]
        recs = []

        for condition in conditions:
            actual = crop_data.get(condition, None)
            ideal = ideal_data.get(condition, None)
            if condition == 'Pest Occurrences':
                # Handle special case for Pest Occurrences
                if actual != ideal and actual != 'Low':
                    recs.append(f"- **{condition}**: adjust from `{actual}` to `{ideal}`.")
            
            elif condition == 'Disease Occurrences':
                # Handle special case for Disease Occurrences
                if actual == 'None':
                    # Skip adjustment if current disease occurrences are None
                    continue
                else:
                    # Check if actual value is NaN
                    if actual is not np.nan:
                        actual_val = ['None', 'Low', 'Medium','Moderate', 'High'].index(actual)
                        ideal_val = ['None', 'Low', 'Medium','Moderate', 'High'].index(ideal)
                        if actual_val > ideal_val:
                            recs.append(f"- **{condition}**: decrease from `{actual}` to `{ideal}`.")
            elif actual is not None and ideal is not None and actual != ideal:
                # General case for other conditions
                try:
                    actual_val = float(actual) if actual != 'None' else 0
                    ideal_val = float(ideal) if ideal != 'None' else 0
                    if actual_val != ideal_val:
                        action = "increase" if actual_val < ideal_val else "decrease"
                        recs.append(f"- **{condition}**: {action} from `{actual}` to `{ideal}`.")
                except ValueError:
                    # For non-numeric conditions, adjust only if they don't match and neither is 'None'
                    recs.append(f"- **{condition}**: adjust from `{actual}` to `{ideal}`.")

        if not recs:
            recommendations[crop] = "Current conditions are well-aligned with the ideal. Maintain the same practices."
        else:
            recommendations[crop] = "To meet ideal conditions or for better yield, consider adjusting:\n" + "\n".join(recs)

    return recommendations




def display_recommendations(recommendations, predicted_yield):
    # First, sort the crops based on the yield categories 'Moderate', 'Low', then 'High'
    yield_priority = {'High': 1, 'Moderate': 0, 'Low': 2}  # Lower numbers indicate higher priority
    sorted_crops = sorted(predicted_yield.items(), key=lambda x: yield_priority[x[1]])

    # Display recommendations according to the sorted order
    for crop, yield_category in sorted_crops:
        if crop in recommendations:
            with st.expander(f"Recommendations for {crop} - {yield_category} Yield"):
                st.markdown(recommendations[crop])



def display_recommendations(recommendations, predicted_yield):
    # First, sort the crops based on the yield categories 'Moderate', 'Low', then 'High'
    yield_priority = {'High': 1, 'Moderate': 0, 'Low': 2}  # Lower numbers indicate higher priority
    sorted_crops = sorted(predicted_yield.items(), key=lambda x: yield_priority[x[1]])

    # Display recommendations according to the sorted order
    for crop, yield_category in sorted_crops:
        if crop in recommendations:
            with st.expander(f"Recommendations for {crop} - {yield_category} Yield"):
                st.markdown(recommendations[crop])

def main():
    st.title("Farming Advice Personalization System")
    st.sidebar.title("Upload Files")
    
    user_data_file = st.sidebar.file_uploader("Upload user data (CSV or Excel file)", type=["csv", "xlsx"])
    if user_data_file:
        user_data = load_data(user_data_file)
        if user_data is not None:
            st.subheader("User Data")
            st.write(user_data)
            
            file_extension = 'csv' if user_data_file.name.endswith('.csv') else 'xlsx'
            ideal_conditions_file = os.path.join('FarmingRecSystem', f'ideal_crop_conditions.{file_extension}')
            
            ideal_crop_conditions = load_data(ideal_conditions_file)
            if ideal_crop_conditions is not None:
                st.subheader("Ideal Crop Conditions")
                st.write(ideal_crop_conditions)

                if user_data.equals(ideal_crop_conditions):
                    st.subheader("Predicted Yield for Each Crop")
                    st.write("Your farm conditions are perfect for growing all types of crops.")
                    st.subheader("Categorized Crops")
                    st.write("Good Yield: " + ", ".join(user_data['Crop'].unique()))
                    st.subheader("Crop Recommendations based on Predicted Yield")
                    st.write("All crops are predicted to have a high yield. No further recommendations needed.")
                else:
                    process_and_display_data(user_data, ideal_crop_conditions)
            else:
                st.error("Failed to load the ideal crop conditions file. Please check the file path and format.")
        else:
            st.error("Failed to load user data properly. Please check the file content.")
    else:
        st.info("Awaiting user data file upload.")

def process_and_display_data(user_data, ideal_crop_conditions):
    predicted_yield = calculate_predicted_yield(user_data, ideal_crop_conditions)
    if predicted_yield is not None:
        st.subheader("Predicted Yield for Each Crop")
        st.write(predicted_yield)
        
        categorized_crops = categorize_crops(predicted_yield)
        st.subheader("Categorized Crops")
        st.write(categorized_crops)
        
        recommendations = get_detailed_recommendations(user_data, ideal_crop_conditions)
        st.subheader("Detailed Crop Recommendations")
        display_recommendations(recommendations, predicted_yield.set_index('Crop')['Predicted Yield'].to_dict())

if __name__ == "__main__":
    main()
