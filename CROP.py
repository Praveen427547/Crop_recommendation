import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

# Set page title
st.set_page_config(page_title="Crop Recommendation System")
st.title("Crop Recommendation System")

# Load datasets
@st.cache_data
def load_data():
    df = pd.read_excel('Crop_recommendatio.xlsx', engine='openpyxl')  
    Kerala = pd.read_excel('Kerala_data.xlsx', engine='openpyxl')
    Himachal_Pradesh = pd.read_excel('HP_data.xlsx', engine='openpyxl')
    Uttarakhand = pd.read_excel('Uttarakhand_data.xlsx', engine='openpyxl')
    return df, Kerala, Himachal_Pradesh, Uttarakhand

# Load the data
try:
    df, Kerala, Himachal_Pradesh, Uttarakhand = load_data()
    
    # Train the model (moved outside of the prediction function to avoid retraining)
    # Select features and target
    X = df[["N", "P", "K", "rainfall", "humidity", "temperature"]]
    y = df["label"]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply feature scaling to balance all parameters
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply polynomial transformation for non-linearity
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # Feature Selection: Pick best 10 features based on correlation with output
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_poly, y_encoded)

    # Handle class imbalance using SMOTE (Synthetic Minority Oversampling)
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_selected, y_encoded)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Train the Random Forest model with optimized hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Function to get optimal parameter ranges and means for each crop
    def get_crop_parameter_stats(df):
        crop_params = {}
        for crop in df["label"].unique():
            crop_data = df[df["label"] == crop]
            crop_params[crop] = {
                "N": {
                    "min": crop_data["N"].min(),
                    "max": crop_data["N"].max(),
                    "mean": crop_data["N"].mean(),
                    "std": crop_data["N"].std()
                },
                "P": {
                    "min": crop_data["P"].min(),
                    "max": crop_data["P"].max(),
                    "mean": crop_data["P"].mean(),
                    "std": crop_data["P"].std()
                },
                "K": {
                    "min": crop_data["K"].min(),
                    "max": crop_data["K"].max(),
                    "mean": crop_data["K"].mean(),
                    "std": crop_data["K"].std()
                },
                "rainfall": {
                    "min": crop_data["rainfall"].min(),
                    "max": crop_data["rainfall"].max(),
                    "mean": crop_data["rainfall"].mean(),
                    "std": crop_data["rainfall"].std()
                },
                "humidity": {
                    "min": crop_data["humidity"].min(),
                    "max": crop_data["humidity"].max(),
                    "mean": crop_data["humidity"].mean(),
                    "std": crop_data["humidity"].std()
                },
                "temperature": {
                    "min": crop_data["temperature"].min(),
                    "max": crop_data["temperature"].max(),
                    "mean": crop_data["temperature"].mean(),
                    "std": crop_data["temperature"].std()
                }
            }
        return crop_params

    # Calculate the crop parameter statistics
    crop_parameter_stats = get_crop_parameter_stats(df)

    # Function to check if a parameter is within the optimal range or close to it
    def get_parameter_match_score(param_name, param_value, crop, crop_params, proximity_threshold=0.15):
        param_stats = crop_params[crop][param_name]
        min_val, max_val = param_stats["min"], param_stats["max"]
        mean_val, std_val = param_stats["mean"], param_stats["std"]
        
        # If the value is within range, it's a perfect match
        if min_val <= param_value <= max_val:
            return 1.0
        
        # Calculate how close the value is to the range
        if param_value < min_val:
            # Calculate proximity as percentage of the range or std deviation
            range_size = max(max_val - min_val, std_val * 2)
            if range_size == 0:  # Avoid division by zero
                range_size = 1
            distance = (min_val - param_value) / range_size
        else:  # param_value > max_val
            range_size = max(max_val - min_val, std_val * 2)
            if range_size == 0:  # Avoid division by zero
                range_size = 1
            distance = (param_value - max_val) / range_size
        
        # If the distance is within the proximity threshold, it's a partial match
        if distance <= proximity_threshold:
            # Convert distance to a score between 0 and 1
            return max(0, 1 - (distance / proximity_threshold))
        
        # Otherwise, it's not a match
        return 0.0

    # Function to calculate parameter matching scores for each crop
    def calculate_parameter_match_scores(input_values, crop_params, proximity_threshold=0.15):
        match_scores = {}
        
        for crop in crop_params:
            param_scores = []
            param_names = ["N", "P", "K", "rainfall", "humidity", "temperature"]
            
            for i, param_name in enumerate(param_names):
                score = get_parameter_match_score(
                    param_name, 
                    input_values[0][i], 
                    crop, 
                    crop_params, 
                    proximity_threshold
                )
                param_scores.append(score)
            
            # Calculate overall match score (average of parameter scores)
            match_scores[crop] = {
                "total_score": sum(param_scores),
                "avg_score": sum(param_scores) / len(param_scores),
                "param_scores": param_scores
            }
        
        return match_scores

    # Function to get crops with best parameter matches
    def get_best_matching_crops(match_scores, top_n=3, min_score_threshold=3.0):
        # Filter crops by minimum total score
        filtered_crops = [(crop, scores) for crop, scores in match_scores.items() 
                        if scores["total_score"] >= min_score_threshold]
        
        # Sort by total score, descending
        sorted_crops = sorted(filtered_crops, key=lambda x: x[1]["total_score"], reverse=True)
        
        # Return top N crops
        return sorted_crops[:top_n]

    # Function to fetch expected values for a given state and season from selected dataset
    def get_expected_values(state, season, dataset):
        # Choose the correct dataset based on user input
        if dataset == "Kerala":
            selected_dataset = Kerala
        elif dataset == "Himachal Pradesh":
            selected_dataset = Himachal_Pradesh
        elif dataset == "Uttarakhand":
            selected_dataset = Uttarakhand
        else:
            raise ValueError(f"Dataset '{dataset}' is not valid. Choose from Kerala, Himachal Pradesh, or Uttarakhand.")
        
        state_data = selected_dataset[selected_dataset["state"].str.lower() == state.lower()]
        if state_data.empty:
            raise ValueError(f"State '{state}' not found in the dataset.")

        season_map = {"zaid": "zaid", "rabi": "rabi", "kharif": "kharif"}
        if season.lower() not in season_map:
            raise ValueError(f"Season '{season}' is not valid. Choose from zaid, rabi, or kharif.")

        season_suffix = season_map[season.lower()]

        expected_temperature = state_data[f"temperature_{season_suffix}"].values[0]
        expected_humidity = state_data[f"humidity_{season_suffix}"].values[0]
        expected_rainfall = state_data[f"rainfall_{season_suffix}"].values[0]
        expected_n = state_data["N"].values[0]
        expected_p = state_data["P"].values[0]
        expected_k = state_data["K"].values[0]

        return expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall

    # Create the Streamlit interface
    st.subheader("Enter your details")
    
    # Dataset selection
    dataset_choice = st.selectbox(
        "Select dataset", 
        options=["Kerala", "Himachal Pradesh", "Uttarakhand"]
    )
    
    # Get district and season
    col1, col2 = st.columns(2)
    with col1:
        state = st.text_input("Enter district name")
    with col2:
        season = st.selectbox("Select season", options=["zaid", "rabi", "kharif"])
    
    # Check if we can get expected values
    expected_values_available = False
    expected_n, expected_p, expected_k = 0, 0, 0
    expected_temperature, expected_humidity, expected_rainfall = 0, 0, 0
    
    if state and season and dataset_choice:
        try:
            expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall = get_expected_values(
                state, season, dataset_choice
            )
            expected_values_available = True
        except ValueError as e:
            st.warning(str(e))
    
    # Input fields for soil parameters and environmental conditions
    st.subheader("Soil and Environmental Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        if expected_values_available:
            N = st.number_input(f"N (Nitrogen) [Expected: {expected_n:.2f}]", value=float(expected_n))
            P = st.number_input(f"P (Phosphorus) [Expected: {expected_p:.2f}]", value=float(expected_p))
            K = st.number_input(f"K (Potassium) [Expected: {expected_k:.2f}]", value=float(expected_k))
        else:
            N = st.number_input("N (Nitrogen)", value=50.0)
            P = st.number_input("P (Phosphorus)", value=50.0)
            K = st.number_input("K (Potassium)", value=50.0)
    
    with col2:
        if expected_values_available:
            rainfall = st.number_input(f"Rainfall [Expected: {expected_rainfall:.2f}]", value=float(expected_rainfall))
            humidity = st.number_input(f"Humidity [Expected: {expected_humidity:.2f}]", value=float(expected_humidity))
            temperature = st.number_input(f"Temperature [Expected: {expected_temperature:.2f}]", value=float(expected_temperature))
        else:
            rainfall = st.number_input("Rainfall", value=100.0)
            humidity = st.number_input("Humidity", value=50.0)
            temperature = st.number_input("Temperature", value=25.0)
    
    # Prediction button
    if st.button("Get Crop Recommendation"):
        # Store input values for parameter matching
        input_features = np.array([[N, P, K, rainfall, humidity, temperature]])
        
        # 1. Model-based prediction
        input_features_scaled = scaler.transform(input_features)
        input_features_poly = poly.transform(input_features_scaled)
        input_features_selected = selector.transform(input_features_poly)

        # Predict using the trained model
        prediction_encoded = rf_model.predict(input_features_selected)[0]
        predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        prediction_probs = rf_model.predict_proba(input_features_selected)[0]
        max_prob = max(prediction_probs)
        
        # 2. Parameter matching with proximity consideration
        match_scores = calculate_parameter_match_scores(
            input_features, 
            crop_parameter_stats, 
            proximity_threshold=0.15
        )
        best_matches = get_best_matching_crops(
            match_scores, 
            top_n=3, 
            min_score_threshold=3.0
        )
        
        # Final recommendation: enhanced decision logic for a single crop
        final_crop = predicted_crop
        
        # If parameter matching found good matches
        if best_matches:
            top_match_crop = best_matches[0][0]
            top_match_score = best_matches[0][1]["total_score"]
            
            # Case 1: If model confidence is low but parameter matching is strong
            if max_prob < 0.6 and top_match_score >= 4.5:
                final_crop = top_match_crop
                
            # Case 2: If model and parameter matching disagree but are close in confidence
            elif predicted_crop != top_match_crop:
                # Find the model's confidence for the top parameter match
                top_match_index = list(label_encoder.classes_).index(top_match_crop)
                top_match_model_prob = prediction_probs[top_match_index]
                
                # If parameter match crop has comparable model probability and high parameter score
                if top_match_model_prob >= max_prob * 0.8 and top_match_score >= 4.0:
                    final_crop = top_match_crop
                # If model probability is borderline and parameter matching is decent
                elif max_prob < 0.7 and top_match_score > 3.5:
                    final_crop = top_match_crop
        
        # Display the recommendation
        st.success(f"## Recommended Crop: {final_crop}")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.warning("Please make sure all required Excel files are in the same directory as this app.")
