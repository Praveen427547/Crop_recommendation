import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾")

st.title("Crop Recommendation System")

# Load data and prepare model
@st.cache_resource
def load_data_and_prepare_model():
    # Load datasets
    df = pd.read_excel('Crop_recommendation.xlsx', engine='openpyxl')  
    Kerala = pd.read_excel('Kerala_data.xlsx', engine='openpyxl')
    Himachal_Pradesh = pd.read_excel('HP_data.xlsx', engine='openpyxl')
    Uttarakhand = pd.read_excel('Uttarakhand_data.xlsx', engine='openpyxl')
    


    
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
    
    # Calculate crop parameter statistics
    crop_parameter_stats = get_crop_parameter_stats(df)
    
    return rf_model, label_encoder, scaler, poly, selector, df, crop_parameter_stats, Kerala, Himachal_Pradesh, Uttarakhand

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

# Function to fetch expected values for a given state and season
def get_expected_values(state, season, dataset, Kerala, Himachal_Pradesh, Uttarakhand):
    # Choose the correct dataset based on user input
    if dataset == "Kerala":
        selected_dataset = Kerala
    elif dataset == "Himachal Pradesh":
        selected_dataset = Himachal_Pradesh
    elif dataset == "Uttarakhand":
        selected_dataset = Uttarakhand
    else:
        st.error(f"Dataset '{dataset}' is not valid. Choose from Kerala, Himachal Pradesh, or Uttarakhand.")
        return None
    
    state_data = selected_dataset[selected_dataset["state"].str.lower() == state.lower()]
    if state_data.empty:
        st.error(f"State '{state}' not found in the dataset.")
        return None

    season_map = {"zaid": "zaid", "rabi": "rabi", "kharif": "kharif"}
    if season.lower() not in season_map:
        st.error(f"Season '{season}' is not valid. Choose from zaid, rabi, or kharif.")
        return None

    season_suffix = season_map[season.lower()]

    expected_temperature = state_data[f"temperature_{season_suffix}"].values[0]
    expected_humidity = state_data[f"humidity_{season_suffix}"].values[0]
    expected_rainfall = state_data[f"rainfall_{season_suffix}"].values[0]
    expected_n = state_data["N"].values[0]
    expected_p = state_data["P"].values[0]
    expected_k = state_data["K"].values[0]

    return expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall

# Load the model and data
try:
    rf_model, label_encoder, scaler, poly, selector, df, crop_parameter_stats, Kerala, Himachal_Pradesh, Uttarakhand = load_model_and_data()
except Exception as e:
    st.error(f"Error loading model and data: {e}")
    st.stop()

# Create Streamlit interface
st.subheader("Input Parameters")

# UI for dataset selection
col1, col2 = st.columns(2)
with col1:
    dataset_choice = st.selectbox(
        "Select dataset", 
        ["Kerala", "Himachal Pradesh", "Uttarakhand"]
    )
    
    state = st.text_input("Enter state name").strip()
    season = st.selectbox("Select season", ["zaid", "rabi", "kharif"])

# Get expected values
expected_values = None
if state and season:
    try:
        expected_values = get_expected_values(state, season, dataset_choice, Kerala, Himachal_Pradesh, Uttarakhand)
    except Exception as e:
        st.error(f"Error retrieving expected values: {e}")

# Input fields with default values from expected values
with col2:
    if expected_values:
        expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall = expected_values
        
        N = st.number_input(f"N (Nitrogen) [Expected: {expected_n}]", value=float(expected_n))
        P = st.number_input(f"P (Phosphorus) [Expected: {expected_p}]", value=float(expected_p))
        K = st.number_input(f"K (Potassium) [Expected: {expected_k}]", value=float(expected_k))
        rainfall = st.number_input(f"Rainfall [Expected: {expected_rainfall}]", value=float(expected_rainfall))
        humidity = st.number_input(f"Humidity [Expected: {expected_humidity}]", value=float(expected_humidity))
        temperature = st.number_input(f"Temperature [Expected: {expected_temperature}]", value=float(expected_temperature))
    else:
        N = st.number_input("N (Nitrogen)", value=50.0)
        P = st.number_input("P (Phosphorus)", value=30.0)
        K = st.number_input("K (Potassium)", value=20.0)
        rainfall = st.number_input("Rainfall", value=100.0)
        humidity = st.number_input("Humidity", value=75.0)
        temperature = st.number_input("Temperature", value=25.0)

# Advanced settings in expander
with st.expander("Advanced Settings"):
    proximity_threshold = st.slider("Proximity Threshold", 0.05, 0.50, 0.15, 0.01)
    min_score_threshold = st.slider("Minimum Score Threshold", 1.0, 6.0, 3.0, 0.1)

# Prediction button
predict_button = st.button("Predict Crop")

# Display results
if predict_button:
    try:
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
            proximity_threshold
        )
        best_matches = get_best_matching_crops(
            match_scores, 
            top_n=3, 
            min_score_threshold=min_score_threshold
        )
        
        # Display prediction results
        st.subheader("Prediction Results")
        
        # Final recommendation: combine both approaches
        if best_matches and predicted_crop == best_matches[0][0]:
            # Model and parameter matching agree
            st.success(f"Recommended Crop: {predicted_crop}")
        elif best_matches and predicted_crop != best_matches[0][0]:
            top_match_crop = best_matches[0][0]
            top_match_score = best_matches[0][1]["total_score"]
            
            if max_prob > 0.7:  # High confidence in the model
                st.success(f"Recommended Crop: {predicted_crop}")
            elif top_match_score >= 4.5:  # Strong parameter matching
                st.success(f"Recommended Crop: {top_match_crop}")
            else:
                st.success(f"Recommended Crops: {predicted_crop} or {top_match_crop}")
        else:
            # No strong parameter matches, rely on model
            st.success(f"Recommended Crop: {predicted_crop}")
            
        # Detailed results in expander
        with st.expander("See detailed prediction analysis"):
            st.write("### Model Prediction")
            st.write(f"Model predicted crop: {predicted_crop} with confidence: {max_prob:.2f}")
            
            st.write("### Top Parameter Matches")
            if best_matches:
                for crop, scores in best_matches:
                    st.write(f"**{crop}** - Total Score: {scores['total_score']:.2f}, Avg Score: {scores['avg_score']:.2f}")
                    param_names = ["N", "P", "K", "rainfall", "humidity", "temperature"]
                    for i, param in enumerate(param_names):
                        st.write(f"- {param}: Match Score {scores['param_scores'][i]:.2f}")
            else:
                st.write("No strong parameter matches found.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
