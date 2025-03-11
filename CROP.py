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
    df = pd.read_excel('Crop_recommendatio.xlsx', engine='openpyxl')  
    Kerala = pd.read_excel('Kerala_data.xlsx', engine='openpyxl')
    Himachal_Pradesh = pd.read_excel('HP_data.xlsx', engine='openpyxl')
    Uttarakhand = pd.read_excel('Uttarakhand_data.xlsx', engine='openpyxl')
    
    # Select features and target
    X = df[["N", "P", "K", "rainfall", "humidity", "temperature"]]
    y = df["label"]

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    # Feature Selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X_poly, y_encoded)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_selected, y_encoded)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Train the Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Get crop parameter stats
    crop_parameter_stats = get_crop_parameter_stats(df)
    
    return df, Kerala, Himachal_Pradesh, Uttarakhand, rf_model, label_encoder, scaler, poly, selector, crop_parameter_stats

# Function to get optimal parameter stats for crops
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

# Parameter match score functions
def get_parameter_match_score(param_name, param_value, crop, crop_params, proximity_threshold=0.15):
    param_stats = crop_params[crop][param_name]
    min_val, max_val = param_stats["min"], param_stats["max"]
    mean_val, std_val = param_stats["mean"], param_stats["std"]
    
    if min_val <= param_value <= max_val:
        return 1.0
    
    if param_value < min_val:
        range_size = max(max_val - min_val, std_val * 2)
        if range_size == 0:
            range_size = 1
        distance = (min_val - param_value) / range_size
    else:
        range_size = max(max_val - min_val, std_val * 2)
        if range_size == 0:
            range_size = 1
        distance = (param_value - max_val) / range_size
    
    if distance <= proximity_threshold:
        return max(0, 1 - (distance / proximity_threshold))
    
    return 0.0

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
        
        match_scores[crop] = {
            "total_score": sum(param_scores),
            "avg_score": sum(param_scores) / len(param_scores),
            "param_scores": param_scores
        }
    
    return match_scores

def get_best_matching_crops(match_scores, top_n=3, min_score_threshold=3.0):
    filtered_crops = [(crop, scores) for crop, scores in match_scores.items() 
                     if scores["total_score"] >= min_score_threshold]
    
    sorted_crops = sorted(filtered_crops, key=lambda x: x[1]["total_score"], reverse=True)
    
    return sorted_crops[:top_n]

# Function to get expected values based on region, district and season
def get_expected_values(state, season, dataset, datasets):
    if dataset == "Kerala":
        selected_dataset = datasets[1]
    elif dataset == "Himachal Pradesh":
        selected_dataset = datasets[2]
    elif dataset == "Uttarakhand":
        selected_dataset = datasets[3]
    else:
        st.error(f"Dataset '{dataset}' is not valid.")
        return None
    
    state_data = selected_dataset[selected_dataset["state"].str.lower() == state.lower()]
    if state_data.empty:
        st.error(f"District '{state}' not found in the dataset.")
        return None

    season_map = {"zaid": "zaid", "rabi": "rabi", "kharif": "kharif"}
    if season.lower() not in season_map:
        st.error(f"Season '{season}' is not valid.")
        return None

    season_suffix = season_map[season.lower()]

    expected_temperature = state_data[f"temperature_{season_suffix}"].values[0]
    expected_humidity = state_data[f"humidity_{season_suffix}"].values[0]
    expected_rainfall = state_data[f"rainfall_{season_suffix}"].values[0]
    expected_n = state_data["N"].values[0]
    expected_p = state_data["P"].values[0]
    expected_k = state_data["K"].values[0]

    return expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall

# Load data and prepare model
try:
    datasets = load_data_and_prepare_model()
    df, Kerala, Himachal_Pradesh, Uttarakhand, rf_model, label_encoder, scaler, poly, selector, crop_parameter_stats = datasets
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Input area
st.subheader("Enter your details")

# First row - Region and District
col1, col2, col3 = st.columns(3)
with col1:
    dataset_choice = st.selectbox(
        "Region",
        ["Kerala", "Himachal Pradesh", "Uttarakhand"]
    )

    # Get districts based on region selection
    if dataset_choice == "Kerala":
        districts = Kerala["state"].unique().tolist()
    elif dataset_choice == "Himachal Pradesh":
        districts = Himachal_Pradesh["state"].unique().tolist()
    else:  # Uttarakhand
        districts = Uttarakhand["state"].unique().tolist()

with col2:
    district = st.selectbox("District", districts)

with col3:
    season = st.selectbox("Season", ["kharif", "rabi", "zaid"])

# Get expected values
expected_values = get_expected_values(district, season, dataset_choice, datasets)

if expected_values:
    expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall = expected_values
    
    # Input parameters with higher limits
    st.subheader("Soil and Environmental Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0, value=float(expected_n), step=1.0)
        P = st.number_input("Phosphorus (P)", min_value=0.0, value=float(expected_p), step=1.0)
    with col2:
        K = st.number_input("Potassium (K)", min_value=0.0, value=float(expected_k), step=1.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=float(expected_rainfall), step=1.0)
    with col3:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(expected_humidity), step=0.1)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=float(expected_temperature), step=0.1)
    
    # Prediction button
    if st.button("Get Recommendation", type="primary"):
        # Store input values for predictions
        input_features = np.array([[N, P, K, rainfall, humidity, temperature]])
        
        # Model-based prediction
        input_features_scaled = scaler.transform(input_features)
        input_features_poly = poly.transform(input_features_scaled)
        input_features_selected = selector.transform(input_features_poly)

        # Get prediction
        prediction_encoded = rf_model.predict(input_features_selected)[0]
        predicted_crop = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        prediction_probs = rf_model.predict_proba(input_features_selected)[0]
        max_prob = max(prediction_probs)
        
        # Parameter matching
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
        
        # Determine final recommendation
        final_crop = predicted_crop
        
        if best_matches:
            top_match_crop = best_matches[0][0]
            top_match_score = best_matches[0][1]["total_score"]
            
            # If model confidence is low but parameter matching is strong
            if max_prob < 0.6 and top_match_score >= 4.5:
                final_crop = top_match_crop
                
            # If model and parameter matching disagree but are close in confidence
            elif predicted_crop != top_match_crop:
                top_match_index = list(label_encoder.classes_).index(top_match_crop)
                top_match_model_prob = prediction_probs[top_match_index]
                
                if top_match_model_prob >= max_prob * 0.8 and top_match_score >= 4.0:
                    final_crop = top_match_crop
                elif max_prob < 0.7 and top_match_score > 3.5:
                    final_crop = top_match_crop
        
        # Display only the recommended crop
        st.success(f"# Recommended Crop: {final_crop}")
else:
    st.warning("Please select valid district, season, and region to proceed.")
