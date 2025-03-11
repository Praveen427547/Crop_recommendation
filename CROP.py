import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import os

st.set_page_config(page_title="Crop Recommendation System", layout="wide")

st.title("Crop Recommendation System")
st.write("This application helps farmers decide which crop to plant based on soil nutrients and environmental factors.")

# Data loading and model preparation
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
    
    # Function to get optimal parameter ranges and means for each crop
    crop_parameter_stats = get_crop_parameter_stats(df)
    
    return df, Kerala, Himachal_Pradesh, Uttarakhand, rf_model, label_encoder, scaler, poly, selector, crop_parameter_stats

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
            "param_scores": {param_names[i]: param_scores[i] for i in range(len(param_names))}
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
def get_expected_values(state, season, dataset, datasets):
    # Choose the correct dataset based on user input
    if dataset == "Kerala":
        selected_dataset = datasets[1]  # Kerala
    elif dataset == "Himachal Pradesh":
        selected_dataset = datasets[2]  # Himachal_Pradesh
    elif dataset == "Uttarakhand":
        selected_dataset = datasets[3]  # Uttarakhand
    else:
        st.error(f"Dataset '{dataset}' is not valid. Choose from Kerala, Himachal Pradesh, or Uttarakhand.")
        return None
    
    state_data = selected_dataset[selected_dataset["state"].str.lower() == state.lower()]
    if state_data.empty:
        st.error(f"District '{state}' not found in the {dataset} dataset.")
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

# Load data and prepare model
try:
    datasets = load_data_and_prepare_model()
    df, Kerala, Himachal_Pradesh, Uttarakhand, rf_model, label_encoder, scaler, poly, selector, crop_parameter_stats = datasets
    st.success("Data and model loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Create sidebar for inputs
st.sidebar.header("Input Parameters")

# Dataset selection
dataset_choice = st.sidebar.selectbox(
    "Select Region",
    ["Kerala", "Himachal Pradesh", "Uttarakhand"],
    index=0
)

# Get unique districts from the selected dataset
if dataset_choice == "Kerala":
    districts = Kerala["state"].unique().tolist()
elif dataset_choice == "Himachal Pradesh":
    districts = Himachal_Pradesh["state"].unique().tolist()
else:  # Uttarakhand
    districts = Uttarakhand["state"].unique().tolist()

# District selection
district = st.sidebar.selectbox("Select District", districts)

# Season selection
season = st.sidebar.selectbox("Select Season", ["kharif", "rabi", "zaid"])

# Get expected values based on selections
expected_values = get_expected_values(district, season, dataset_choice, datasets)

if expected_values:
    expected_n, expected_p, expected_k, expected_temperature, expected_humidity, expected_rainfall = expected_values
    
    # Create two columns for inputs
    col1, col2 = st.sidebar.columns(2)
    
    # First column: Soil nutrients
    with col1:
        st.subheader("Soil Nutrients")
        N = st.number_input(f"Nitrogen (N)", min_value=0.0, max_value=500.0, value=float(expected_n), step=1.0)
        P = st.number_input(f"Phosphorus (P)", min_value=0.0, max_value=500.0, value=float(expected_p), step=1.0)
        K = st.number_input(f"Potassium (K)", min_value=0.0, max_value=500.0, value=float(expected_k), step=1.0)
    
    # Second column: Environmental factors
    with col2:
        st.subheader("Environmental Factors")
        temperature = st.number_input(f"Temperature (°C)", min_value=0.0, max_value=50.0, value=float(expected_temperature), step=0.1)
        humidity = st.number_input(f"Humidity (%)", min_value=0.0, max_value=100.0, value=float(expected_humidity), step=0.1)
        rainfall = st.number_input(f"Rainfall (mm)", min_value=0.0, max_value=5000.0, value=float(expected_rainfall), step=1.0)

    # Prediction button
    predict_button = st.sidebar.button("Predict Recommended Crop", type="primary")
    
    # Set up the main content area
    if predict_button:
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
        
        # Display the results
        st.header("Crop Recommendation Results")
        
        # Create columns for recommendation and details
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"## 🌱 Recommended Crop: **{final_crop}**")
            
            st.markdown("### Why this crop?")
            
            # Display model confidence
            st.write(f"**Model Confidence:** {max_prob:.2f}")
            
            # Display parameter match score if available
            if best_matches:
                st.write(f"**Parameter Match Score:** {best_matches[0][1]['total_score']:.2f}")
                
                # Show whether final recommendation came from model or parameter matching
                if final_crop == predicted_crop:
                    st.info("The machine learning model has high confidence in this recommendation.")
                else:
                    st.info("This recommendation is based on optimal growing conditions for the crop.")
        
        with col2:
            # Display the input values and their match to the recommended crop
            st.markdown("### Input Values Analysis")
            
            # Get match scores for the recommended crop
            recommended_crop_scores = match_scores[final_crop]["param_scores"]
            
            # Create a DataFrame to display the inputs and their match scores
            param_names = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", 
                           "Rainfall (mm)", "Humidity (%)", "Temperature (°C)"]
            param_keys = ["N", "P", "K", "rainfall", "humidity", "temperature"]
            param_values = [N, P, K, rainfall, humidity, temperature]
            
            match_data = {
                "Parameter": param_names,
                "Your Value": param_values,
                "Optimal Range": [f"{crop_parameter_stats[final_crop][k]['min']:.1f} - {crop_parameter_stats[final_crop][k]['max']:.1f}" 
                                  for k in param_keys],
                "Match Score": [recommended_crop_scores[k] for k in param_keys]
            }
            
            match_df = pd.DataFrame(match_data)
            
            # Function to color code based on match score
            def color_match_score(val):
                if val >= 0.8:
                    return 'background-color: #c6efce; color: #006100'  # Green
                elif val >= 0.4:
                    return 'background-color: #ffeb9c; color: #9c5700'  # Yellow
                else:
                    return 'background-color: #ffc7ce; color: #9c0006'  # Red
            
            # Apply styling
            styled_df = match_df.style.applymap(color_match_score, subset=['Match Score'])
            
            # Display the styled DataFrame
            st.dataframe(styled_df, hide_index=True)
        
        # Display alternative crop suggestions
        st.markdown("### Alternative Crop Suggestions")
        
        # Get top 3 crops (excluding the recommended one)
        alt_crops = []
        for crop, scores in sorted(match_scores.items(), key=lambda x: x[1]["total_score"], reverse=True):
            if crop != final_crop and scores["total_score"] >= 3.0:
                alt_crops.append((crop, scores["total_score"]))
                if len(alt_crops) >= 3:
                    break
        
        if alt_crops:
            alt_col1, alt_col2, alt_col3 = st.columns(3)
            
            for i, (crop, score) in enumerate(alt_crops):
                col = [alt_col1, alt_col2, alt_col3][i]
                with col:
                    st.markdown(f"#### {crop}")
                    st.write(f"Match Score: {score:.2f}")
                    
                    # Get what parameters are good for this crop
                    crop_param_scores = match_scores[crop]["param_scores"]
                    good_params = [param_names[i] for i, k in enumerate(param_keys) if crop_param_scores[k] >= 0.7]
                    
                    if good_params:
                        st.write("Good match for:")
                        st.write(", ".join(good_params))
        else:
            st.write("No strong alternative matches found.")
else:
    st.warning("Please select valid district, season, and dataset to proceed.")

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.info("""
This application uses machine learning and soil science to recommend crops 
based on soil nutrients and environmental factors specific to your region. 
The model combines a Random Forest classifier with parameter matching to provide 
the best recommendation.
""")

st.markdown("---")
st.markdown("""
### How to use this app:
1. Select your region, district, and season from the sidebar
2. Adjust soil and environmental parameters if needed (default values are pre-filled based on historical data)
3. Click "Predict Recommended Crop" to see results
4. Review the recommendation and alternative suggestions
""")
