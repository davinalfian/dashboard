# streamlit_dashboard.py

import os, random
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

trials_data = pd.read_csv("./Simulation/apsim_processed.csv", index_col=0)

selected_features = ['estimatedVPD', 'sw_stress', 'sw_supply', 'sw_demand']
selected_calc = ['yield', 'grain_size', 'grain_no']

X = trials_data[selected_features]
Y = trials_data[selected_calc]
y_y = trials_data['yield']
y_s = trials_data['grain_size']
y_n = trials_data['grain_no']

year_data = trials_data.groupby("Year").agg({
    'yield': 'mean',
    'grain_size': 'mean',
    'grain_no': 'mean',
})

# Trained models
# model_rf_yield = joblib.load("./Simulation/RF_model_yield.joblib")
# model_rf_gsize = joblib.load("./Simulation/RF_model_gsize.joblib")

# model_gb_yield = joblib.load("./Simulation/GB_model_yield.joblib")
# model_gb_gsize = joblib.load("./Simulation/GB_model_gsize.joblib")

# model_rf_multi = joblib.load("./Simulation/RF_model_multivariate.joblib")
# model_rf_stand = joblib.load("./Simulation/RF_model_multistandardize.joblib")


# -----------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------

st.set_page_config(layout="wide")

# Sidebar controls
st.sidebar.title("Navigation")
st.sidebar.header("Tabular Data Analysis")
st.sidebar.markdown("[Model Performance Overview](#model-performance-overview)")
st.sidebar.markdown("[Yearly Trends](#yearly-trends)")
st.sidebar.markdown("[Feature Importance & SHAP](#feature-importance-and-feature-contributions-shap)")
st.sidebar.header("Computer Vision")
st.sidebar.markdown("[Model Comparison](#table-of-model-comparison)")
st.sidebar.markdown("[Visualization](#visualization-inference)")
st.sidebar.markdown("[Loss & Metrics](#training-and-metrics-curves)")

st.title("Tabular Data Analysis")
st.markdown("This dashboard explores APSIM outputs using Random Forest and Gradient Boosting models.")


# 1. PERFORMANCE OVERVIEW
tabular_col1, _, tabular_c2 = st.columns([10, 1, 10])

with tabular_col1:
    st.markdown("## Model Performance Overview", unsafe_allow_html=True)
    st.markdown("Compare R² between models.")
    st.markdown(" ")

    r2_rf = [0.9067, 0.6166, 0.8892]  # RF values for 2 tasks
    r2_gb = [0.9119, 0.5875, 0.8862]  # GB values for 2 tasks
    tasks = ["Yield", "Grain-Size", "Grain-Number"]

    comparison_data = pd.DataFrame({
        "Model": ["Random Forest", "Gradient Boosting", 
                "Random Forest", "Gradient Boosting",
                "Random Forest", "Gradient Boosting",
                ],
        "Task": ["Yield", "Yield", "Grain-Size", "Grain-Size", "Grain-Number", "Grain-Number"],
        "R2": [0.9067, 0.9119, 0.6166, 0.5875, 0.8892, 0.8862]
    })

    # Side-by-side bar chart
    st.bar_chart(comparison_data, x="Task", y="R2", color="Model", stack=False)

# ================================================================================================================= #
#                                                                                                                   #
# ================================================================================================================= #

# 2. YEARLY TRENDS

with tabular_c2:

    yearly_col1, yearly_col2 = st.columns([3, 1])
    with yearly_col1:
        st.markdown("## Yearly Trends", unsafe_allow_html=True) 
    
    with yearly_col2:
        year_target = st.selectbox("Select Variable", ["yield", "grain_size", "grain_no"]) 


    st.markdown("Distribution of observed and predicted variables per year.")
    color_map = {
        "yield": "orange",
        "grain_size": "goldenrod",
        "grain_no": "sandybrown"
    }

    year_bar = year_data[[year_target]].reset_index()

    # Build chart
    chart = (
        alt.Chart(year_bar)
        .mark_bar(color=color_map[year_target])
        .encode(
            x="Year:O",
            y=f"{year_target}:Q",
            tooltip=["Year", year_target]
        )
        # .properties(width=600, height=400, title=f"{year_target} by Year")
        .properties(width=600, height=300)
    )

    # Show in Streamlit
    # st.altair_chart(chart, use_container_width=True)
    st.altair_chart(chart)

# ================================================================================================================= #
#                                                                                                                   #
# ================================================================================================================= #

# 3. Feature Importance & SHAP summary plot

st.markdown("## Feature Importance and Feature Contributions (SHAP)", unsafe_allow_html=True)
target = st.selectbox("Select Model and Target variable (Model-Target)",
                       ["RandomForest-Yield", "RandomForest-GrainSize", "RandomForest-GrainNumber", 
                        "GradientBoosting-Yield", "GradientBoosting-GrainSize", "GradientBoosting-GrainNumber",])
feature_col1, _, feature_col2 = st.columns([10, 1, 10])

# Feature Importance
with feature_col1:

    rf_yield = pd.DataFrame({
        "Feature": ["air-vapor", "water-stress", "water-supply", "water-demand"],
        "Importance": [0.052787, 0.075079, 0.048475, 0.823659]
    })

    rf_size = pd.DataFrame({
        "Feature": ["air-vapor", "water-stress", "water-supply", "water-demand"],
        "Importance": [0.355363, 0.113221, 0.149190, 0.382227]
    })

    rf_number = pd.DataFrame({
        "Feature": ["air-vapor", "water-stress", "water-supply", "water-demand"],
        "Importance": [0.045335, 0.090575, 0.061543, 0.802547]
    })

    gb_yield = pd.DataFrame({
        "Feature": ["air-vapor", "water-stress", "water-supply", "water-demand"],
        "Importance": [0.058137, 0.085303, 0.041140, 0.815420]
    })

    gb_size = pd.DataFrame({
        "Feature": ["air-vapor", "water-stress", "water-supply", "water-demand"],
        "Importance": [0.360081, 0.115519, 0.137709, 0.386692]
    })

    gb_number = pd.DataFrame({
        "Feature": ["air-vapor", "water-stress", "water-supply", "water-demand"],
        "Importance": [0.042944, 0.099121, 0.053783, 0.804152]
    })

    model_data = {
        "RandomForest-Yield": rf_yield,
        "RandomForest-GrainSize": rf_size,
        "RandomForest-GrainNumber": rf_number,
        "GradientBoosting-Yield": gb_yield,
        "GradientBoosting-GrainSize": gb_size,
        "GradientBoosting-GrainNumber": gb_number,
    }

    alt_bar = model_data[target]

    # Bar chart with Altair
    chart = (
        alt.Chart(alt_bar)
        .mark_bar(size=40)
        .encode(
            x=alt.X("Importance:Q", title="Gini Importance", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Feature:N"),  # sort features by importance
            tooltip=["Feature", "Importance"]
        )
        .properties(
            title=f"Feature Importance for {target}",
            width=500,   # wider
            height=300   # taller
        )
    )

    # st.altair_chart(chart, use_container_width=True)
    st.altair_chart(chart)

# ================================================================================================================= #
#                                                                                                                   #
# ================================================================================================================= #

# FEATURE CONTRIBUTIONS (SHAP)

with feature_col2:
    # st.markdown("Shows which features drive predictions the most.")
    st.markdown(f"Shap Summary Plot for **{target}**.")

    shap_images = {
        "RandomForest-Yield": "shap_rf_yield.png",
        "RandomForest-GrainSize": "shap_rf_gsize.png",
        "RandomForest-GrainNumber": "shap_rf_number.png",
        "GradientBoosting-Yield": "shap_gb_yield.png",
        "GradientBoosting-GrainSize": "shap_gb_gsize.png",
        "GradientBoosting-GrainNumber": "shap_gb_number.png",
    }

    shap_image = "./Simulation/" + shap_images[target]

    # st.image(shap_image, use_container_width=True)
    st.image(shap_image)


# ================================================================================================================= #
#                                                                                                                   #
# ================================================================================================================= #


st.title("Computer Vision")
st.markdown("Dataset of **819** numbers of image was used for training, **110** was used for validation, and **109** was used for testing")

# 1. Model Comparison

st.markdown("## Table of Model Comparison", unsafe_allow_html=True)

models_matrix = pd.DataFrame({
    "Model": ["fcn_resnet50", "deeplabv3_resnet50", "segformer-b0"],
    "Loss Function": ["CrossEntropy Loss", "CrossEntropy Loss", "CrossEntropy Loss"],
    "Optimizer": ["AdamW optimizer", "AdamW optimizer", "AdamW optimizer"],
    "Learning Rate": ["0.0001", "0.0001", "0.0001"],
    "mean IoU": ["0.4348", "0.4162", "0.5579"],
    "mean PA": ["0.5728", "0.5575", "0.6963"],
}).set_index("Model")

models_classes = pd.DataFrame({
    "FCN_IoU": ["0.7820", "0.3289", "0.0027", "0.5996"],
    "FCN_Accuracy": ["0.8563", "0.4456", "0.0033", "0.8571"],
    "Deeplab_IoU": ["0.7811", "0.2627", "0.0074", "0.6137"],
    "Deeplab_Accuracy": ["0.8180", "0.3414", "0.0131", "0.9390"],
    "SegFormer_IoU": ["0.8100", "0.6673", "0.0181", "0.6035"],
    "SegFormer_Accuracy": ["0.8509", "0.8226", "0.0188", "0.8943"],
}, index=["Background","Head","Stem","Leaf"])

st.text(' ')
st.text(' ')

st.markdown("**The model parameters and result.**")
st.table(models_matrix)

st.text(' ')

st.markdown("**The model IoU and Accuracy for each class.**")
st.table(models_classes)


# ================================================================================================================= #
#                                                                                                                   #
# ================================================================================================================= #

# 2. Inference

st.text(' ')
st.markdown("## Visualization Inference", unsafe_allow_html=True)


inf_col1, _, inf_col2 = st.columns([3, 1, 1])

with inf_col1:
    # st.markdown("Visualization of the model")
    model_target = st.selectbox("**Select Visualization Model**", ["FCN", "Deeplab", "SegFormer",])
    images_dir = f"./inference_outputs/{model_target.lower()}"

    # initialize with one random image on first load
    if "random_image" not in st.session_state:
        files = os.listdir(images_dir)
        st.session_state["random_image"] = os.path.join(images_dir, random.choice(files))
        st.session_state["last_model"] = model_target

    if st.session_state["last_model"] != model_target:
        files = os.listdir(images_dir)
        st.session_state["random_image"] = os.path.join(images_dir, random.choice(files))
        st.session_state["last_model"] = model_target

with inf_col2:
    # st.markdown("<div style='text-align: right'>", unsafe_allow_html=True)
    if st.button("Randomize", type="primary"):
        files = os.listdir(images_dir)
        st.session_state["random_image"] = os.path.join(images_dir, random.choice(files))
    st.markdown("</div>", unsafe_allow_html=True)


    
st.image(st.session_state['random_image'], caption=f"Inference result ({model_target})")


# ================================================================================================================= #
#                                                                                                                   #
# ================================================================================================================= #

# 3. Training and Validation Loss

st.markdown("## Training and Metrics Curves", unsafe_allow_html=True)
st.markdown(f"**{model_target} Curves**")

img_col1, img_col2 = st.columns(2)

with img_col1:
    st.image(f"./{model_target.lower()}-output/{model_target.lower()}-model-v2-loss-resized.png")

with img_col2:
    st.image(f"./{model_target.lower()}-output/{model_target.lower()}-model-v2-metrics-resized.png")


