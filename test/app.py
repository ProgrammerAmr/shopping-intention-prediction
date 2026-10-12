import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Online Purchase Predictor",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Online Shopping Purchase Predictor")
st.write(
    "Enter a website session's characteristics. "
    "The trained pipeline returns the estimated probability "
    "that the session ends with a purchase."
)

MODEL_FILES = {
    "XGBoost": "xgboost.joblib",
    "Random Forest": "random_forest.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)


model_name = st.sidebar.selectbox(
    "Model",
    list(MODEL_FILES.keys()),
)

artifact = load_model(MODEL_FILES[model_name])
model = artifact["pipeline"]
threshold = artifact["threshold"]

st.sidebar.info(
    f"Decision threshold used by this model: {threshold:.3f}"
)

st.subheader("Session features")

col1, col2, col3 = st.columns(3)

with col1:
    administrative = st.number_input(
        "Administrative pages",
        min_value=0,
        value=0,
        step=1,
    )
    administrative_duration = st.number_input(
        "Administrative duration (seconds)",
        min_value=0.0,
        value=0.0,
    )
    informational = st.number_input(
        "Informational pages",
        min_value=0,
        value=0,
        step=1,
    )
    informational_duration = st.number_input(
        "Informational duration (seconds)",
        min_value=0.0,
        value=0.0,
    )
    product_related = st.number_input(
        "Product-related pages",
        min_value=0,
        value=1,
        step=1,
    )
    product_related_duration = st.number_input(
        "Product-related duration (seconds)",
        min_value=0.0,
        value=0.0,
    )

with col2:
    bounce_rates = st.number_input(
        "Bounce rate",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
    )
    exit_rates = st.number_input(
        "Exit rate",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
    )
    page_values = st.number_input(
        "Page value",
        min_value=0.0,
        value=0.0,
    )
    special_day = st.selectbox(
        "Special day",
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    month = st.selectbox(
        "Month",
        ["Feb", "Mar", "May", "June", "Jul", "Aug",
         "Sep", "Oct", "Nov", "Dec"],
    )
    visitor_type = st.selectbox(
        "Visitor type",
        ["New_Visitor", "Returning_Visitor", "Other"],
    )

with col3:
    operating_systems = st.number_input(
        "Operating system code",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
    )
    browser = st.number_input(
        "Browser code",
        min_value=1,
        max_value=13,
        value=2,
        step=1,
    )
    region = st.number_input(
        "Region code",
        min_value=1,
        max_value=9,
        value=1,
        step=1,
    )
    traffic_type = st.number_input(
        "Traffic type code",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )
    weekend = st.checkbox("Weekend")

if st.button("Predict purchase intention", type="primary"):

    input_df = pd.DataFrame(
        [{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_systems,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }]
    )

    probability = model.predict_proba(input_df)[0, 1]
    prediction = int(probability >= threshold)

    st.divider()

    if prediction == 1:
        st.success("Predicted outcome: Purchase")
    else:
        st.info("Predicted outcome: No purchase")

    st.metric(
        "Estimated purchase probability",
        f"{probability:.1%}",
    )

    st.caption(
        "This probability is a model output, not a guarantee of purchase."
    )
