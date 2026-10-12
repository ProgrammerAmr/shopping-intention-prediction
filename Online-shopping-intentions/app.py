import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Online Shopping Intention",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    return joblib.load("trained_models.pkl")


try:
    models = load_models()
except FileNotFoundError:
    st.error(
        "trained_models.pkl was not found. "
        "Train and save your models from the notebook first."
    )
    st.stop()


# ============================================================
# UI → MODEL MAPPINGS
# ============================================================
# IMPORTANT:
# The labels below are UI/demo labels.
# The numbers are the categorical codes expected by the dataset.
#
# The original dataset uses numeric codes for several categorical
# features. These labels are only there to make the GUI easier
# for a human to understand.
# ============================================================

OPERATING_SYSTEMS = {
    "Windows": 1,
    "macOS": 2,
    "Linux": 3,
    "Android": 4,
    "iOS": 5,
    "ChromeOS": 6,
    "Other": 7,
    "Other / Unknown": 8,
}

BROWSERS = {
    "Chrome": 1,
    "Firefox": 2,
    "Safari": 3,
    "Edge": 4,
    "Opera": 5,
    "Internet Explorer": 6,
    "Other": 7,
    "Other / Unknown": 8,
    "Other / Unknown ": 9,
    "Other / Unknown 2": 10,
    "Other / Unknown 3": 11,
    "Other / Unknown 4": 12,
    "Other / Unknown 5": 13,
}

REGIONS = {
    "Middle East": 1,
    "North Aerica": 2,
    "south America": 3,
    "West Europ": 4,
    "East Europ": 5,
    "south Asia": 6,
    "East Asia": 7,
    "Austrilia": 8,
    "Africa": 9,
}

TRAFFIC_TYPES = {
    "Direct / Campaign Type 1": 1,
    "Facebook": 2,
    "Instagram": 3,
    "Tiktook": 4,
    "Traffic Source 5": 5,
    "Traffic Source 6": 6,
    "Traffic Source 7": 7,
    "Traffic Source 8": 8,
    "Traffic Source 9": 9,
    "Traffic Source 10": 10,
    "Traffic Source 11": 11,
    "Traffic Source 12": 12,
    "Traffic Source 13": 13,
    "Traffic Source 14": 14,
    "Traffic Source 15": 15,
    "Traffic Source 16": 16,
    "Traffic Source 17": 17,
    "Traffic Source 18": 18,
    "Traffic Source 19": 19,
    "Traffic Source 20": 20,
}

MONTHS = {
    "January": "Jan",
    "February": "Feb",
    "March": "Mar",
    "April": "Apr",
    "May": "May",
    "June": "June",
    "July": "Jul",
    "August": "Aug",
    "September": "Sep",
    "October": "Oct",
    "November": "Nov",
    "December": "Dec",
}

VISITOR_TYPES = {
    "New Visitor": "New_Visitor",
    "Returning Visitor": "Returning_Visitor",
    "Other": "Other",
}


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🛒 Online Shopping Intention Predictor")
st.write(
    "Predict whether an online shopping session is likely to end "
    "with a purchase using different machine learning models."
)

# ============================================================
# MODEL SELECTION
# ============================================================

st.subheader("🤖 Choose a Machine Learning Model")

model_display_names = list(models.keys())

# Make the notebook's "Xgboost" look nicer in the GUI.
display_to_model = {
    "Random Forest": "Random Forest",
    "XGBoost": "Xgboost",
    "Decision Tree": "Decision Tree",
    "SVM" : "SVM" ,
    "KNN": "KNN" , 
    "Logistic Regression": "Logistic Regression"
}

selected_display_name = st.selectbox(
    "Model",
    list(display_to_model.keys())
)

selected_model_name = display_to_model[selected_display_name]
selected_model = models[selected_model_name]

st.caption(
    f"Currently selected: **{selected_display_name}**"
)


# ============================================================
# BASIC VISITOR INFORMATION
# ============================================================

st.subheader("👤 Visitor Information")

col1, col2, col3 = st.columns(3)

with col1:
    visitor_type_label = st.selectbox(
        "Visitor Type",
        list(VISITOR_TYPES.keys()),
        help="Whether this is a new or returning visitor."
    )

with col2:
    month_label = st.selectbox(
        "Visit Month",
        list(MONTHS.keys()),
        index=10,
        help="Month in which the shopping session occurred."
    )

with col3:
    weekend_label = st.selectbox(
        "Weekend Visit",
        ["No", "Yes"],
        help="Was the session during a weekend?"
    )


# ============================================================
# SHOPPING ACTIVITY
# ============================================================

st.subheader("🛍️ Shopping Activity")

col1, col2, col3 = st.columns(3)

with col1:
    administrative = st.number_input(
        "Administrative Pages",
        min_value=0,
        value=1,
        step=1,
        help="Number of administrative pages visited."
    )

with col2:
    informational = st.number_input(
        "Informational Pages",
        min_value=0,
        value=1,
        step=1,
        help="Number of informational pages visited."
    )

with col3:
    product_related = st.number_input(
        "Product Pages",
        min_value=0,
        value=10,
        step=1,
        help="Number of product-related pages visited."
    )

col1, col2, col3 = st.columns(3)

with col1:
    administrative_duration = st.number_input(
        "Administrative Time",
        min_value=0.0,
        value=20.0,
        step=1.0,
        help="Time spent on administrative pages."
    )

with col2:
    informational_duration = st.number_input(
        "Informational Time",
        min_value=0.0,
        value=20.0,
        step=1.0,
        help="Time spent on informational pages."
    )

with col3:
    product_related_duration = st.number_input(
        "Product Browsing Time",
        min_value=0.0,
        value=300.0,
        step=10.0,
        help="Time spent on product-related pages."
    )


# ============================================================
# ADVANCED FEATURES
# ============================================================

with st.expander("⚙️ Advanced Session Details"):
    st.info(
        "These features are kept under Details because they are less "
        "natural for a normal user to enter."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        bounce_rate = st.number_input(
            "Bounce Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            format="%.3f",
            help="Rate associated with sessions leaving after limited interaction."
        )

    with col2:
        exit_rate = st.number_input(
            "Exit Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Rate associated with exits from pages during sessions."
        )

    with col3:
        page_value = st.number_input(
            "Page Value",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Estimated value associated with the pages visited."
        )

    special_day = st.slider(
        "Special Day Proximity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Proximity of the visit to a special day such as a holiday."
    )


# ============================================================
# TECHNICAL / CATEGORICAL DETAILS
# ============================================================

with st.expander("💻 Technical Session Details"):
    st.info(
        "These values are categorical codes in the original dataset. "
        "The GUI gives them human-friendly demo labels, then converts "
        "them back to the codes expected by the trained model."
    )

    col1, col2 = st.columns(2)

    with col1:
        os_label = st.selectbox(
            "Operating System",
            list(OPERATING_SYSTEMS.keys())
        )

        browser_label = st.selectbox(
            "Browser",
            list(BROWSERS.keys())
        )

        region_label = st.selectbox(
            "Region",
            list(REGIONS.keys())
        )

    with col2:
        traffic_label = st.selectbox(
            "Traffic Type",
            list(TRAFFIC_TYPES.keys())
        )

        st.write("")

        st.caption(
            "Note: OS and browser names are educational UI labels. "
            "The original dataset stores categorical IDs."
        )


# ============================================================
# BUILD MODEL INPUT
# ============================================================

input_data = pd.DataFrame({
    "SpecialDay": [special_day],
    "PageValues": [page_value],
    "ExitRates": [exit_rate],
    "BounceRates": [bounce_rate],
    "ProductRelated_Duration": [product_related_duration],
    "ProductRelated": [product_related],
    "Informational_Duration": [informational_duration],
    "Informational": [informational],
    "Administrative_Duration": [administrative_duration],
    "Administrative": [administrative],

    "Weekend": [weekend_label == "Yes"],
    "Month": [MONTHS[month_label]],
    "OperatingSystems": [OPERATING_SYSTEMS[os_label]],
    "Browser": [BROWSERS[browser_label]],
    "Region": [REGIONS[region_label]],
    "TrafficType": [TRAFFIC_TYPES[traffic_label]],
    "VisitorType": [VISITOR_TYPES[visitor_type_label]],
})


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Purchase Intention",
    type="primary",
    use_container_width=True
)

if predict_button:

    try:
        prediction = selected_model.predict(input_data)[0]

        # All classifiers used in the notebook support predict_proba.
        probabilities = selected_model.predict_proba(input_data)[0]
        purchase_probability = probabilities[1]

        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.success("🛍️ Purchase Intention: Positive")
            else:
                st.info("👀 Purchase Intention: Negative")

        with result_col2:
            st.metric(
                "Purchase Probability",
                f"{purchase_probability:.1%}"
            )

        st.progress(float(purchase_probability))

        st.caption(
            f"Prediction generated by **{selected_display_name}**."
        )

    except Exception as e:
        st.error(
            "Prediction failed. Check that the saved model pipeline "
            "was trained with the same feature names and data format "
            "used by this application."
        )

        with st.expander("Technical error"):
            st.exception(e)


# ============================================================
# OPTIONAL DEBUG VIEW
# ============================================================

with st.expander("🔍 View Model Input (for learning/debugging)"):
    st.dataframe(input_data, use_container_width=True)
