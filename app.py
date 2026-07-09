import pickle
import pandas as pd
import streamlit as st

# ------------------ PAGE CONFIG ------------------
# This must be the first Streamlit command
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Loads the machine learning model, caching it to prevent reloading on every run."""
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        st.error("🚨 Error: 'model.pkl' not found. Please ensure the model file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred while loading the model: {e}")
        st.stop()

def main():
    model = load_model()

    # ------------------ HEADER ------------------
    st.title("🏠 California Housing Price Predictor")
    st.markdown("""
    Predict the **Median House Value** in California using a
    **Random Forest Regressor** trained on the California Housing Dataset.
    """)
    st.divider()

    # ------------------ SIDEBAR ------------------
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        **Dataset:** California Housing Dataset  
        **Model:** Random Forest Regressor  
        **Library:** Scikit-learn
        """)
        st.info("Enter all house details in the main form and click **Predict Price**.")

    # ------------------ FORM ------------------
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            MedInc = st.number_input(
                "Median Income",
                min_value=0.0,
                value=5.0,
                step=0.5,
                help="Median income in block group (in tens of thousands of US Dollars, e.g., 5.0 = $50,000)"
            )

            AveRooms = st.number_input(
                "Average Rooms",
                min_value=0.0,
                value=5.0,
                step=0.5,
                help="Average number of rooms per household"
            )

            Population = st.number_input(
                "Population",
                min_value=0.0,
                value=1000.0,
                step=10.0,
                help="Block group population"
            )

            Latitude = st.number_input(
                "Latitude",
                value=34.05,
                step=0.1,
                help="Block group latitude"
            )

        with col2:
            HouseAge = st.number_input(
                "House Age",
                min_value=0.0,
                value=20.0,
                step=1.0,
                help="Median house age in block group"
            )

            AveBedrms = st.number_input(
                "Average Bedrooms",
                min_value=0.0,
                value=1.0,
                step=0.5,
                help="Average number of bedrooms per household"
            )

            AveOccup = st.number_input(
                "Average Occupancy",
                min_value=0.0,
                value=3.0,
                step=1.0,
                help="Average number of household members"
            )

            Longitude = st.number_input(
                "Longitude",
                value=-118.25,
                step=0.1,
                help="Block group longitude"
            )

        # Submit button
        submitted = st.form_submit_button("🔮 Predict Price", type="primary")

    # ------------------ PREDICTION ------------------
    if submitted:
        # Show a spinner while the model calculates (great for UX, even if prediction is fast)
        with st.spinner("Calculating price..."):
            new_house = pd.DataFrame({
                "MedInc": [MedInc],
                "HouseAge": [HouseAge],
                "AveRooms": [AveRooms],
                "AveBedrms": [AveBedrms],
                "Population": [Population],
                "AveOccup": [AveOccup],
                "Latitude": [Latitude],
                "Longitude": [Longitude]
            })

            # Predict
            prediction = model.predict(new_house)
            
            # The original target variable in the California dataset is expressed in hundreds of thousands
            predicted_price = prediction[0] * 100000

        # Results display
        st.success("Prediction Successful!")
        
        # Using columns to center/highlight the metric
        metric_col1, metric_col2, metric_col3 = st.columns([1, 2, 1])
        with metric_col2:
            st.metric(
                label="🏡 Estimated Median House Value",
                value=f"${predicted_price:,.2f}"
            )

        st.divider()

        with st.expander("📋 View Input Data Summary"):
            st.dataframe(new_house, use_container_width=True, hide_index=True)

    # ------------------ FOOTER ------------------
    st.divider()
    st.caption("Built with ❤️ using Streamlit and Scikit-learn")


if __name__ == "__main__":
    main()