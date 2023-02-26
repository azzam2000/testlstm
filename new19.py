# Import required libraries
import pandas as pd
import numpy as np
import streamlit as st
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

# Set app title and sidebar
st.set_page_config(page_title="Sales Forecasting App", page_icon=":moneybag:", layout="wide")
st.sidebar.title("Sales Forecasting App")

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload your sales dataset (CSV format)", type="csv")

if uploaded_file is not None:
    # Read data
    sales_df = pd.read_csv(uploaded_file, parse_dates=["Date"])

    # Show data
    st.write("## Sales Dataset")
    st.write(sales_df)

    # Prepare data
    sales_df = sales_df[["Date", "Sales"]]
    sales_df = sales_df.rename(columns={"Date": "ds", "Sales": "y"})

    # Build the model
    model = Prophet()
    model.fit(sales_df)

    # Make predictions
    period = st.sidebar.selectbox("Select the period of prediction", ("Daily", "Weekly", "Monthly", "Yearly"))
    if period == "Daily":
        future = model.make_future_dataframe(periods=365, freq="D")
        forecast = model.predict(future)
        result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(365)
    elif period == "Weekly":
        future = model.make_future_dataframe(periods=52, freq="W")
        forecast = model.predict(future)
        result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(52)
    elif period == "Monthly":
        future = model.make_future_dataframe(periods=12, freq="M")
        forecast = model.predict(future)
        result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12)
    else:
        future = model.make_future_dataframe(periods=5, freq="Y")
        forecast = model.predict(future)
        result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5)

    # Show predictions
    st.write("## Sales Predictions")
    st.write(result_df)

    # Visualize results
    st.write("## Sales Forecast Plot")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

    st.write("## Sales Forecast Components Plot")
    fig_components = plot_components_plotly(model, forecast)
    st.plotly_chart(fig_components)
