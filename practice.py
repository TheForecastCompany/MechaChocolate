import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import r2_score
import statsmodels.api as sm

st.title("🍫 ChocoLuxe 52-Week Sales Forecast")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("chocoluxe_weekly_data.csv", parse_dates=["Date"], index_col="Date")
    df = df.asfreq("W-SUN")
    return df

df = load_data()
y = df["Weekly_Sales"]
X = df[["Marketing_Spend", "Avg_Temp", "Holiday"]]

st.write("📈 Weekly Sales Data", df.tail())

# Sidebar inputs
st.sidebar.header("Model Settings")

use_auto = st.sidebar.checkbox("🔁 Use auto_arima", value=True)

if not use_auto:
    p = st.sidebar.slider("AR (p)", 0, 3, 1)
    d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
    q = st.sidebar.slider("MA (q)", 0, 3, 1)

# Future exogenous variable inputs
st.sidebar.header("Forecast Inputs")
future_spend = st.sidebar.slider("Marketing Spend ($)", 0, 1000, 500)
future_temp = st.sidebar.slider("Avg Temp (°F)", 20, 100, 45)

# Create future exogenous values
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(weeks=1), periods=52, freq="W-SUN")
future_X = pd.DataFrame({
    "Marketing_Spend": [future_spend]*52,
    "Avg_Temp": [future_temp]*52,
    "Holiday": [0]*6 + [1] + [0]*10 + [1] + [0]*10 + [1] + [0]*23
}, index=future_dates)

# Fit model
with st.spinner("Training model..."):
    if use_auto:
        auto_model = auto_arima(y, exogenous=X, seasonal=False, trace=False,
                                error_action='ignore', suppress_warnings=True,
                                max_p=3, max_q=3, stepwise=True)
        order = auto_model.order
        st.success(f"Best ARIMA order from auto_arima: {order}")
        model = SARIMAX(y, exog=X, order=order)
    else:
        model = SARIMAX(y, exog=X, order=(p, d, q))

    model_fit = model.fit(disp=False)

# Metrics on training data
# Predict in-sample with exog
y_pred_in_sample = model_fit.predict(start=y.index[0], end=y.index[-1], exog=X)
r2 = r2_score(y, y_pred_in_sample)
# For p-value, get the p-value of the overall model or of coefficients?
# We'll show p-values of coefficients as a table
coef_pvalues = model_fit.pvalues

st.subheader("📊 Model Diagnostics")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**R² (In-sample):** {r2:.4f}")
    st.markdown(f"**AIC:** {model_fit.aic:.2f}")
    st.markdown(f"**BIC:** {model_fit.bic:.2f}")

with col2:
    st.markdown("**Coefficient p-values:**")
    st.write(coef_pvalues)

# Forecast next 52 weeks with confidence intervals
forecast_res = model_fit.get_forecast(steps=52, exog=future_X)
forecast = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

# Combine and plot
full_series = pd.concat([y, forecast])
full_series.name = "Weekly Sales"

st.subheader("🔮 Forecast: Next 52 Weeks")
fig, ax = plt.subplots(figsize=(10, 5))
full_series.plot(ax=ax, label="Actual + Forecast")
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label="95% CI")
ax.axvline(df.index[-1], color="gray", linestyle="--", label="Forecast Start")
ax.set_title("Actual and Forecasted Weekly Sales")
ax.legend()
st.pyplot(fig)
