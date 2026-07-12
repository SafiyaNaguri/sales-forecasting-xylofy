"""
Intelligent Sales Forecasting Dashboard
Superstore Sales Dataset - Streamlit App
Author: Safiya Naguri
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


# ------------------------------------------------------------------
# Data loading & caching
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


def make_features(series):
    d = pd.DataFrame({"y": series})
    d["lag1"] = d["y"].shift(1)
    d["lag2"] = d["y"].shift(2)
    d["lag3"] = d["y"].shift(3)
    d["rolling_mean3"] = d["y"].shift(1).rolling(3).mean()
    d["month"] = d.index.month
    d["quarter"] = d.index.quarter
    return d


@st.cache_data
def forecast_xgb(monthly_series, steps=3):
    feat = make_features(monthly_series).dropna()
    X, y = feat.drop(columns="y"), feat["y"]
    model = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X, y)

    history = monthly_series.copy()
    preds = []
    for _ in range(steps):
        next_date = history.index[-1] + pd.DateOffset(months=1)
        row = pd.DataFrame(
            {
                "lag1": [history.iloc[-1]],
                "lag2": [history.iloc[-2]],
                "lag3": [history.iloc[-3]],
                "rolling_mean3": [history.iloc[-3:].mean()],
                "month": [next_date.month],
                "quarter": [next_date.quarter],
            }
        )
        p = model.predict(row)[0]
        preds.append(p)
        history.loc[next_date] = p

    # In-sample MAE/RMSE (fit vs actual on training data) as a rough accuracy indicator
    in_sample_pred = model.predict(X)
    mae = np.mean(np.abs(y.values - in_sample_pred))
    rmse = np.sqrt(np.mean((y.values - in_sample_pred) ** 2))
    return preds, history.index[-steps:], mae, rmse


df = load_data()

st.title("📊 Intelligent Sales Forecasting Dashboard")
st.caption("Superstore Sales — Forecasting, Anomaly Detection & Product Segmentation")

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ------------------------------------------------------------------
# PAGE 1 - Sales Overview
# ------------------------------------------------------------------
if page == "Sales Overview":
    st.header("Sales Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Sales by Year")
        yearly = df.groupby("Year")["Sales"].sum()
        fig, ax = plt.subplots()
        ax.bar(yearly.index.astype(str), yearly.values, color="#2E86AB")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = df.set_index("Order Date").resample("MS")["Sales"].sum()
        fig, ax = plt.subplots()
        ax.plot(monthly.index, monthly.values, color="#A23B72")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    st.subheader("Sales by Region and Category")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        regions = st.multiselect(
            "Filter by Region", options=sorted(df["Region"].unique()), default=sorted(df["Region"].unique())
        )
    with filter_col2:
        categories = st.multiselect(
            "Filter by Category", options=sorted(df["Category"].unique()), default=sorted(df["Category"].unique())
        )

    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]
    pivot = filtered.groupby(["Region", "Category"])["Sales"].sum().unstack("Category").fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Sales ($)")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.dataframe(pivot.style.format("{:.0f}"))

# ------------------------------------------------------------------
# PAGE 2 - Forecast Explorer
# ------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.header("Forecast Explorer")
    st.write("Forecasts generated using the XGBoost model (best performer from the model comparison in the notebook).")

    dim_type = st.selectbox("Forecast by:", ["Category", "Region"])
    if dim_type == "Category":
        options = sorted(df["Category"].unique())
    else:
        options = sorted(df["Region"].unique())

    selected = st.selectbox(f"Select {dim_type}", options)
    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    if dim_type == "Category":
        seg_df = df[df["Category"] == selected]
    else:
        seg_df = df[df["Region"] == selected]

    seg_monthly = seg_df.set_index("Order Date").resample("MS")["Sales"].sum()

    with st.spinner("Training model and generating forecast..."):
        preds, future_idx, mae, rmse = forecast_xgb(seg_monthly, steps=3)

    preds = preds[:horizon]
    future_idx = future_idx[:horizon]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(seg_monthly.index, seg_monthly.values, label="Actual", color="black")
    ax.plot(future_idx, preds, "o--", label="Forecast", color="crimson")
    ax.axvline(seg_monthly.index[-1], color="gray", linestyle=":")
    ax.legend()
    ax.set_title(f"{selected} — {horizon}-Month Forecast")
    ax.set_ylabel("Sales ($)")
    st.pyplot(fig)

    forecast_table = pd.DataFrame({"Month": future_idx.strftime("%b %Y"), "Forecasted Sales": np.round(preds, 0)})
    st.table(forecast_table)

    st.markdown(f"**Model accuracy (in-sample):** MAE = `{mae:,.0f}`  |  RMSE = `{rmse:,.0f}`")
    st.caption(
        "Note: MAE/RMSE shown here are in-sample fit metrics for this specific segment's model. "
        "See the notebook's Task 3 for out-of-sample holdout metrics on the aggregate series."
    )

# ------------------------------------------------------------------
# PAGE 3 - Anomaly Report
# ------------------------------------------------------------------
elif page == "Anomaly Report":
    st.header("Anomaly Report")

    weekly = df.set_index("Order Date").resample("W")["Sales"].sum()
    weekly_df = weekly.to_frame("Sales")

    iso = IsolationForest(contamination=0.06, random_state=42)
    weekly_df["iso_anomaly"] = iso.fit_predict(weekly_df[["Sales"]])
    iso_anomalies = weekly_df[weekly_df["iso_anomaly"] == -1]

    roll_mean = weekly.rolling(8, min_periods=4).mean()
    roll_std = weekly.rolling(8, min_periods=4).std()
    z_scores = (weekly - roll_mean) / roll_std
    z_anomalies = weekly[np.abs(z_scores) > 2]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly.index, weekly.values, label="Weekly Sales", color="steelblue")
    ax.scatter(iso_anomalies.index, iso_anomalies["Sales"], color="red", s=60, zorder=5, label="Isolation Forest")
    ax.scatter(z_anomalies.index, z_anomalies.values, color="orange", marker="x", s=80, zorder=5, label="Z-Score")
    ax.legend()
    ax.set_title("Weekly Sales Anomalies")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Isolation Forest Anomalies")
        st.table(iso_anomalies["Sales"].reset_index().rename(columns={"Order Date": "Week", "Sales": "Sales ($)"}))
    with col2:
        st.subheader("Z-Score Anomalies")
        st.table(z_anomalies.reset_index().rename(columns={"Order Date": "Week", 0: "Sales ($)", "Sales": "Sales ($)"}))

# ------------------------------------------------------------------
# PAGE 4 - Product Demand Segments
# ------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.header("Product Demand Segments")

    subcat_rows = []
    for subcat, g in df.groupby("Sub-Category"):
        seg_monthly = g.set_index("Order Date").resample("MS")["Sales"].sum()
        total_sales = g["Sales"].sum()
        yearly = g.groupby("Year")["Sales"].sum()
        growth_rate = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100 if len(yearly) >= 2 else 0
        volatility = seg_monthly.std()
        avg_order_value = g["Sales"].mean()
        subcat_rows.append([subcat, total_sales, growth_rate, volatility, avg_order_value])

    feat_df = pd.DataFrame(
        subcat_rows, columns=["SubCategory", "TotalSales", "GrowthRate", "Volatility", "AvgOrderValue"]
    ).set_index("SubCategory")

    X_scaled = StandardScaler().fit_transform(feat_df.values)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat_df["Cluster"] = km.fit_predict(X_scaled)

    cluster_labels = {
        0: "High Volume, Stable Demand",
        1: "Growing Demand, High Volatility",
        2: "Low Volume, Stable Demand",
        3: "Declining Demand, High Volatility",
    }
    feat_df["ClusterLabel"] = feat_df["Cluster"].map(cluster_labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    feat_df["PC1"], feat_df["PC2"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7))
    for c in sorted(feat_df["Cluster"].unique()):
        sub = feat_df[feat_df["Cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], s=110, label=cluster_labels[c])
    for name, row in feat_df.iterrows():
        ax.annotate(name, (row["PC1"], row["PC2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-Categories by Demand Cluster")
    display_df = feat_df[["TotalSales", "GrowthRate", "Volatility", "ClusterLabel"]].sort_values("ClusterLabel")
    st.dataframe(display_df.style.format({"TotalSales": "{:.0f}", "GrowthRate": "{:.1f}%", "Volatility": "{:.0f}"}))
