import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

st.set_page_config(page_title="MSFT RegimeLens", layout="wide")

st.title("📈 MSFT RegimeLens – Market Regime & Up/Down Analyzer")

uploaded_file = st.file_uploader("Upload your MSFT Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Expecting a column like your dataset: 'UTC Date'
    if "UTC Date" not in df.columns:
        st.error("Your file must contain a column named 'UTC Date'.")
        st.stop()

    df["Date"] = pd.to_datetime(df["UTC Date"])
    df = df.set_index("Date").sort_index()

    required_cols = {"Close", "High", "Low", "Open", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"Your file must contain these columns: {required_cols}")
        st.stop()

    df["Return"] = df["Close"].pct_change()
    df["Range"] = df["High"] - df["Low"]
    df["ReturnLag1"] = df["Return"].shift(1)
    df["VolumeLag1"] = df["Volume"].shift(1)
    df = df.dropna()

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # -------------------------
    # K-Means Regimes
    # -------------------------
    st.subheader("K-Means Market Regimes (3 clusters)")

    X_cluster = df[["Return", "Range", "Volume"]]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["RegimeCluster"] = kmeans.fit_predict(X_cluster)

    regime_summary = df.groupby("RegimeCluster").agg(
        mean_return=("Return", "mean"),
        std_return=("Return", "std"),
        mean_range=("Range", "mean"),
        mean_volume=("Volume", "mean"),
        count=("RegimeCluster", "count")
    )

    st.write(regime_summary)

    # -------------------------
    # Naive Bayes UpDay
    # -------------------------
    st.subheader("Naive Bayes: Predict UpDay (Return > 0)")

    df["UpDay"] = (df["Return"] > 0).astype(int)

    features = df[["Range", "Volume", "ReturnLag1", "VolumeLag1"]]
    target = df["UpDay"]

    # simple train/test split by time (70/30)
    split_idx = int(len(df) * 0.7)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    preds = nb.predict(X_test)

    acc = (preds == y_test.values).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Train rows", f"{len(X_train)}")
    c2.metric("Test rows", f"{len(X_test)}")
    c3.metric("Accuracy", f"{acc:.4f}")

    st.subheader("Last 10 rows (with regime + predictions)")
    df_out = df.copy()
    df_out["Pred_UpDay"] = np.nan
    df_out.loc[df_out.index[split_idx:], "Pred_UpDay"] = preds
    st.dataframe(df_out.tail(10))
