
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Cloud ECG Analysis",
    page_icon="💓",
    layout="wide"
)

st.title("Cloud-Based ECG Signal Analysis")
st.write(
    "Upload ECG signal data or explore a simulated example. "
    "This educational application is not a diagnostic tool."
)

st.sidebar.header("Data Options")
choice = st.sidebar.radio(
    "Choose your data source",
    ["Simulated example", "Upload CSV"]
)

signal = None
source_name = ""

if choice == "Simulated example":
    st.info(
        "This is a simulated waveform for demonstrating the app, "
        "not a real patient's ECG."
    )
    t = np.linspace(0, 10, 2000)
    signal = (
        0.15 * np.sin(2 * np.pi * 1.2 * t)
        + 0.8 * np.exp(
            -((t % 0.83 - 0.12) ** 2) / 0.0015
        )
    )
    source_name = "Simulated example"

else:
    uploaded_file = st.file_uploader(
        "Upload a CSV file containing ECG values",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if df.empty or df.select_dtypes(
            include=np.number
        ).empty:
            st.error(
                "The CSV must contain at least one numeric column."
            )
            st.stop()

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        selected_column = st.selectbox(
            "Select the column containing ECG measurements",
            numeric_columns
        )

        signal = df[selected_column].dropna().to_numpy(
            dtype=float
        )
        source_name = uploaded_file.name

if signal is not None:
    if len(signal) < 2:
        st.error("Please provide at least two numeric measurements.")
        st.stop()

    if not np.isfinite(signal).all():
        st.error("The signal contains invalid numeric values.")
        st.stop()

    st.subheader("Analysis Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Number of Samples", f"{len(signal):,}")
    col2.metric("Mean Signal Value", f"{np.mean(signal):.4f}")
    col3.metric("Signal Standard Deviation", f"{np.std(signal):.4f}")

    st.subheader("ECG Signal Waveform")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(signal)), signal, linewidth=0.8)
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Signal Amplitude")
    ax.set_title("ECG Signal")
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Signal Summary")
    st.write(f"**Data source:** {source_name}")
    st.write(f"**Minimum value:** {np.min(signal):.4f}")
    st.write(f"**Maximum value:** {np.max(signal):.4f}")

    st.download_button(
        "Download Analysis Summary",
        data=(
            f"Data source: {source_name}\n"
            f"Samples: {len(signal)}\n"
            f"Mean: {np.mean(signal):.4f}\n"
            f"Standard deviation: {np.std(signal):.4f}\n"
            f"Minimum: {np.min(signal):.4f}\n"
            f"Maximum: {np.max(signal):.4f}\n"
        ),
        file_name="ecg_analysis_summary.txt",
        mime="text/plain"
    )
else:
    st.info("Choose the simulated example or upload a CSV to begin.")
