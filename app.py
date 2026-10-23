from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CardioCloud | ECG Analysis",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- LOAD SEPARATE CSS FILE ----------------
CSS_PATH = Path(__file__).parent / "styles.css"
if CSS_PATH.exists():
    st.markdown(
        f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )
else:
    st.warning("styles.css was not found. The app will run without custom styling.")


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 💓 CardioCloud")
    st.caption("CLOUD ECG ANALYSIS")
    st.divider()

    page = st.radio("NAVIGATION", ["Dashboard", "About Project"])

    st.divider()
    st.markdown("### Data source")
    source = st.radio("Choose input", ["Demo ECG Signal", "Upload ECG CSV"])

    uploaded_file = None
    if source == "Upload ECG CSV":
        uploaded_file = st.file_uploader(
            "Upload your signal file",
            type=["csv"],
        )

        # MIT-BIH Arrhythmia Database commonly uses 360 Hz.
        # The exact rate of a converted Kaggle CSV should be verified.
        sampling_rate_input = st.number_input(
            "Sampling rate (Hz)",
            min_value=1,
            max_value=5000,
            value=360,
            step=1,
            help=(
                "Enter the sampling rate documented by your dataset. "
                "360 Hz is common for the original MIT-BIH Arrhythmia Database, "
                "but converted datasets may differ."
            ),
        )
    else:
        sampling_rate_input = 250

    st.markdown("---")
    st.caption("Educational demonstration only")


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">ECG Analysis Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Visualize electrocardiogram signals and explore basic signal statistics.</div>',
    unsafe_allow_html=True,
)


# ---------------- ABOUT PAGE ----------------
if page == "About Project":
    st.subheader("About CardioCloud")
    st.write(
        "CardioCloud is an educational cloud application that demonstrates "
        "ECG signal visualization and basic numerical analysis."
    )
    st.markdown("### Technology stack")
    st.markdown(
        """
        - **Python:** application logic
        - **Streamlit:** user interface
        - **NumPy:** numerical calculations
        - **Pandas:** CSV data handling
        - **Matplotlib:** signal visualization
        - **Streamlit Community Cloud:** planned hosting platform
        """
    )
    st.info(
        "This application does not diagnose heart conditions. "
        "The demo waveform is simulated and is not patient data."
    )
    st.stop()


# ---------------- LOAD SIGNAL ----------------
signal = None
source_name = ""
selected_column = ""
sampling_rate = float(sampling_rate_input)

if source == "Demo ECG Signal":
    sampling_rate = 250.0
    t = np.arange(0, 10, 1 / sampling_rate)

    # Simulated waveform for UI demonstration only
    signal = 0.025 * np.sin(2 * np.pi * 1.2 * t)

    for beat in np.arange(0.5, 10, 0.83):
        signal += 0.12 * np.exp(-((t - (beat - 0.18)) / 0.045) ** 2)
        signal -= 0.08 * np.exp(-((t - (beat - 0.035)) / 0.012) ** 2)
        signal += 0.85 * np.exp(-((t - beat) / 0.014) ** 2)
        signal -= 0.20 * np.exp(-((t - (beat + 0.035)) / 0.018) ** 2)
        signal += 0.20 * np.exp(-((t - (beat + 0.22)) / 0.065) ** 2)

    source_name = "Simulated demo"

elif uploaded_file is not None:
    try:
        # Streamlit may rerun the app; read the uploaded file from its start.
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        with st.expander("CSV diagnostics", expanded=False):
            st.write("CSV shape:", df.shape)
            st.write("Column names:", df.columns.tolist())
            st.write("First five rows:")
            st.dataframe(df.head())
            st.write("Data types:")
            st.write(df.dtypes.astype(str))

        # Convert candidate columns to numbers even if the CSV stores values
        # as text. This also handles numeric columns with occasional bad cells.
        numeric_data = {}
        for col in df.columns:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= 2:
                numeric_data[col] = converted

        excluded_names = {
            "sample", "sample #", "sample number", "sample_number",
            "sample no", "sample_no", "index", "unnamed: 0", "time",
            "timestamp", "row", "row number", "row_number",
        }

        signal_columns = []
        automatically_excluded = []
        for col, values in numeric_data.items():
            normalized_name = " ".join(str(col).strip().lower().split())
            clean = values.dropna()

            # Exclude common row/sample counters by name.
            if normalized_name in excluded_names:
                automatically_excluded.append(str(col))
                continue

            # Also catch a counter with an unexpected header, such as 0, 1, 2...
            # This prevents accidentally plotting a steadily increasing index.
            arr = clean.to_numpy(dtype=float)
            if len(arr) >= 3 and np.all(np.diff(arr) > 0):
                steps = np.diff(arr)
                is_integer_counter = np.allclose(arr, np.round(arr))
                regular_steps = np.allclose(steps, steps[0])
                starts_like_index = np.isclose(arr[0], 0) or np.isclose(arr[0], 1)
                if is_integer_counter and regular_steps and starts_like_index:
                    automatically_excluded.append(str(col))
                    continue

            signal_columns.append(col)

        if not signal_columns:
            st.error(
                "No likely ECG signal columns were found. Check that the CSV "
                "contains a sample-by-sample signal column such as MLII or V1."
            )
            st.stop()

        if automatically_excluded:
            st.caption(
                "Automatically ignored possible sample/index columns: "
                + ", ".join(automatically_excluded)
            )

        # Prefer MLII, then V1, for common MIT-BIH CSV exports.
        default_index = 0
        for preferred in ("MLII", "V1"):
            match = next(
                (i for i, col in enumerate(signal_columns)
                 if str(col).strip().upper() == preferred),
                None,
            )
            if match is not None:
                default_index = match
                break

        selected_column = st.selectbox(
            "Select the ECG signal column",
            options=signal_columns,
            index=default_index,
            help="Choose the column containing ECG amplitude values, for example MLII or V1. Do not choose a sample counter.",
        )

        signal_series = numeric_data[selected_column]
        # Keep only valid numeric samples and remove infinities. The graph uses
        # the cleaned samples so invalid CSV cells do not break the chart.
        signal_series = signal_series.replace([np.inf, -np.inf], np.nan).dropna()
        signal = signal_series.to_numpy(dtype=float)
        source_name = uploaded_file.name

        with st.expander("Selected signal diagnostics", expanded=False):
            st.write("Selected signal column:", selected_column)
            st.write("Number of valid samples:", len(signal))
            if len(signal):
                st.write("First 10 values:", signal[:10].tolist())
                st.write("Minimum:", float(np.min(signal)))
                st.write("Maximum:", float(np.max(signal)))
                st.write("Standard deviation:", float(np.std(signal)))

        if len(signal) > 0 and np.std(signal) == 0:
            st.warning(
                "The selected column contains the same value in every valid row, "
                "so it will appear as a flat line. Choose another signal column "
                "or verify the CSV export."
            )

    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")
        st.stop()


# ---------------- ANALYSIS VALIDATION ----------------
if signal is None:
    st.info("Choose Demo ECG Signal or upload a CSV in the sidebar.")
    st.stop()

if len(signal) < 2:
    st.error("The signal must contain at least two valid numeric values.")
    st.stop()

if not np.isfinite(signal).all():
    st.error("The selected signal contains invalid numeric values.")
    st.stop()


# ---------------- STATUS ----------------
st.markdown(
    '<div class="notice">● &nbsp; Signal data loaded successfully</div>',
    unsafe_allow_html=True,
)


# ---------------- METRICS ----------------
st.markdown(
    '<div class="section-title">Signal Overview</div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Samples", f"{len(signal):,}")
c2.metric("Mean Amplitude", f"{np.mean(signal):.4f}")
c3.metric("Minimum", f"{np.min(signal):.4f}")
c4.metric("Maximum", f"{np.max(signal):.4f}")


# ---------------- ECG CHART ----------------
st.markdown(
    '<div class="section-title">ECG Waveform</div>',
    unsafe_allow_html=True,
)
with st.container(border=True):
    st.caption(f"Current data: {source_name}")

    fig, ax = plt.subplots(figsize=(13, 4))
    time_axis = np.arange(len(signal), dtype=float) / sampling_rate

    # Very large CSVs can contain hundreds of thousands of points. Plot a
    # representative subset for responsiveness while keeping all samples for
    # metrics and the downloadable summary.
    max_plot_points = 20000
    if len(signal) > max_plot_points:
        plot_indices = np.linspace(
            0, len(signal) - 1, num=max_plot_points, dtype=int
        )
        plot_time = time_axis[plot_indices]
        plot_signal = signal[plot_indices]
        st.caption(
            f"Showing {len(plot_signal):,} evenly spaced points for display; "
            f"statistics use all {len(signal):,} valid samples."
        )
    else:
        plot_time = time_axis
        plot_signal = signal

    ax.set_facecolor("#FFFDFD")
    ax.plot(plot_time, plot_signal, color="#0D9488", linewidth=0.8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (input units)")
    ax.set_title("ECG Signal Visualization")
    ax.grid(True, color="#E5BFC4", linewidth=0.5, alpha=0.8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------- DETAILS ----------------
st.markdown(
    '<div class="section-title">Signal Details</div>',
    unsafe_allow_html=True,
)
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("Dataset Information")
        st.markdown(f"**Data source:** {source_name}")
        if selected_column:
            st.markdown(f"**Selected column:** {selected_column}")
        st.markdown(f"**Total samples:** {len(signal):,}")
        st.markdown(f"**Mean amplitude:** {np.mean(signal):.4f}")
        st.markdown(f"**Standard deviation:** {np.std(signal):.4f}")

with right:
    with st.container(border=True):
        st.subheader("Signal Range")
        st.markdown(f"**Minimum amplitude:** {np.min(signal):.4f}")
        st.markdown(f"**Maximum amplitude:** {np.max(signal):.4f}")
        st.markdown(f"**Sampling rate used:** {sampling_rate:g} Hz")
        st.markdown(
            f"**Displayed duration:** {len(signal) / sampling_rate:.2f} seconds"
        )


# ---------------- DOWNLOAD ----------------
summary = (
    "CardioCloud ECG Signal Summary\n"
    "Educational use only - not a medical diagnosis\n\n"
    f"Source: {source_name}\n"
    f"Selected column: {selected_column or 'Simulated demo'}\n"
    f"Samples: {len(signal)}\n"
    f"Sampling rate used (Hz): {sampling_rate:g}\n"
    f"Mean amplitude: {np.mean(signal):.6f}\n"
    f"Standard deviation: {np.std(signal):.6f}\n"
    f"Minimum amplitude: {np.min(signal):.6f}\n"
    f"Maximum amplitude: {np.max(signal):.6f}\n"
)
st.download_button(
    label="⬇ Download Analysis Summary",
    data=summary,
    file_name="ecg_analysis_summary.txt",
    mime="text/plain",
)


# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">CardioCloud • Cloud Applications Project • Educational use only</div>',
    unsafe_allow_html=True,
)
