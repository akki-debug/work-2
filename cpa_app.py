import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fpdf
from scalib.preprocessing import StandardScaler
import scalib.attacks as attacks  # Import the entire module

# ==========================
# Streamlit UI Setup
# ==========================
st.title("⚡ SCALib-Based CPA Attack")
st.markdown("Perform high-performance Correlation Power Analysis using SCALib.")

# ==========================
# File Upload Section
# ==========================
st.sidebar.header("📂 Upload Dataset Files")
waveforms_file = st.sidebar.file_uploader("Upload waveforms.npy", type=["npy"])
inputs_file = st.sidebar.file_uploader("Upload inputs.npy", type=["npy"])
weights_file = st.sidebar.file_uploader("Upload weights.npy", type=["npy"])

if waveforms_file and inputs_file and weights_file:
    # Load datasets
    waveforms = np.load(waveforms_file)
    inputs = np.load(inputs_file)
    weights = np.load(weights_file)

    num_traces, num_samples = waveforms.shape
    st.sidebar.success(f"Loaded {num_traces} traces with {num_samples} samples each.")
    st.sidebar.success(f"Ground truth weights: {weights}")

    # ==========================
    # Parameter Tuning
    # ==========================
    st.sidebar.subheader("🔧 CPA Parameters")
    num_candidates = st.sidebar.slider("Number of Candidates", 100, 5000, 1000)
    apply_scaling = st.sidebar.checkbox("Apply Standard Scaling", value=True)

    # ==========================
    # Feature Scaling (Optional)
    # ==========================
    if apply_scaling:
        waveforms = StandardScaler().fit_transform(waveforms)
        st.sidebar.success("Applied standard scaling.")

    # ==========================
    # CPA Attack Using SCALib
    # ==========================
    st.subheader("🔍 Running CPA Attack with SCALib")

    candidates = np.linspace(-2, 2, num_candidates)  # Range of possible weight values
    correlation_matrix = np.zeros((2, num_candidates))

    # Attempt to fetch CPA function dynamically
    cpa_function = getattr(attacks, "cpa", None)

    if not cpa_function:
        st.error("Error: CPA function not found in SCALib. Please check your SCALib installation.")
    else:
        for weight_idx in range(2):  # Two weights to recover
            leakage_hypotheses = np.array([inputs[:, weight_idx] * w for w in candidates]).T
            cpa_result = cpa_function(waveforms, leakage_hypotheses)
            correlation_matrix[weight_idx, :] = np.max(np.abs(cpa_result), axis=0)

        # Identify the best-matching weights
        recovered_weights = [candidates[np.argmax(correlation_matrix[i])] for i in range(2)]
        st.write(f"**Recovered Weights:** {recovered_weights}")
        st.write(f"**Actual Weights:** {weights}")

        # ==========================
        # Correlation Heatmap
        # ==========================
        st.subheader("📊 CPA Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, cmap="coolwarm", xticklabels=100, yticklabels=50)
        plt.title("CPA Correlation Heatmap (SCALib)")
        plt.xlabel("Candidate Weights")
        plt.ylabel("Weight Index (0=First Weight, 1=Second Weight)")
        st.pyplot(fig)

        # ==========================
        # Downloadable Report
        # ==========================
        st.sidebar.subheader("📥 Download Report")

        def generate_pdf():
            pdf = fpdf.FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="SCALib CPA Attack Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Recovered Weights: {recovered_weights}", ln=True)
            pdf.cell(200, 10, txt=f"Actual Weights: {weights}", ln=True)
            pdf.output("scalib_cpa_report.pdf")

        generate_pdf()

        with open("scalib_cpa_report.pdf", "rb") as f:
            st.sidebar.download_button("Download Report", f, file_name="SCALib_CPA_Report.pdf", mime="application/pdf")

else:
    st.warning("Please upload all three files to run the attack.")

