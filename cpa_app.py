import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import fpdf

# ==========================
# Streamlit UI Setup
# ==========================
st.title("🔍 Correlation Power Analysis (CPA) Attack")
st.markdown("Recover neural network weights from side-channel traces.")

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
    min_weight = st.sidebar.slider("Min Weight", -5.0, 0.0, -2.0)
    max_weight = st.sidebar.slider("Max Weight", 0.0, 5.0, 2.0)
    num_candidates = st.sidebar.slider("Number of Candidates", 100, 5000, 1000)
    leakage_model = st.sidebar.radio("Select Leakage Model", ["Hamming Weight", "Hamming Distance"])

    # ==========================
    # IEEE-754 Float Decoder
    # ==========================
    def float_to_bin(f):
        """Convert IEEE 754 float to binary string."""
        return format(np.frombuffer(np.float32(f).tobytes(), dtype=np.uint32)[0], '032b')

    def hamming_weight(n):
        """Compute the Hamming weight (number of 1s in binary representation)."""
        return bin(n).count("1")

    def hamming_distance(n1, n2):
        """Compute the Hamming distance between two numbers."""
        return bin(n1 ^ n2).count("1")

    st.sidebar.subheader("🔍 IEEE 754 Float Decoder")
    user_float = st.sidebar.number_input("Enter a Float", value=0.0, step=0.01)
    st.sidebar.text(f"IEEE 754 Binary: {float_to_bin(user_float)}")

    # ==========================
    # Answer to Questions
    # ==========================
    st.subheader("📖 Question 1: Number of Possible Weight Values")
    st.write("""
        Each weight is stored in IEEE 754 32-bit format, meaning there are \(2^{32}\) (around 4.29 billion) 
        possible values. This is computationally infeasible for brute-force attacks.
    """)

    st.subheader("📖 Question 2: Creating Smaller Chunks")
    st.write("""
        Instead of brute-force searching all 32-bit values, we use smaller mantissa chunks while keeping 
        the exponent fixed. This reduces search space and makes the attack feasible.
    """)

    # ==========================
    # Noise Reduction (Signal Filtering)
    # ==========================
    st.sidebar.subheader("🛠 Noise Reduction")
    apply_filter = st.sidebar.checkbox("Apply Savitzky-Golay Filter", value=True)

    if apply_filter:
        waveforms = savgol_filter(waveforms, window_length=11, polyorder=2, axis=1)
        st.sidebar.success("Applied noise filtering.")

    # ==========================
    # CPA Attack Implementation
    # ==========================
    candidates = np.linspace(min_weight, max_weight, num_candidates)
    correlation_matrix = np.zeros((2, num_candidates))
    
    progress_bar = st.progress(0)  # Initialize progress bar

    for weight_idx in range(2):  # Two weights to recover
        for i, w in enumerate(candidates):
            if leakage_model == "Hamming Weight":
                hw_model = np.array([hamming_weight(int(float_to_bin(x * w), 2)) for x in inputs[:, weight_idx]])
            else:  # Hamming Distance
                hw_model = np.array([hamming_distance(int(float_to_bin(x * w), 2), int(float_to_bin(0.0))) for x in inputs[:, weight_idx]])

            correlations = [pearsonr(hw_model, waveforms[:, sample_idx])[0] for sample_idx in range(num_samples)]
            correlation_matrix[weight_idx, i] = max(correlations)
            
            progress_bar.progress((i + 1) / num_candidates)

    recovered_weights = [candidates[np.argmax(correlation_matrix[i])] for i in range(2)]

    # ==========================
    # Answer to Question 3
    # ==========================
    st.subheader("📖 Question 3: Using Chunks in CPA")
    st.write("""
        We recover weights by:
        1. Guessing small weight chunks.
        2. Computing hypothetical leakage values.
        3. Performing CPA on each chunk separately.
        4. Combining results to reconstruct the full weight.
    """)

    # ==========================
    # Display Results
    # ==========================
    st.subheader("🔑 Recovered Weights")
    st.write(f"**Weight 1:** {recovered_weights[0]}")
    st.write(f"**Weight 2:** {recovered_weights[1]}")
    st.write(f"**Actual Weights:** {weights}")

    # ==========================
    # Correlation Heatmap
    # ==========================
    st.subheader("📊 CPA Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, cmap="coolwarm", xticklabels=100, yticklabels=50)
    plt.title("Correlation Heatmap")
    plt.xlabel("Time Samples")
    plt.ylabel("Weight Candidates")
    st.pyplot(fig)

    # ==========================
    # Downloadable Report
    # ==========================
    st.sidebar.subheader("📥 Download Report")

    def generate_pdf():
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="CPA Attack Report", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Recovered Weights: {recovered_weights}", ln=True)
        pdf.cell(200, 10, txt=f"Actual Weights: {weights}", ln=True)
        pdf.output("cpa_report.pdf")

    generate_pdf()

    with open("cpa_report.pdf", "rb") as f:
        st.sidebar.download_button("Download Report", f, file_name="CPA_Report.pdf", mime="application/pdf")

else:
    st.warning("Please upload all three files to run the attack.")
