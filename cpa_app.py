import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ==========================
# Streamlit UI Setup
# ==========================
st.title("🔍 Correlation Power Analysis (CPA) Attack")
st.markdown("Recover neural network weights from side-channel traces.")

# ==========================
# File Upload Section
# ==========================
st.sidebar.header("Upload Dataset Files")
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
    # Answer to Question 1
    # ==========================
    st.subheader("Question 1: Number of Possible Weight Values")
    st.write("""
        Each weight is stored in IEEE 754 32-bit format, meaning there are \(2^{32}\) (around 4.29 billion) 
        possible values. This is computationally infeasible for brute-force attacks.
    """)

    # ==========================
    # CPA Attack Implementation
    # ==========================
    def float_to_bin(f):
        """Convert IEEE 754 float to binary string."""
        return format(np.frombuffer(np.float32(f).tobytes(), dtype=np.uint32)[0], '032b')

    def hamming_weight(n):
        """Compute the Hamming weight (number of 1s in binary representation)."""
        return bin(n).count("1")

    # Answer to Question 2
    st.subheader("Question 2: Creating Smaller Chunks")
    st.write("""
        Instead of brute-force searching all 32-bit values, we use smaller mantissa chunks while keeping 
        the exponent fixed. This reduces search space and makes the attack feasible.
    """)

    # Weight candidates
    num_candidates = 1000
    candidates = np.linspace(-2, 2, num_candidates)

    # CPA computation
    correlation_matrix = np.zeros((2, num_candidates))

    for weight_idx in range(2):  # Two weights to recover
        for i, w in enumerate(candidates):
            hw_model = np.array([hamming_weight(int(float_to_bin(x * w), 2)) for x in inputs[:, weight_idx]])
            correlations = [pearsonr(hw_model, waveforms[:, sample_idx])[0] for sample_idx in range(num_samples)]
            correlation_matrix[weight_idx, i] = max(correlations)

    recovered_weights = [candidates[np.argmax(correlation_matrix[i])] for i in range(2)]

    # ==========================
    # Answer to Question 3
    # ==========================
    st.subheader("Question 3: Using Chunks in CPA")
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
    # Correlation Plot
    # ==========================
    st.subheader("📊 Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(candidates, correlation_matrix[0], label="Weight 1 Correlation")
    ax.plot(candidates, correlation_matrix[1], label="Weight 2 Correlation")
    ax.set_xlabel("Weight Candidates")
    ax.set_ylabel("Max Correlation")
    ax.set_title("Correlation Power Analysis (CPA) Results")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

else:
    st.warning("Please upload all three files to run the attack.")
