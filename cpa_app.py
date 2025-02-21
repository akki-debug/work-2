import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ==========================
# Streamlit UI
# ==========================
st.title("⚡ Side-Channel Attack Demo")
st.markdown("Learn how attackers recover secret information from power traces!")

# ==========================
# Simulating Power Traces
# ==========================
st.sidebar.subheader("🔧 Simulation Settings")
num_traces = st.sidebar.slider("Number of Traces", 100, 5000, 1000)
num_samples = st.sidebar.slider("Samples per Trace", 10, 100, 25)
secret_multiplier = st.sidebar.slider("Secret Multiplier", 1, 10, 3)

# Generate random inputs
inputs = np.random.randint(1, 10, size=num_traces)

# Generate power traces (Simulated Power = Hamming Weight of Result)
power_traces = np.array([[bin(inputs[i] * secret_multiplier).count("1") + np.random.normal(0, 0.5) 
                          for _ in range(num_samples)] for i in range(num_traces)])

st.sidebar.success(f"Generated {num_traces} power traces with {num_samples} samples each.")

# ==========================
# CPA Attack
# ==========================
st.subheader("🔍 Correlation Power Analysis (CPA) Attack")

# Test different multipliers to find the secret one
candidates = range(1, 11)  # Possible multiplier values
correlations = []

for candidate in candidates:
    hw_model = np.array([bin(inputs[i] * candidate).count("1") for i in range(num_traces)])
    max_corr = max([pearsonr(hw_model, power_traces[:, sample])[0] for sample in range(num_samples)])
    correlations.append(max_corr)

# Identify the recovered secret multiplier
recovered_multiplier = candidates[np.argmax(correlations)]
st.write(f"**Recovered Multiplier:** {recovered_multiplier}")
st.write(f"**Actual Secret Multiplier:** {secret_multiplier}")

# ==========================
# Plot Correlation
# ==========================
st.subheader("📊 CPA Correlation Plot")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(candidates, correlations, marker='o', linestyle='-')
ax.set_xlabel("Multiplier Candidates")
ax.set_ylabel("Max Correlation")
ax.set_title("CPA Attack - Correlation vs. Multiplier")
st.pyplot(fig)
