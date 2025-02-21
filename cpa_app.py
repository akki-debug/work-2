import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Tuple, List
import base64
from io import BytesIO

class CPADemo:
    def __init__(self):
        self.setup_ui()
        
    def setup_ui(self):
        """Configure the Streamlit interface"""
        st.title("⚡ Advanced Side-Channel Attack Demo")
        st.markdown("""
            Learn how attackers recover secret information from power traces using Correlation Power Analysis (CPA).
            
            <style>
                .stProgress > div > div > div {
                    background-color: #ff4d4d;
                }
            </style>
        """, unsafe_allow_html=True)
        
        self.simulation_settings()
    
    def simulation_settings(self):
        """Configure simulation parameters with advanced options"""
        st.sidebar.header("🔧 Advanced Simulation Settings")
        
        # Basic settings
        self.num_traces = st.sidebar.slider(
            "Number of Traces", 
            min_value=100, 
            max_value=10000, 
            value=2000,
            help="More traces generally improves attack success rate"
        )
        
        self.num_samples = st.sidebar.slider(
            "Samples per Trace",
            min_value=10,
            max_value=500,
            value=50,
            help="Higher resolution timing measurements"
        )
        
        # Advanced noise settings
        self.noise_type = st.sidebar.selectbox(
            "Noise Type",
            ["Gaussian", "Uniform", "Mixed"],
            index=0,
            help="Different noise models simulate various hardware conditions"
        )
        
        self.signal_to_noise_ratio = st.sidebar.slider(
            "Signal-to-Noise Ratio (dB)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Lower SNR makes the attack more challenging"
        )
        
        # Attack configuration
        self.attack_method = st.sidebar.selectbox(
            "Attack Method",
            ["Classic CPA", "Normalized CPA", "Zero-Mean CPA"],
            index=0,
            help="Different CPA variants offer varying robustness"
        )

    def generate_power_traces(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simulated power traces with configurable noise"""
        inputs = np.random.randint(1, 100, size=self.num_traces)
        self.secret_multiplier = st.session_state.get('secret_multiplier', 
                                                    np.random.randint(1, 10))
        
        # Base power consumption model
        base_consumption = np.array([
            [bin(inputs[i] * self.secret_multiplier).count("1") 
             for _ in range(self.num_samples)]
            for i in range(self.num_traces)
        ])
        
        # Add noise based on selected type
        if self.noise_type == "Gaussian":
            noise = np.random.normal(
                loc=0,
                scale=10 ** (-self.signal_to_noise_ratio / 20),
                size=base_consumption.shape
            )
        elif self.noise_type == "Uniform":
            noise = np.random.uniform(
                low=-10 ** (-self.signal_to_noise_ratio / 20),
                high=10 ** (-self.signal_to_noise_ratio / 20),
                size=base_consumption.shape
            )
        else:  # Mixed noise
            gaussian = np.random.normal(
                loc=0,
                scale=10 ** (-self.signal_to_noise_ratio / 20),
                size=base_consumption.shape
            )
            uniform = np.random.uniform(
                low=-10 ** (-self.signal_to_noise_ratio / 20),
                high=10 ** (-self.signal_to_noise_ratio / 20),
                size=base_consumption.shape
            )
            noise = gaussian + uniform
            
        power_traces = base_consumption + noise
        
        # Store session state
        st.session_state['secret_multiplier'] = self.secret_multiplier
        st.session_state['inputs'] = inputs
        
        return power_traces, inputs
    
    def process_attack(self, power_traces: np.ndarray, inputs: np.ndarray) -> dict:
        """Execute CPA attack with selected method"""
        candidates = range(1, 11)
        correlations = []
        
        for candidate in candidates:
            # Calculate power model
            hw_model = np.array([bin(inputs[i] * candidate).count("1") 
                               for i in range(len(inputs))])
            
            # Apply preprocessing based on attack method
            if self.attack_method == "Normalized CPA":
                hw_model = (hw_model - np.mean(hw_model)) / np.std(hw_model)
                processed_traces = (power_traces - np.mean(power_traces, axis=0)) / \
                                 np.std(power_traces, axis=0)
            elif self.attack_method == "Zero-Mean CPA":
                hw_model = hw_model - np.mean(hw_model)
                processed_traces = power_traces - np.mean(power_traces, axis=0)
            else:  # Classic CPA
                processed_traces = power_traces
                
            # Calculate correlations
            corrs = [
                pearsonr(hw_model, processed_traces[:, sample])[0]
                for sample in range(self.num_samples)
            ]
            correlations.append(max(abs(np.array(corrs))))
        
        return {
            'correlations': correlations,
            'best_guess': candidates[np.argmax(correlations)],
            'success_rate': int((np.argmax(correlations) == self.secret_multiplier - 1) * 100)
        }
    
    def plot_correlations(self, correlations: List[float]):
        """Create enhanced visualization of attack results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Main correlation plot
        x = range(1, 11)
        ax1.plot(x, correlations, marker='o', linestyle='-', linewidth=2)
        ax1.axvline(x=self.secret_multiplier, color='red', linestyle='--',
                   label=f'Secret ({self.secret_multiplier})')
        ax1.set_xlabel("Multiplier Candidates")
        ax1.set_ylabel("Maximum Absolute Correlation")
        ax1.set_title("CPA Attack Results")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Success probability distribution
        probabilities = np.array(correlations) / sum(correlations)
        ax2.bar(x, probabilities, alpha=0.7)
        ax2.set_xlabel("Multiplier Candidates")
        ax2.set_ylabel("Probability")
        ax2.set_title("Success Probability Distribution")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def run_demo(self):
        """Main demo execution flow"""
        if st.button("Run Attack"):
            with st.spinner("Running CPA Attack..."):
                power_traces, inputs = self.generate_power_traces()
                results = self.process_attack(power_traces, inputs)
                
                # Display results
                st.subheader("🎯 Attack Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Recovered Value", 
                             value=str(results['best_guess']))
                with col2:
                    st.metric(label="Success Rate", 
                             value=f"{results['success_rate']}%")
                
                # Show correlation plots
                st.subheader("📊 Correlation Analysis")
                fig = self.plot_correlations(results['correlations'])
                st.pyplot(fig)
                
                # Additional insights
                st.subheader("🔍 Technical Details")
                st.code({
                    "Classic CPA": """
                        Uses raw Pearson correlation coefficient
                        Best for clean signals with high SNR
                    """,
                    "Normalized CPA": """
                        Normalizes both traces and model
                        More robust against amplitude variations
                    """,
                    "Zero-Mean CPA": """
                        Removes DC offset from signals
                        Effective against constant power bias
                    """
                }[self.attack_method])

if __name__ == "__main__":
    demo = CPADemo()
    demo.run_demo()
