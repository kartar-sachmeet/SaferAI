"""
KL Divergence Dashboard for analyzing diff-SAE features.

This dashboard allows interactive exploration of:
1. KL divergence between base and IT model outputs
2. Feature activations and their impact on KL
3. Steering experiments with specific latents
"""

import streamlit as st
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import GemmaModelPair, BatchTopKSAE
from src.utils import DiffActivationCollector


class KLDashboard:
    """Dashboard for KL divergence analysis."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the dashboard."""
        self.config_path = config_path
        self.load_config()

        # Initialize models and SAE (lazy loading)
        self.models = None
        self.sae = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_config(self):
        """Load configuration."""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    @st.cache_resource
    def load_models(_self):
        """Load Gemma models (cached)."""
        st.info("Loading models... This may take a few minutes.")
        models = GemmaModelPair(_self.config_path)
        models.load_models()
        st.success("Models loaded!")
        return models

    @st.cache_resource
    def load_sae(_self, checkpoint_path: str):
        """Load trained SAE (cached)."""
        st.info(f"Loading SAE from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=_self.device)

        sae = BatchTopKSAE.from_config(checkpoint['config'])
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae = sae.to(_self.device)
        sae.eval()

        st.success("SAE loaded!")
        return sae

    def compute_kl_divergence(
        self,
        base_logits: torch.Tensor,
        instruct_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute KL divergence between base and instruct model outputs.

        Args:
            base_logits: Logits from base model
            instruct_logits: Logits from instruct model
            temperature: Temperature for softmax

        Returns:
            KL divergence per token
        """
        # Apply temperature
        base_logits = base_logits / temperature
        instruct_logits = instruct_logits / temperature

        # Compute log probabilities
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        instruct_log_probs = F.log_softmax(instruct_logits, dim=-1)

        # Compute KL divergence (IT || base)
        # KL(P || Q) = sum(P * log(P/Q))
        instruct_probs = torch.exp(instruct_log_probs)
        kl = (instruct_probs * (instruct_log_probs - base_log_probs)).sum(dim=-1)

        return kl

    def get_model_outputs(
        self,
        text: str,
        steering_latent: Optional[int] = None,
        steering_coef: float = 1.0
    ):
        """
        Get outputs from both models, optionally with steering.

        Args:
            text: Input text
            steering_latent: Latent index to steer with (on base model)
            steering_coef: Steering coefficient

        Returns:
            Dictionary with outputs and KL divergence
        """
        # Tokenize
        inputs = self.models.tokenize([text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get base model output
            if steering_latent is not None and self.sae is not None:
                # Apply steering to base model
                base_outputs = self.get_steered_output(
                    inputs,
                    steering_latent,
                    steering_coef
                )
            else:
                base_outputs = self.models.base_model(**inputs)

            # Get instruct model output
            instruct_outputs = self.models.instruct_model(**inputs)

        # Compute KL divergence
        kl_per_token = self.compute_kl_divergence(
            base_outputs.logits[0],
            instruct_outputs.logits[0]
        )

        # Get tokens
        tokens = self.models.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        return {
            'tokens': tokens,
            'base_logits': base_outputs.logits[0],
            'instruct_logits': instruct_outputs.logits[0],
            'kl_per_token': kl_per_token,
            'total_kl': kl_per_token.sum().item()
        }

    def get_steered_output(
        self,
        inputs: dict,
        latent_idx: int,
        coef: float
    ):
        """
        Get output from base model with steering.

        Args:
            inputs: Tokenized inputs
            latent_idx: Index of latent to steer with
            coef: Steering coefficient

        Returns:
            Model outputs
        """
        # Hook to add steering vector
        steering_vector = self.sae.decoder.weight[:, latent_idx] * coef

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add steering vector to all positions
            hidden_states = hidden_states + steering_vector.view(1, 1, -1)

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        # Register hook on target layer
        target_layer = self.models.base_model.model.layers[self.config['models']['target_layer']]
        hook = target_layer.register_forward_hook(steering_hook)

        # Get output
        outputs = self.models.base_model(**inputs)

        # Remove hook
        hook.remove()

        return outputs

    def analyze_feature_impact(
        self,
        text: str,
        top_k: int = 20
    ):
        """
        Analyze which features most impact KL divergence for given text.

        Args:
            text: Input text
            top_k: Number of top features to return

        Returns:
            DataFrame with feature impacts
        """
        # Get baseline KL
        baseline = self.get_model_outputs(text)
        baseline_kl = baseline['total_kl']

        # Test steering with each active feature
        # First, get active features for this text
        inputs = self.models.tokenize([text])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract activations
        collector = DiffActivationCollector(
            self.models.base_model,
            self.models.instruct_model,
            self.models.tokenizer,
            self.config['models']['target_layer'],
            self.device
        )

        activations = collector.collect_diff_activations([text])
        diff_act = activations['diff'][0]  # First sample

        # Get SAE latents
        with torch.no_grad():
            latents = self.sae.encode(diff_act.to(self.device))

        # Find top-k active latents (averaged across sequence)
        avg_latents = latents.mean(dim=0)
        top_k_values, top_k_indices = torch.topk(avg_latents.abs(), k=top_k)

        # Test steering with each top latent
        results = []
        for idx, latent_idx in enumerate(top_k_indices):
            latent_idx = latent_idx.item()

            # Test with positive steering
            steered = self.get_model_outputs(text, latent_idx, 1.0)
            kl_reduction = baseline_kl - steered['total_kl']

            results.append({
                'latent_idx': latent_idx,
                'activation': top_k_values[idx].item(),
                'baseline_kl': baseline_kl,
                'steered_kl': steered['total_kl'],
                'kl_reduction': kl_reduction,
                'kl_reduction_pct': (kl_reduction / baseline_kl) * 100
            })

        return pd.DataFrame(results)

    def plot_kl_per_token(self, outputs: dict):
        """Plot KL divergence per token."""
        df = pd.DataFrame({
            'Token': outputs['tokens'],
            'KL Divergence': outputs['kl_per_token'].cpu().numpy(),
            'Position': range(len(outputs['tokens']))
        })

        fig = px.bar(
            df,
            x='Position',
            y='KL Divergence',
            hover_data=['Token'],
            title='KL Divergence per Token',
            labels={'Position': 'Token Position', 'KL Divergence': 'KL(IT || Base)'}
        )

        return fig

    def plot_feature_impact(self, feature_df: pd.DataFrame):
        """Plot feature impact on KL divergence."""
        fig = px.bar(
            feature_df,
            x='latent_idx',
            y='kl_reduction_pct',
            hover_data=['activation', 'baseline_kl', 'steered_kl'],
            title='Feature Impact on KL Divergence',
            labels={
                'latent_idx': 'Latent Index',
                'kl_reduction_pct': 'KL Reduction (%)'
            }
        )

        return fig


def run_dashboard():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="Diff-SAE KL Dashboard",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Diff-SAE KL Divergence Dashboard")
    st.markdown("""
    Explore how features in the diff-SAE affect the KL divergence between
    Gemma 2 base and instruction-tuned models.
    """)

    # Initialize dashboard
    dashboard = KLDashboard()

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # SAE checkpoint selection
    checkpoint_path = st.sidebar.text_input(
        "SAE Checkpoint Path",
        value="checkpoints/final_model.pt"
    )

    # Load models and SAE
    if st.sidebar.button("Load Models & SAE"):
        dashboard.models = dashboard.load_models()
        dashboard.sae = dashboard.load_sae(checkpoint_path)

    # Main content
    if dashboard.models is None or dashboard.sae is None:
        st.warning("⚠️ Please load models and SAE using the sidebar.")
        return

    # Input text
    st.header("1️⃣ Input Text")
    input_text = st.text_area(
        "Enter text to analyze:",
        value="How do I make a bomb?",
        height=100
    )

    # Analyze button
    if st.button("🔍 Analyze"):
        with st.spinner("Computing KL divergence and feature impacts..."):
            # Get baseline outputs
            outputs = dashboard.get_model_outputs(input_text)

            # Display results
            st.header("2️⃣ KL Divergence Analysis")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total KL Divergence", f"{outputs['total_kl']:.4f}")
            with col2:
                st.metric("Average KL per Token", f"{outputs['kl_per_token'].mean().item():.4f}")

            # Plot KL per token
            fig = dashboard.plot_kl_per_token(outputs)
            st.plotly_chart(fig, use_container_width=True)

            # Feature impact analysis
            st.header("3️⃣ Feature Impact Analysis")
            feature_df = dashboard.analyze_feature_impact(input_text, top_k=20)

            # Display table
            st.dataframe(
                feature_df.style.format({
                    'activation': '{:.4f}',
                    'baseline_kl': '{:.4f}',
                    'steered_kl': '{:.4f}',
                    'kl_reduction': '{:.4f}',
                    'kl_reduction_pct': '{:.2f}%'
                }),
                use_container_width=True
            )

            # Plot feature impact
            fig = dashboard.plot_feature_impact(feature_df)
            st.plotly_chart(fig, use_container_width=True)

    # Steering experiment
    st.header("4️⃣ Steering Experiment")
    st.markdown("Test how steering the base model with specific latents affects KL divergence.")

    col1, col2 = st.columns(2)
    with col1:
        steering_latent = st.number_input(
            "Latent Index",
            min_value=0,
            max_value=dashboard.config['sae']['n_latents'] - 1,
            value=0
        )
    with col2:
        steering_coef = st.slider(
            "Steering Coefficient",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1
        )

    if st.button("🎯 Test Steering"):
        with st.spinner("Computing steered outputs..."):
            baseline = dashboard.get_model_outputs(input_text)
            steered = dashboard.get_model_outputs(
                input_text,
                steering_latent=int(steering_latent),
                steering_coef=steering_coef
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Baseline KL", f"{baseline['total_kl']:.4f}")
            with col2:
                st.metric("Steered KL", f"{steered['total_kl']:.4f}")
            with col3:
                reduction = baseline['total_kl'] - steered['total_kl']
                reduction_pct = (reduction / baseline['total_kl']) * 100
                st.metric("KL Reduction", f"{reduction:.4f} ({reduction_pct:.1f}%)")


if __name__ == "__main__":
    run_dashboard()
