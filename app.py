import streamlit as st
import torch
import numpy as np
from model_utils import FeatureExtractor, load_model
import os

# Page configuration
st.set_page_config(
    page_title="🎭 Sentiment Analysis Tool",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load feature extractor
    feature_extractor = FeatureExtractor(device=device)

    # Load trained model
    model_path = "models/feature_attention_model.pth"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = load_model(model_path, device=device)

    return feature_extractor, model, device


def predict_sentiment(text, feature_extractor, model, device):
    """Make sentiment prediction"""
    try:
        # Extract features
        bert_feat, lex_feat, syn_feat = feature_extractor.extract_all_features(text)

        # Convert to tensors
        bert_tensor = torch.tensor(bert_feat, dtype=torch.float).unsqueeze(0).to(device)
        lex_tensor = torch.tensor(lex_feat, dtype=torch.float).unsqueeze(0).to(device)
        syn_tensor = torch.tensor(syn_feat, dtype=torch.float).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(bert_tensor, lex_tensor, syn_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence, bert_feat, lex_feat, syn_feat  # ← Include bert_feat

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None, None  # ← Add extra None



def main():
    # Title and description
    st.title("🎭 Sentiment Analysis Tool")
    st.markdown("""
    This tool uses a **Feature Attention Fusion Model** that combines:
    - **BERT embeddings** for semantic understanding
    - **Lexical features** (VADER sentiment scores)
    - **Syntactic features** (linguistic patterns)
    """)

    # Initialize models
    feature_extractor, model, device = initialize_models()

    # Sidebar information
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"**Device**: {device}")
        st.info("**Model**: Feature Attention Fusion")
        st.info("**Features**: BERT + Lexical + Syntactic")

        # Model performance metrics
        st.subheader("🎯 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Amazon", "89.48%", "accuracy")
            st.metric("IMDB", "76.48%", "accuracy")
        with col2:
            st.metric("Education", "88.63%", "accuracy")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("✍️ Enter Your Review")
        user_input = st.text_area(
            "Type your review here:",
            placeholder="Enter your review to analyze its sentiment...",
            height=150,
            help="The model works best with complete sentences and natural language."
        )

        # Analysis button
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    with col2:
        st.subheader("📈 Quick Stats")
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            st.metric("Words", word_count)
            st.metric("Characters", char_count)

    # Analysis results
    if analyze_button and user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            predicted_class, confidence, bert_feat, lex_feat, syn_feat = predict_sentiment(
                user_input, feature_extractor, model, device
            )

            if predicted_class is not None:
                # Results display
                st.markdown("---")
                st.subheader("🎯 Analysis Results")

                # Main result
                sentiment = "Positive 😊" if predicted_class == 1 else "Negative 😔"
                color = "green" if predicted_class == 1 else "red"

                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.markdown(f"### Sentiment: :{color}[{sentiment}]")

                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")

                with col3:
                    # Confidence indicator
                    if confidence > 0.8:
                        conf_text = "Very High 🟢"
                    elif confidence > 0.6:
                        conf_text = "High 🟡"
                    else:
                        conf_text = "Moderate 🟠"
                    st.markdown(f"**Reliability**: {conf_text}")

                # Progress bar for confidence
                st.progress(confidence)

                # Feature breakdown
                with st.expander("🔍 Feature Analysis", expanded=False):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**🤖 BERT Embeddings:**")
                        st.write(f"• Dimensions: {len(bert_feat)}")
                        st.write(f"• Mean: {np.mean(bert_feat):.3f}")
                        st.write(f"• Std: {np.std(bert_feat):.3f}")
                        st.write(f"• Max: {np.max(bert_feat):.3f}")
                        st.write(f"• Min: {np.min(bert_feat):.3f}")

                    with col2:
                        st.markdown("**📊 VADER Sentiment Scores:**")
                        st.write(f"• Positive: {lex_feat[0]:.3f}")
                        st.write(f"• Neutral: {lex_feat[1]:.3f}")
                        st.write(f"• Negative: {lex_feat[2]:.3f}")
                        st.write(f"• Compound: {lex_feat[3]:.3f}")

                    with col3:
                        st.markdown("**🔤 Syntactic Features:**")
                        st.write(f"• Negations: {int(syn_feat[0])}")
                        st.write(f"• Nouns: {int(syn_feat[1])}")
                        st.write(f"• Verbs: {int(syn_feat[2])}")

                # Additional insights
                st.markdown("---")
                st.subheader("💡 Insights")

                insights = []
                if lex_feat[0] > 0.5:
                    insights.append("✅ Strong positive language detected")
                if lex_feat[2] > 0.5:
                    insights.append("❌ Strong negative language detected")
                if syn_feat[0] > 0:
                    insights.append("🔄 Negation patterns found - may indicate contrast")
                if confidence < 0.6:
                    insights.append("⚠️ Mixed signals - review may contain both positive and negative elements")

                if insights:
                    for insight in insights:
                        st.write(insight)
                else:
                    st.write("📝 Standard sentiment patterns detected")

    elif analyze_button:
        st.warning("⚠️ Please enter a review to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Powered by Feature Attention Fusion Model | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
