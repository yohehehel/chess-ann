"""
Streamlit Dashboard for Chess Outcome Prediction
Run with: streamlit run ui/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import chess
import chess.svg
import pickle
import os
from pathlib import Path

# Page config
st.set_page_config(page_title="Chess Outcome Predictor", layout="wide")

st.title("Chess Game Outcome Prediction Dashboard")

# Sidebar for controls
st.sidebar.header("Controls")

# Check if model and data exist
model_path = "models/best_model.h5"
data_path = "data"

if not os.path.exists(model_path):
    st.warning("⚠️ Model not found. Please train the model first (Step 11).")
    st.stop()

# Load training history if available
@st.cache_data
def load_history():
    """Load training history from pickle if it exists."""
    hist_path = "visuals/training_history.pkl"
    if os.path.exists(hist_path):
        with open(hist_path, 'rb') as f:
            return pickle.load(f)
    return None

history = load_history()

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Training Metrics", "🎯 Predictions", "📈 Model Info"])

with tab1:
    st.header("Training Progress")
    
    if history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy")
            acc_df = pd.DataFrame({
                'Epoch': range(1, len(history['accuracy']) + 1),
                'Train': history['accuracy'],
                'Validation': history['val_accuracy']
            })
            st.line_chart(acc_df.set_index('Epoch'))
        
        with col2:
            st.subheader("Loss")
            loss_df = pd.DataFrame({
                'Epoch': range(1, len(history['loss']) + 1),
                'Train': history['loss'],
                'Validation': history['val_loss']
            })
            st.line_chart(loss_df.set_index('Epoch'))
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Train Accuracy", f"{history['accuracy'][-1]:.2%}")
        with col2:
            st.metric("Final Val Accuracy", f"{history['val_accuracy'][-1]:.2%}")
        with col3:
            st.metric("Final Train Loss", f"{history['loss'][-1]:.4f}")
        with col4:
            st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
    else:
        st.info("Training history not found. Run training with history saving enabled.")

with tab2:
    st.header("Interactive Predictions")
    
    # Load model
    @st.cache_resource
    def load_model():
        from tensorflow import keras
        return keras.models.load_model(model_path)
    
    try:
        model = load_model()
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Prediction interface
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Input Parameters")
        white_elo = st.slider("White Elo", 1000, 3000, 1800)
        black_elo = st.slider("Black Elo", 1000, 3000, 1750)
        
        # FEN input or move sequence
        input_method = st.radio("Input method", ["FEN", "Move Sequence"])
        
        if input_method == "FEN":
            fen_input = st.text_input("FEN string", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        else:
            moves_input = st.text_area("Moves (SAN)", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4")
    
    with col1:
        st.subheader("Chess Board & Prediction")
        
        try:
            if input_method == "FEN":
                board = chess.Board(fen_input)
            else:
                board = chess.Board()
                # Simple move parsing (would need full parser in production)
                moves = moves_input.split()
                for move_str in moves[:10]:  # Limit to 10 moves
                    try:
                        board.push_san(move_str)
                    except:
                        break
            
            # Display board
            svg = chess.svg.board(board=board)
            st.markdown(f'<div style="text-align: center;">{svg}</div>', unsafe_allow_html=True)
            
            # Encode and predict
            # This would need the encode_board_features function imported
            st.info("💡 Prediction feature requires encoding functions - integrate from notebook")
            
        except Exception as e:
            st.error(f"Error processing board: {e}")
    
    # Sample predictions from test set
    st.subheader("Sample Test Predictions")
    
    if st.button("Generate New Samples"):
        st.cache_data.clear()
    
    # Load sample predictions if available
    samples_path = "visuals/sample_predictions.pkl"
    if os.path.exists(samples_path):
        with open(samples_path, 'rb') as f:
            samples = pickle.load(f)
        
        for i, sample in enumerate(samples[:5]):
            with st.expander(f"Sample {i+1}: {'✅ Correct' if sample['correct'] else '❌ Incorrect'}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f'<div>{sample["svg"]}</div>', unsafe_allow_html=True)
                with col2:
                    st.write(f"**Predicted:** {sample['predicted']} ({sample['confidence']:.1%})")
                    st.write(f"**Actual:** {sample['actual']}")
                    st.write(f"**White Elo:** {sample['white_elo']} | **Black Elo:** {sample['black_elo']}")

with tab3:
    st.header("Model Information")
    
    # Model architecture summary
    st.subheader("Architecture")
    st.info("Load model and display summary here")
    
    # Training configuration
    st.subheader("Training Configuration")
    config = {
        "Input Features": 770,
        "Output Classes": 2,
        "Hidden Layers": "256 → 128",
        "Optimizer": "Adam",
        "Loss Function": "Sparse Categorical Crossentropy"
    }
    
    for key, value in config.items():
        st.write(f"**{key}:** {value}")

