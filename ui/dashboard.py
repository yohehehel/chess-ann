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
from typing import List, Optional
import threading
from queue import Queue

# Reduce TF thread contention and enable cleaner terminal logs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Pipeline utilities (robust import for script execution)
try:
    from ui.pipeline import (
        load_data,
        make_binary_labels,
        build_dataset,
        split_data,
        build_binary_model,
        evaluate_binary,
        generate_sample_predictions,
    )
except Exception:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ui.pipeline import (
        load_data,
        make_binary_labels,
        build_dataset,
        split_data,
        build_binary_model,
        evaluate_binary,
        generate_sample_predictions,
    )

# Page config
st.set_page_config(page_title="Chess Outcome Predictor", layout="wide")

st.title("Chess Game Outcome Prediction Dashboard")

# Sidebar for controls
st.sidebar.header("Controls")

model_path = "models/best_model.h5"
data_default = "data/chess_games.csv"

if 'epoch_logs' not in st.session_state:
    st.session_state.epoch_logs = []
if 'run_completed' not in st.session_state:
    st.session_state.run_completed = False
if 'latest_results' not in st.session_state:
    st.session_state.latest_results = None

# Tabs for different views
tab_run, tab_metrics, tab_preds, tab_info = st.tabs(["🚀 Run Pipeline", "📊 Training Metrics", "🎯 Predictions", "📈 Model Info"])

with tab_run:
    st.header("Run End-to-End Pipeline (Binary: White vs Black)")
    with st.form("run_form"):
        colA, colB, colC = st.columns(3)
        with colA:
            csv_path = st.text_input("Dataset CSV path", data_default)
            n_plies = st.slider("Half-moves (plies) n", 2, 40, 10, 2)
            demo_mode = st.checkbox("Demo mode (sample subset)", value=True)
            demo_percent = st.slider("Demo percentage of dataset", 1, 50, 10, 1)
        with colB:
            sample_n = st.number_input("(Optional) Sample size override", min_value=0, max_value=500000, value=0, step=1000, help="Leave 0 to use percentage above")
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
            batch_size = st.selectbox("Batch size", [32, 64, 128], index=0)
        with colC:
            epochs = st.slider("Epochs", 2, 50, 10, 1)
            early_stop = st.checkbox("Early stopping", value=True)
            patience = st.slider("Early stop patience", 1, 10, 3, 1)
            encode_batch = st.number_input("Encoding batch size", min_value=1000, max_value=50000, value=5000, step=1000)
            verbose_train = st.checkbox("Verbose training logs (terminal)", value=True)
        submitted = st.form_submit_button("Run Pipeline")

    status = st.empty()
    prog = st.progress(0)
    metrics_placeholder = st.empty()

    class StreamlitCallback:
        def __init__(self):
            self.logs: List[dict] = []

        def as_keras(self):
            from tensorflow import keras
            class _CB(keras.callbacks.Callback):
                def __init__(self, outer):
                    super().__init__()
                    self._outer = outer
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    entry = {'epoch': epoch + 1}
                    entry.update({k: float(v) for k, v in logs.items()})
                    self._outer.logs.append(entry)
                    st.session_state.epoch_logs = self._outer.logs
                    # Live chart update
                    dfm = pd.DataFrame(self._outer.logs)
                    cols = [c for c in ['accuracy', 'val_accuracy', 'loss', 'val_loss'] if c in dfm.columns]
                    if not dfm.empty and cols:
                        with metrics_placeholder.container():
                            c1, c2 = st.columns(2)
                            with c1:
                                st.subheader("Accuracy")
                                acc_df = dfm[['epoch'] + [c for c in ['accuracy', 'val_accuracy'] if c in dfm.columns]].set_index('epoch')
                                st.line_chart(acc_df)
                            with c2:
                                st.subheader("Loss")
                                loss_df = dfm[['epoch'] + [c for c in ['loss', 'val_loss'] if c in dfm.columns]].set_index('epoch')
                                st.line_chart(loss_df)
            return _CB(self)

    if submitted:
        try:
            st.session_state.epoch_logs = []
            st.session_state.run_completed = False
            st.session_state.latest_results = None

            status.info("Step 1/6: Loading data...")
            prog.progress(5)
            # Load full file, then sample by percentage for demo
            df = load_data(csv_path, sample_n=None)
            if demo_mode:
                if int(sample_n) > 0:
                    df = df.sample(n=int(sample_n), random_state=42).reset_index(drop=True)
                else:
                    frac = max(1, int(demo_percent)) / 100.0
                    df = df.sample(frac=frac, random_state=42).reset_index(drop=True)

            status.info("Step 2/6: Preparing binary labels (dropping draws)...")
            prog.progress(15)
            df_bin = make_binary_labels(df, drop_draws=True)

            status.info("Step 3/6: Reconstructing boards and encoding features...")
            prog.progress(40)
            from ui.pipeline import build_dataset_batched
            X, y = build_dataset_batched(df_bin, n_plies=n_plies, batch_size=int(encode_batch))

            status.info("Step 4/6: Train/test split...")
            prog.progress(55)
            X_train, X_test, y_train, y_test, df_train, df_test = split_data(X, y, df_bin, test_size=float(test_size))

            status.info("Step 5/6: Building and training model...")
            prog.progress(60)
            model = build_binary_model(input_dim=X.shape[1])

            # Terminal logs for debugging
            print("[Dashboard] Model build complete.", flush=True)
            print(f"[Dashboard] X shape: {X.shape}, y shape: {y.shape}", flush=True)
            print(f"[Dashboard] Train size: {len(y_train)}, Test size: {len(y_test)}", flush=True)
            try:
                # Ensure TF does not try to use GPU/Metal to avoid hangs on some macOS setups
                from tensorflow import config as tf_config
                try:
                    tf_config.set_visible_devices([], 'GPU')
                except Exception:
                    pass

                def _safe_list_tf_devices(timeout: float = 5.0) -> Optional[List[str]]:
                    """List TensorFlow devices without risking a hard hang.

                    On some macOS/Metal setups, `list_local_devices` can block indefinitely
                    which would freeze the Streamlit pipeline run. We execute the call in a
                    daemon thread and skip reporting if it exceeds a short timeout.
                    """

                    try:
                        from tensorflow.python.client import device_lib  # type: ignore
                    except Exception as exc:  # pragma: no cover - import failure already safe
                        print(f"[TF] Unable to import device_lib: {exc}", flush=True)
                        return None

                    result: Queue = Queue()

                    def _worker():
                        try:
                            result.put([d.name for d in device_lib.list_local_devices()])
                        except Exception as err:
                            result.put(err)

                    thread = threading.Thread(target=_worker, daemon=True)
                    thread.start()
                    thread.join(timeout)
                    if thread.is_alive():
                        print("[TF] Skipping device enumeration due to timeout.", flush=True)
                        return None

                    if result.empty():
                        return None

                    devices = result.get()
                    if isinstance(devices, Exception):
                        print(f"[TF] Device enumeration failed: {devices}", flush=True)
                        return None
                    return devices

                devices = _safe_list_tf_devices()
                if devices:
                    print(f"[TF] Devices: {devices}", flush=True)
                model.summary(print_fn=lambda x: print(f"[Model] {x}", flush=True))
            except Exception:
                pass

            from tensorflow import keras
            callbacks = []
            live_cb = StreamlitCallback().as_keras()
            callbacks.append(live_cb)
            if early_stop:
                callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(patience), restore_best_weights=True))

            history = model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=int(epochs),
                batch_size=int(batch_size),
                verbose=2 if verbose_train else 0,
                callbacks=callbacks,
            )

            status.info("Step 6/6: Evaluating on test set and preparing examples...")
            prog.progress(90)
            eval_res = evaluate_binary(model, X_test, y_test)
            samples = generate_sample_predictions(df_test, n_plies=n_plies, model=model, max_samples=5)

            st.session_state.run_completed = True
            st.session_state.latest_results = {
                'eval': eval_res,
                'samples': samples,
            }

            prog.progress(100)
            status.success("Pipeline completed.")

        except Exception as e:
            status.error(f"Error during pipeline run: {e}")

with tab_metrics:
    st.header("Training Progress")
    if st.session_state.epoch_logs:
        dfm = pd.DataFrame(st.session_state.epoch_logs)
        c1, c2, c3, c4 = st.columns(4)
        if 'accuracy' in dfm.columns:
            c1.metric("Final Train Acc", f"{dfm['accuracy'].iloc[-1]:.2%}")
        if 'val_accuracy' in dfm.columns:
            c2.metric("Final Val Acc", f"{dfm['val_accuracy'].iloc[-1]:.2%}")
        if 'loss' in dfm.columns:
            c3.metric("Final Train Loss", f"{dfm['loss'].iloc[-1]:.4f}")
        if 'val_loss' in dfm.columns:
            c4.metric("Final Val Loss", f"{dfm['val_loss'].iloc[-1]:.4f}")

        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Accuracy")
            cols = ['epoch'] + [c for c in ['accuracy', 'val_accuracy'] if c in dfm.columns]
            st.line_chart(dfm[cols].set_index('epoch'))
        with c6:
            st.subheader("Loss")
            cols = ['epoch'] + [c for c in ['loss', 'val_loss'] if c in dfm.columns]
            st.line_chart(dfm[cols].set_index('epoch'))
    else:
        st.info("Run the pipeline to see live training metrics.")

with tab_preds:
    st.header("Predictions & Examples")
    if st.session_state.latest_results:
        eval_res = st.session_state.latest_results['eval']
        samples = st.session_state.latest_results['samples']

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Test Accuracy", f"{eval_res['accuracy']:.2%}")
        with c2:
            cm = eval_res['confusion_matrix']
            cm_df = pd.DataFrame(cm, index=["Black", "White"], columns=["Black", "White"])  # rows: actual, cols: predicted
            st.write("Confusion Matrix (Actual vs Predicted)")
            st.dataframe(cm_df)

        st.subheader("Sample Test Predictions")
        for i, sample in enumerate(samples):
            with st.expander(f"Sample {i+1}: {'✅ Correct' if sample['correct'] else '❌ Incorrect'}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f'<div>{sample["svg"]}</div>', unsafe_allow_html=True)
                with col2:
                    st.write(f"**Predicted:** {sample['predicted']} ({sample['confidence']:.1%})")
                    st.write(f"**Actual:** {sample['actual']}")
                    st.write(f"**White Elo:** {sample['white_elo']} | **Black Elo:** {sample['black_elo']}")
    else:
        st.info("Run the pipeline to view predictions and examples.")

with tab_info:
    st.header("Model Information")
    st.subheader("Training Configuration")
    st.write("Binary classification: 1 = White win, 0 = Black win")
    st.write("Loss: Binary Crossentropy; Metric: Accuracy; Optimizer: Adam")
    st.write("Architecture: Dense(256, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)")

