import os
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import chess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def load_data(csv_path: str, sample_n: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    return df


def make_binary_labels(df: pd.DataFrame, drop_draws: bool = True) -> pd.DataFrame:
    if 'Result' not in df.columns:
        raise ValueError("Expected 'Result' column in dataset")
    if drop_draws:
        df = df[df['Result'].isin(['1-0', '0-1'])].copy()
    # Map: White win -> 1, Black win -> 0
    mapping = {'1-0': 1, '0-1': 0}
    df['label'] = df['Result'].map(mapping)
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    return df


def _tokenize_san_moves(movetext: str) -> List[str]:
    if not isinstance(movetext, str):
        return []
    tokens = movetext.replace('\n', ' ').split()
    # Remove move numbers like "1.", "23..."
    return [t for t in tokens if not ('.' in t and all(ch.isdigit() or ch == '.' for ch in t))]


def get_board_after_n(moves_str: str, n_plies: int) -> chess.Board:
    board = chess.Board()
    if n_plies <= 0:
        return board
    tokens = _tokenize_san_moves(moves_str)
    plies_applied = 0
    for t in tokens:
        if plies_applied >= n_plies:
            break
        try:
            board.push_san(t)
            plies_applied += 1
        except Exception:
            break
    return board


_PIECE_TYPES = [
    (chess.PAWN, 0),
    (chess.KNIGHT, 1),
    (chess.BISHOP, 2),
    (chess.ROOK, 3),
    (chess.QUEEN, 4),
    (chess.KING, 5),
]


def encode_board(board: chess.Board, white_elo: int, black_elo: int, normalize_elo: bool = True) -> np.ndarray:
    # 12 x 64 one-hot (6 white types + 6 black types)
    features = np.zeros((12, 64), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        base_idx = 0 if piece.color == chess.WHITE else 6
        for p_type, offset in _PIECE_TYPES:
            if piece.piece_type == p_type:
                features[base_idx + offset, square] = 1.0
                break
    flat = features.reshape(-1)
    if normalize_elo:
        w = float(white_elo) / 3000.0
        b = float(black_elo) / 3000.0
    else:
        w = float(white_elo)
        b = float(black_elo)
    return np.concatenate([flat, np.array([w, b], dtype=np.float32)], axis=0)


def build_binary_model(input_dim: int):
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_dataset(
    df: pd.DataFrame,
    n_plies: int,
    moves_column: str = 'AN',
    white_elo_col: str = 'WhiteElo',
    black_elo_col: str = 'BlackElo'
) -> Tuple[np.ndarray, np.ndarray, List[chess.Board]]:
    """Build full dataset in-memory (kept for compatibility)."""
    if moves_column not in df.columns:
        raise ValueError(f"Expected moves column '{moves_column}' in dataset")
    X: List[np.ndarray] = []
    boards: List[chess.Board] = []
    for _, row in df.iterrows():
        board = get_board_after_n(row[moves_column], n_plies)
        boards.append(board)
        x = encode_board(board, int(row[white_elo_col]), int(row[black_elo_col]))
        X.append(x)
    Xnp = np.stack(X, axis=0)
    ynp = df['label'].values.astype(np.int32)
    return Xnp, ynp, boards


def build_dataset_batched(
    df: pd.DataFrame,
    n_plies: int,
    batch_size: int = 5000,
    moves_column: str = 'AN',
    white_elo_col: str = 'WhiteElo',
    black_elo_col: str = 'BlackElo'
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode the dataset in batches to limit peak memory usage.

    Returns X, y (without boards list to reduce memory).
    """
    if moves_column not in df.columns:
        raise ValueError(f"Expected moves column '{moves_column}' in dataset")
    num_rows = len(df)
    X_parts: List[np.ndarray] = []
    y_all = df['label'].values.astype(np.int32)
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        chunk = df.iloc[start:end]
        X_chunk: List[np.ndarray] = []
        for _, row in chunk.iterrows():
            board = get_board_after_n(row[moves_column], n_plies)
            x = encode_board(board, int(row[white_elo_col]), int(row[black_elo_col]))
            X_chunk.append(x)
        X_parts.append(np.stack(X_chunk, axis=0))
    X = np.concatenate(X_parts, axis=0) if len(X_parts) > 1 else X_parts[0]
    return X, y_all


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    df_source: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    idx = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=test_size, random_state=random_state, stratify=y
    )
    df_train = df_source.iloc[idx_train].reset_index(drop=True)
    df_test = df_source.iloc[idx_test].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, df_train, df_test


def evaluate_binary(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, np.ndarray]:
    preds = model.predict(X_test, verbose=0).reshape(-1)
    pred_labels = (preds >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_labels, labels=[0, 1])
    acc = (pred_labels == y_test).mean()
    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'pred_probs': preds,
        'pred_labels': pred_labels,
    }


def generate_sample_predictions(
    df_test: pd.DataFrame,
    n_plies: int,
    model,
    max_samples: int = 5
) -> List[Dict]:
    import chess.svg
    samples: List[Dict] = []
    take = min(max_samples, len(df_test))
    for i in range(take):
        row = df_test.iloc[i]
        board = get_board_after_n(row['AN'], n_plies)
        x = encode_board(board, int(row['WhiteElo']), int(row['BlackElo']))
        prob = float(model.predict(np.expand_dims(x, 0), verbose=0).reshape(-1)[0])
        pred_label = 1 if prob >= 0.5 else 0
        actual = int(row['label'])
        svg = chess.svg.board(board=board)
        samples.append({
            'svg': svg,
            'predicted': 'White win' if pred_label == 1 else 'Black win',
            'confidence': prob if pred_label == 1 else 1.0 - prob,
            'actual': 'White win' if actual == 1 else 'Black win',
            'correct': pred_label == actual,
            'white_elo': int(row['WhiteElo']),
            'black_elo': int(row['BlackElo']),
        })
    return samples


