"""
Phase 3: Model Training
========================
Architecture: Dual-input TextCNN + Handcrafted features

Pipeline:
  phase2_outputs/
      ├── X_text.npy       (N, 300)  integer sequences
      ├── X_hand.npy       (N, 7)    handcrafted features (unscaled)
      ├── y.npy            (N,)      labels  0=legit  1=malicious
      └── tokenizer.json             vocab for embedding layer

  Steps:
    1. Load phase 2 outputs
    2. Train / val / test split  (80 / 10 / 10, stratified)
    3. Scale handcrafted features (fit on train only — no leakage)
    4. Compute class weights     (penalise missed phishing more)
    5. Build dual-input CNN model
    6. Train with EarlyStopping + ReduceLROnPlateau
    7. Evaluate on held-out test set
    8. Save model + scaler
"""

import json
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers                        # type: ignore
from tensorflow.keras.models import Model                  # type: ignore
from tensorflow.keras.callbacks import (                   # type: ignore
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_DIR   = Path("phase2_outputs")
OUTPUT_DIR  = Path("phase3_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Must match Phase 2 settings exactly
MAX_VOCAB   = 20_000
MAX_LEN     = 300

# Embedding
EMBED_DIM   = 64        # each token → 64-dim learned vector

# CNN
NUM_FILTERS = 128       # filters per kernel size (128 is TextCNN standard)
KERNEL_SIZES= [3, 4, 5] # captures trigrams, 4-grams, 5-grams

# Training
BATCH_SIZE  = 64
MAX_EPOCHS  = 30        # EarlyStopping will cut this short
DROPOUT_CNN = 0.3       # after pooling (text branch)
DROPOUT_OUT = 0.4       # after fusion dense layer

# EarlyStopping
ES_PATIENCE = 4         # stop if val_loss doesn't improve for 4 epochs
LR_PATIENCE = 2         # halve LR if val_loss stagnates for 2 epochs


# ─────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────
def load_data():
    print(f"\n[1/7] Loading phase 2 outputs from '{INPUT_DIR}/' …")

    X_text = np.load(INPUT_DIR / "X_text.npy")
    X_hand = np.load(INPUT_DIR / "X_hand.npy")
    y      = np.load(INPUT_DIR / "y.npy")

    with open(INPUT_DIR / "hand_feature_names.json") as f:
        feature_names = json.load(f)

    print(f"      X_text : {X_text.shape}  dtype={X_text.dtype}")
    print(f"      X_hand : {X_hand.shape}  dtype={X_hand.dtype}")
    print(f"      y      : {y.shape}  dist={dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"      Features: {feature_names}")

    return X_text, X_hand, y, feature_names


# ─────────────────────────────────────────────
# STEP 2 — SPLIT
# ─────────────────────────────────────────────
def split_data(X_text, X_hand, y):
    """
    80 / 10 / 10 stratified split.

    Stratified = class ratio preserved in every split.
    Important here because even a small imbalance compounds
    when computing class weights and metrics.
    """
    print("\n[2/7] Splitting data (80/10/10 stratified) …")

    # First cut: 80% train, 20% temp
    (X_text_train, X_text_temp,
     X_hand_train, X_hand_temp,
     y_train,      y_temp) = train_test_split(
        X_text, X_hand, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Second cut: split temp 50/50 → 10% val, 10% test
    (X_text_val, X_text_test,
     X_hand_val, X_hand_test,
     y_val,      y_test) = train_test_split(
        X_text_temp, X_hand_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    print(f"      Train : {len(y_train):,}   "
          f"malicious={y_train.sum():,}  legit={(y_train==0).sum():,}")
    print(f"      Val   : {len(y_val):,}    "
          f"malicious={y_val.sum():,}   legit={(y_val==0).sum():,}")
    print(f"      Test  : {len(y_test):,}    "
          f"malicious={y_test.sum():,}   legit={(y_test==0).sum():,}")

    return (X_text_train, X_text_val, X_text_test,
            X_hand_train, X_hand_val, X_hand_test,
            y_train, y_val, y_test)


# ─────────────────────────────────────────────
# STEP 3 — SCALE HANDCRAFTED FEATURES
# ─────────────────────────────────────────────
def scale_features(X_hand_train, X_hand_val, X_hand_test):
    """
    StandardScaler: zero mean, unit variance.

    FIT on train only, then TRANSFORM all three splits.
    Fitting on the full dataset would leak test statistics
    into training → artificially inflated metrics.

    After scaling:
        num_words=300  → ~1.2  (a few std deviations above mean)
        contains_click=1 → ~2.1  (binary, but now standardised)
    All features now on comparable scale → balanced gradient updates.
    """
    print("\n[3/7] Scaling handcrafted features …")

    scaler = StandardScaler()
    X_hand_train_s = scaler.fit_transform(X_hand_train)  # fit + transform
    X_hand_val_s   = scaler.transform(X_hand_val)        # transform only
    X_hand_test_s  = scaler.transform(X_hand_test)       # transform only

    print(f"      Scaler fitted on {len(X_hand_train):,} training samples")
    print(f"      Feature means (train): "
          f"{np.round(scaler.mean_, 3)}")
    print(f"      Feature stds  (train): "
          f"{np.round(scaler.scale_, 3)}")

    # Save scaler — must use same scaler at inference time
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    print(f"      ✔ Saved scaler → phase3_outputs/scaler.pkl")

    return X_hand_train_s, X_hand_val_s, X_hand_test_s


# ─────────────────────────────────────────────
# STEP 4 — CLASS WEIGHTS
# ─────────────────────────────────────────────
def get_class_weights(y_train):
    """
    Compute class weights inversely proportional to class frequency.

    Even though our dataset is ~balanced (52/48), we intentionally
    skew weights to penalise false negatives (missed phishing) more.

    compute_class_weight('balanced', ...) gives:
        weight[0] = N / (2 * count_legit)
        weight[1] = N / (2 * count_malicious)

    Then we manually boost class 1 (malicious) by ×1.5 to further
    penalise missing a phishing email vs. a false alarm.

    In practice this means:
        Getting a phishing email wrong costs 1.5× more than
        getting a legit email wrong.
    """
    print("\n[4/7] Computing class weights …")

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))

    # Boost phishing penalty
    class_weight_dict[1] *= 1.5

    print(f"      Base balanced weights : {dict(zip(classes, weights))}")
    print(f"      After phishing boost  : {class_weight_dict}")
    print(f"      → Model penalises missing phishing {class_weight_dict[1] / class_weight_dict[0]:.2f}× "
          f"more than false alarms")

    return class_weight_dict


# ─────────────────────────────────────────────
# STEP 5 — BUILD MODEL
# ─────────────────────────────────────────────
def build_model(num_hand_features: int) -> Model:
    """
    Dual-input TextCNN architecture.

    TEXT BRANCH
    ───────────
    Input(300,) → Embedding(20k, 64) → 3×Conv1D(64, k) + GlobalMaxPool
               → concat(192,) → Dropout(0.3) → Dense(64, ReLU)

    HAND BRANCH
    ───────────
    Input(7,) → Dense(16, ReLU)

    FUSION
    ──────
    Concat(80,) → BatchNorm → Dense(64, ReLU) → Dropout(0.4)
               → Dense(1, Sigmoid)

    Why GlobalMaxPool instead of GlobalAveragePool?
        MaxPool answers "did this pattern appear ANYWHERE?"
        AveragePool answers "how common was this pattern?"
        For phishing, one occurrence of "verify your password"
        is enough signal — max is the right choice.

    Why multiple kernel sizes?
        kernel=3 catches "click here now"   (short phrases)
        kernel=4 catches "verify your account" (medium)
        kernel=5 catches "update your billing info" (longer)
        Concatenating all three gives the model multi-granularity
        pattern detection in a single forward pass.
    """
    # ── Text input branch ──────────────────────────────────
    text_input = keras.Input(shape=(MAX_LEN,), name="text_input")

    # Embedding: integer token IDs → dense 64-dim vectors
    # mask_zero=False — Conv1D doesn't support masking so the mask
    # would be silently dropped anyway; removing it avoids the warning
    x = layers.Embedding(
        input_dim   = MAX_VOCAB,
        output_dim  = EMBED_DIM,
        mask_zero   = False,
        name        = "embedding"
    )(text_input)

    # Parallel convolutions — one branch per kernel size
    pooled_outputs = []
    for k in KERNEL_SIZES:
        conv = layers.Conv1D(
            filters     = NUM_FILTERS,
            kernel_size = k,
            activation  = "relu",
            padding     = "valid",       # no padding — keeps sequence info
            name        = f"conv_k{k}"
        )(x)
        pool = layers.GlobalMaxPooling1D(name=f"pool_k{k}")(conv)
        pooled_outputs.append(pool)

    # Concatenate all 3 pooled vectors → (192,)
    text_concat = layers.Concatenate(name="text_concat")(pooled_outputs)

    # Regularise text branch before fusion
    text_out = layers.Dropout(DROPOUT_CNN, name="dropout_text")(text_concat)
    text_out = layers.Dense(64, activation="relu", name="dense_text")(text_out)

    # ── Handcrafted features branch ────────────────────────
    hand_input = keras.Input(shape=(num_hand_features,), name="hand_input")
    hand_out   = layers.Dense(16, activation="relu", name="dense_hand")(hand_input)

    # ── Fusion ─────────────────────────────────────────────
    # Concatenate both branches → (64 + 16,) = (80,)
    merged = layers.Concatenate(name="fusion_concat")([text_out, hand_out])

    # BatchNorm normalises the merged vector
    # Text branch: Dense(64) ReLU outputs → roughly [0, ~3]
    # Hand branch: StandardScaled inputs  → roughly [-2, 2]
    # Without BatchNorm, text features may dominate purely by scale
    merged = layers.BatchNormalization(name="batch_norm")(merged)

    # Joint dense layer
    merged = layers.Dense(64, activation="relu", name="dense_joint")(merged)
    merged = layers.Dropout(DROPOUT_OUT, name="dropout_out")(merged)

    # ── Output ─────────────────────────────────────────────
    output = layers.Dense(1, activation="sigmoid", name="output")(merged)

    model = Model(
        inputs  = [text_input, hand_input],
        outputs = output,
        name    = "PhishingCNN"
    )

    return model


# ─────────────────────────────────────────────
# STEP 6 — TRAIN
# ─────────────────────────────────────────────
def train_model(model, splits, class_weight_dict):

    (X_text_train, X_text_val,
     X_hand_train_s, X_hand_val_s,
     y_train, y_val) = splits

    print("\n[6/7] Training …")
    model.summary()
    print()

    model.compile(
        optimizer = keras.optimizers.Adam(
            learning_rate = 1e-3,
            clipnorm      = 1.0,   # gradient clipping — prevents exploding
                                   # gradients in deep models
        ),
        loss      = "binary_crossentropy",
        metrics   = [
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]
    )

    callbacks = [
        # Stop when val_loss stops improving
        EarlyStopping(
            monitor              = "val_loss",
            patience             = ES_PATIENCE,
            restore_best_weights = True,   # roll back to best epoch
            verbose              = 1
        ),
        # Halve LR when val_loss stagnates
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = LR_PATIENCE,
            min_lr   = 1e-6,
            verbose  = 1
        ),
        # Save best checkpoint during training
        ModelCheckpoint(
            filepath          = str(OUTPUT_DIR / "best_model.keras"),
            monitor           = "val_loss",
            save_best_only    = True,
            verbose           = 0
        ),
    ]

    history = model.fit(
        x               = [X_text_train, X_hand_train_s],
        y               = y_train,
        validation_data = ([X_text_val, X_hand_val_s], y_val),
        epochs          = MAX_EPOCHS,
        batch_size      = BATCH_SIZE,
        class_weight    = class_weight_dict,
        callbacks       = callbacks,
        verbose         = 1
    )

    return history


# ─────────────────────────────────────────────
# STEP 7 — EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_text_test, X_hand_test_s, y_test):
    """
    Evaluate on the held-out test set.

    Two-part evaluation:

    Part A — standard metrics at threshold=0.5
        Gives a baseline picture of model performance.

    Part B — precision-recall threshold sweep
        Sweeps thresholds from 0.1 → 0.9 and prints the
        precision/recall/F1 at each point.

        Why not just monitor val_recall during training?
            The model would trivially maximise recall by predicting
            everything as malicious (recall=1.0, precision=0.5).
            Class weights push the model toward recall during training;
            the threshold sweep lets YOU pick the operating point
            post-training without retraining.

        How to read the table:
            Lower threshold → higher recall, lower precision
            Higher threshold → lower recall, higher precision
            Pick the row where recall meets your minimum requirement
            (e.g. recall ≥ 0.95) with the best precision you can get.

        The recommended threshold is saved so you can use it at
        inference time without re-running this script.
    """
    print("\n[7/7] Evaluating on test set …")

    # ── Part A: standard metrics at 0.5 ───────────────────
    results = model.evaluate(
        [X_text_test, X_hand_test_s], y_test,
        batch_size = BATCH_SIZE,
        verbose    = 0
    )
    print(f"\n      ── Test results at threshold=0.5 ─────────")
    for name, val in zip(model.metrics_names, results):
        print(f"        {name:<12} : {val:.4f}")

    # Get raw probabilities once — reused for all thresholds below
    y_prob = model.predict(
        [X_text_test, X_hand_test_s],
        batch_size = BATCH_SIZE,
        verbose    = 0
    ).flatten()

    # Confusion matrix at 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    print(f"\n      ── Confusion matrix (threshold=0.5) ─────")
    print(f"        True  positives (caught phishing)  : {tp:,}")
    print(f"        True  negatives (correct legit)    : {tn:,}")
    print(f"        False positives (legit → flagged)  : {fp:,}")
    print(f"        False negatives (phishing → missed): {fn:,}")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"\n        Miss rate   (FNR): {fnr:.3f}  ← minimise this")
    print(f"        False alarm (FPR): {fpr:.3f}")

    # ── Part B: threshold sweep ────────────────────────────
    print(f"\n      ── Precision / Recall sweep ──────────────")
    print(f"        {'Threshold':<12} {'Precision':<12} {'Recall':<10} "
          f"{'F1':<10} {'FNR':<8} {'FPR'}")
    print(f"        {'─'*9:<12} {'─'*9:<12} {'─'*6:<10} "
          f"{'─'*6:<10} {'─'*5:<8} {'─'*5}")

    thresholds = np.arange(0.1, 0.91, 0.05)
    best_threshold  = 0.5
    best_f1         = 0.0

    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp_t = int(((yp == 1) & (y_test == 1)).sum())
        tn_t = int(((yp == 0) & (y_test == 0)).sum())
        fp_t = int(((yp == 1) & (y_test == 0)).sum())
        fn_t = int(((yp == 0) & (y_test == 1)).sum())

        prec  = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        rec   = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fnr_t = fn_t / (fn_t + tp_t) if (fn_t + tp_t) > 0 else 0.0
        fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0

        # Mark row if recall >= 0.95 (your high-recall target)
        flag = " ← recall≥0.95" if rec >= 0.95 else ""

        print(f"        {t:<12.2f} {prec:<12.4f} {rec:<10.4f} "
              f"{f1:<10.4f} {fnr_t:<8.4f} {fpr_t:.4f}{flag}")

        if f1 > best_f1:
            best_f1        = f1
            best_threshold = t

    print(f"\n      Best F1 threshold : {best_threshold:.2f}  "
          f"(F1={best_f1:.4f})")
    print(f"      → To prioritise recall, pick any threshold "
          f"marked '← recall≥0.95' above")

    # Save recommended threshold
    threshold_info = {
        "best_f1_threshold"   : float(round(best_threshold, 2)),
        "best_f1_score"       : float(round(best_f1, 4)),
        "recall_target"       : 0.95,
        "note"                : (
            "Use best_f1_threshold for balanced performance. "
            "Lower the threshold toward the recall>=0.95 rows "
            "if catching all phishing matters more than false alarms."
        )
    }
    with open(OUTPUT_DIR / "threshold_info.json", "w") as f:
        json.dump(threshold_info, f, indent=2)
    print(f"      ✔ Threshold info saved → phase3_outputs/threshold_info.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_pipeline():
    print("=" * 55)
    print("  Phase 3: Model Training")
    print("=" * 55)

    # ── 1. Load ───────────────────────────────────────────
    X_text, X_hand, y, feature_names = load_data()

    # ── 2. Split ──────────────────────────────────────────
    (X_text_train, X_text_val, X_text_test,
     X_hand_train, X_hand_val, X_hand_test,
     y_train, y_val, y_test) = split_data(X_text, X_hand, y)

    # ── 3. Scale ──────────────────────────────────────────
    X_hand_train_s, X_hand_val_s, X_hand_test_s = scale_features(
        X_hand_train, X_hand_val, X_hand_test
    )

    # ── 4. Class weights ──────────────────────────────────
    class_weight_dict = get_class_weights(y_train)

    # ── 5. Build ──────────────────────────────────────────
    print("\n[5/7] Building model …")
    model = build_model(num_hand_features=X_hand.shape[1])
    try:
        keras.utils.plot_model(
            model,
            to_file     = str(OUTPUT_DIR / "model_diagram.png"),
            show_shapes = True,
            dpi         = 120
        )
        print("      ✔ Model diagram saved → phase3_outputs/model_diagram.png")
    except ImportError:
        print("      ⚠ plot_model skipped — run: pip install pydot && brew install graphviz")

    # ── 6. Train ──────────────────────────────────────────
    train_splits = (X_text_train, X_text_val,
                    X_hand_train_s, X_hand_val_s,
                    y_train, y_val)
    history = train_model(model, train_splits, class_weight_dict)

    # ── 7. Evaluate ───────────────────────────────────────
    evaluate_model(model, X_text_test, X_hand_test_s, y_test)

    # ── Save final model ──────────────────────────────────
    model.save(OUTPUT_DIR / "phishing_cnn.keras")
    print(f"\n      ✔ Final model saved → phase3_outputs/phishing_cnn.keras")

    print("\n" + "=" * 55)
    print("  Phase 3 Complete.")
    print("  Saved outputs:")
    print("    • phishing_cnn.keras      — trained model")
    print("    • best_model.keras        — best checkpoint")
    print("    • scaler.pkl              — feature scaler")
    print("    • threshold_info.json     — recommended decision thresholds")
    print("    • model_diagram.png       — architecture diagram (if pydot installed)")
    print("\n  To load for inference:")
    print("    model  = keras.models.load_model('phase3_outputs/phishing_cnn.keras')")
    print("    scaler = joblib.load('phase3_outputs/scaler.pkl')")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()