# Malicious Email Classifier

Binary classification system that detects phishing and spam emails using a dual-input neural network. Combines a TextCNN branch for sequence-level patterns with a handcrafted feature branch for interpretable domain signals.

Target: Recall >= 95%, Accuracy >= 90%, Dataset: ~82,000 emails

---

## Project Structure

```
ANN/
├── phase1_preprocessing.py          # Raw CSV -> cleaned text
├── phase2_FeatureRepresentation.py  # Text sequences + handcrafted features
├── phase3_model.py                  # Model training and evaluation
├── FuzzyLayer.py                    # Post-processing: fuzzy decision layer
├── phishing_email.csv               # Raw dataset
├── preprocessed_emails.csv          # Phase 1 output
├── phase2_outputs/
│   ├── X_text.npy                   # Token sequences (N, 300)
│   ├── X_hand.npy                   # Handcrafted features (N, 7), unscaled
│   ├── y.npy                        # Labels: 0=legit, 1=malicious
│   ├── tokenizer.json               # Fitted Keras tokenizer
│   └── hand_feature_names.json
└── phase3_outputs/
    ├── phishing_cnn.keras            # Final trained model
    ├── best_model.keras             # Best checkpoint (by val_loss)
    ├── scaler.pkl                   # Fitted StandardScaler
    └── threshold_info.json          # Recommended decision thresholds
```

---

## Pipeline

```
Raw CSV
  |
  v
Phase 1 - Preprocessing
  HTML clean, URL extraction, text normalization
  Output: preprocessed_emails.csv
  |
  v
Phase 2 - Feature Representation
  Text: Tokenizer -> integer sequences -> pad/truncate to (N, 300)
  Hand: 7 domain features -> (N, 7)
  |
  v
Phase 3 - Model Training
  Split (80/10/10 stratified), scale hand features, class weights
  Dual-input TextCNN, EarlyStopping, threshold sweep
  Output: phishing_cnn.keras, scaler.pkl, threshold_info.json
  |
  v
FuzzyLayer - Inference Post-Processing
  P(malicious) + feature signals -> ALLOW / WARN / BLOCK
```

---

## Architecture

### Text Branch

| Layer | Detail |
|---|---|
| Input | (N, 300) integer token sequences |
| Embedding | Vocab 20k -> dim 64, trainable |
| Conv1D x 3 | Kernels 3, 4, 5 -- 128 filters each, ReLU |
| GlobalMaxPool x 3 | One per kernel, concatenated to (384,) |
| Dropout | Rate 0.3 |
| Dense | 64 units, ReLU |

### Handcrafted Feature Branch

| Layer | Detail |
|---|---|
| Input | (N, 7) scaled features |
| Dense | 16 units, ReLU |

### Fusion Head

| Layer | Detail |
|---|---|
| Concatenate | (64 + 16,) = (80,) |
| BatchNormalization | Stabilises mixed-scale inputs |
| Dense | 64 units, ReLU |
| Dropout | Rate 0.4 |
| Output | 1 unit, Sigmoid -- P(malicious) in [0, 1] |

---

## Phases

### Phase 1 -- Preprocessing

Cleans raw email text while preserving phishing signals (!, $, @, numbers):

- HTML tag removal and entity decoding
- URL extraction, replaced with `<URL>` token
- Unicode normalization (NFKD)
- Lowercase and whitespace collapsing

### Phase 2 -- Feature Representation

**Text features:** Keras Tokenizer (vocab cap 20k), sequences padded/truncated to length 300.

**Handcrafted features (7 total):**

| Feature | Description |
|---|---|
| num_words | Total token count |
| num_exclamations | Count of ! characters |
| contains_urgent | Urgency keywords (e.g. urgent, asap, final notice) |
| contains_verify | Verification keywords (e.g. verify your identity) |
| contains_click | Action keywords (e.g. click here, sign in) |
| contains_sensitive | Credential keywords (e.g. password, SSN, credit card) |
| repeated_words_ratio | Fraction of tokens that are duplicates |

Scaling is deferred to Phase 3 (fit on train split only) to prevent data leakage.

### Phase 3 -- Model Training

| Setting | Value |
|---|---|
| Split | 80 / 10 / 10, stratified |
| Scaler | StandardScaler, fit on train only |
| Class weights | Balanced + 1.5x boost for malicious class |
| Optimizer | Adam (lr=1e-3, clipnorm=1.0) |
| Loss | Binary crossentropy |
| Batch size | 64 |
| EarlyStopping | patience=4, monitors val_loss |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=2) |

**Results:**

| Metric | Value |
|---|---|
| Best F1 Threshold | 0.25 |
| Best F1 Score | 0.9912 |
| Recall Target | >= 0.95 |

### FuzzyLayer -- Post-processing

Converts raw P(malicious) into a structured verdict with a human-readable reason.

**Probability zones:**

| Zone | Threshold | Action |
|---|---|---|
| SAFE | P < 0.30 | ALLOW |
| SUSPICIOUS | 0.30 <= P < 0.70 | WARN |
| HIGH_RISK | P >= 0.70 | BLOCK |

**Override rules** (can only escalate, never de-escalate):

| Rule | Condition | Effect |
|---|---|---|
| 1 | contains_sensitive at P >= 0.20 | SAFE -> SUSPICIOUS |
| 2 | contains_sensitive + any other signal at P >= 0.30 | -> HIGH_RISK |
| 3 | contains_urgent + contains_verify at P >= 0.30 | -> HIGH_RISK |
| 4 | Weighted signal score >= 3 at P >= 0.30 | -> HIGH_RISK |

contains_sensitive is weighted 2x because credential requests are a stronger phishing signal than urgency or click language alone.

---

## Quickstart

### Requirements

```bash
pip install tensorflow scikit-learn pandas numpy joblib
```

### Run the pipeline

```bash
python phase1_preprocessing.py
python phase2_FeatureRepresentation.py
python phase3_model.py
```

### Inference

```python
import numpy as np
import joblib
from tensorflow import keras
from FuzzyLayer import fuzzy_decision

model  = keras.models.load_model("phase3_outputs/phishing_cnn.keras")
scaler = joblib.load("phase3_outputs/scaler.pkl")

X_text = np.load("phase2_outputs/X_text.npy")[:1]   # (1, 300)
X_hand = np.load("phase2_outputs/X_hand.npy")[:1]   # (1, 7)
X_hand_scaled = scaler.transform(X_hand)

prob = model.predict([X_text, X_hand_scaled])[0][0]

features = {
    "contains_sensitive": int(X_hand[0][5]),
    "contains_urgent":    int(X_hand[0][2]),
    "contains_verify":    int(X_hand[0][3]),
    "contains_click":     int(X_hand[0][4]),
}
verdict = fuzzy_decision(prob, features)
print(verdict)
```

### FuzzyLayer self-test

```bash
python FuzzyLayer.py
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| NumPy | Array operations |
| pandas | Data loading and manipulation |
| scikit-learn | Splits, StandardScaler, class weights |
| joblib | Scaler serialization |
