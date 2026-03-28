"""
Phase 2: Feature Representation
=================================
Pipeline:
  preprocessed_emails.csv
      │
      ├──► [A] Text Features
      │         Tokenizer → integer sequences → pad/truncate to MAX_LEN
      │         Output : X_text.npy  shape (N, 300)
      │
      └──► [B] Handcrafted Features (7 features)
                num_words, num_exclamations,
                contains_urgent, contains_verify, contains_click,
                contains_sensitive, repeated_words_ratio
                Output : X_hand.npy  shape (N, 7)

                Fixes vs v1:
                  - uppercase_ratio dropped (text is lowercased → always 0)
                  - keyword matching uses \b word boundaries (no false submatches)
                  - keyword lists expanded for better coverage

  Labels saved separately : y.npy  shape (N,)

  NOTE: Feature scaling (StandardScaler) is intentionally deferred to
  Phase 3 and fitted on training split only — prevents data leakage.

Outputs fed into Phase 3 (model training).
"""

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer          # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH   = Path("preprocessed_emails.csv")
OUTPUT_DIR   = Path("phase2_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_VOCAB    = 20_000   # keep the 20k most frequent tokens
MAX_LEN      = 300      # pad / truncate every sequence to this length
                        # covers ~95 %+ of emails (avg=160, median will be lower)
PADDING      = "post"   # zeros appended at the END  → better for pooling layers
TRUNCATING   = "post"   # long emails cut from the END (beginning is most signal-dense)

# ─────────────────────────────────────────────
# KEYWORD SETS
# Matching strategy:
#   • Single words  → matched with \b word boundaries
#                     prevents "click" matching "clickbait" etc.
#   • Multi-word phrases → matched with exact substring search
#                     (word boundaries implicit at phrase edges)
#   All keywords lowercase — texts are already normalized.
# ─────────────────────────────────────────────

# Split into single-word and phrase sets for correct matching logic
URGENT_SINGLE   = {"urgent", "immediately", "asap", "expires",
                   "deadline", "hurry", "warning", "alert",
                   "critical", "important", "respond"}
URGENT_PHRASES  = {"right away", "act now", "act immediately",
                   "limited time", "time sensitive", "respond now",
                   "don't delay", "do not delay", "last chance",
                   "final notice", "expiring soon"}

VERIFY_SINGLE   = {"verify", "verification", "confirm", "validate",
                   "authenticate", "reconfirm", "reverify"}
VERIFY_PHRASES  = {"re-confirm", "re-verify", "confirm your",
                   "verify your", "validate your", "update your details",
                   "confirm your identity", "verify your identity"}

CLICK_SINGLE    = {"click", "tap", "visit", "navigate",
                   "login", "signin"}
CLICK_PHRASES   = {"click here", "click now", "click below",
                   "click the link", "follow the link", "open the link",
                   "go to", "log in", "sign in", "access your",
                   "log-in", "sign-in"}

SENSITIVE_SINGLE = {"password", "passwd", "ssn", "cvv",
                    "dob", "pin"}
SENSITIVE_PHRASES = {"social security", "credit card", "bank account",
                     "account number", "date of birth", "card number",
                     "billing info", "billing information",
                     "payment details", "security code"}


# ─────────────────────────────────────────────
# PART A — TEXT FEATURES
# ─────────────────────────────────────────────

def build_tokenizer(texts: pd.Series) -> Tokenizer:
    """
    Fit a Keras Tokenizer on the corpus.

    - oov_token="<OOV>" : unknown words at inference time → token 1
                          (prevents crashes on unseen vocab)
    - num_words=MAX_VOCAB: only the top MAX_VOCAB-1 words are kept;
                           rarer words are treated as OOV automatically
    """
    print(f"      Building tokenizer (vocab cap = {MAX_VOCAB:,}) …")
    tok = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(texts)

    full_vocab = len(tok.word_index)
    print(f"      Full corpus vocabulary : {full_vocab:,} unique tokens")
    print(f"      Tokens kept (MAX_VOCAB) : {MAX_VOCAB:,}")
    print(f"      Tokens discarded (rare) : {max(0, full_vocab - MAX_VOCAB):,}")
    return tok


def texts_to_padded_sequences(tok: Tokenizer, texts: pd.Series) -> np.ndarray:
    """
    Convert texts → integer sequences → padded/truncated numpy array.

    Each row  : one email represented as MAX_LEN integers
    Token 0   : reserved for padding
    Token 1   : <OOV>  (out-of-vocabulary words)
    Token 2+  : actual words, ranked by frequency
    """
    sequences = tok.texts_to_sequences(texts)   # list of lists of ints
    padded    = pad_sequences(
        sequences,
        maxlen     = MAX_LEN,
        padding    = PADDING,
        truncating = TRUNCATING,
        value      = 0,         # padding value (0 = the reserved pad index)
    )
    return padded   # shape: (N, MAX_LEN)


# ─────────────────────────────────────────────
# PART B — HANDCRAFTED FEATURES
# ─────────────────────────────────────────────

def _match_keywords(text: str, single_words: set, phrases: set) -> int:
    """
    Return 1 if any keyword matches in text, else 0.

    Single words : matched with \b word boundaries
                   → "click" matches "click here" but NOT "clickbait"
                   → "pin"   matches "enter pin"  but NOT "opinion"

    Phrases      : matched as exact substrings
                   → word boundaries are implicit at phrase edges
                   → "act now" won't match inside a longer run-on phrase
                      but covers the common phishing phrasing as-is
    """
    # Single word matching — word boundaries prevent false submatches
    for word in single_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            return 1

    # Phrase matching — exact substring (boundaries implicit)
    for phrase in phrases:
        if phrase in text:
            return 1

    return 0


def _repeated_words_ratio(text: str) -> float:
    """
    Fraction of tokens that are duplicates.

    Spam/phishing often pads content with repeated filler words to
    bypass keyword filters.  A legit email rarely repeats the same
    word more than a few times.

    Formula:
        (total tokens - unique tokens) / total tokens
    Returns 0.0 for empty strings.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    counts  = Counter(tokens)
    repeats = sum(v - 1 for v in counts.values() if v > 1)
    return repeats / len(tokens)


def extract_handcrafted(row: str) -> dict:
    """
    Compute 7 handcrafted features for a single cleaned email string.

    Features
    --------
    num_words             : total token count
                            (phishing emails tend to be shorter/more direct)

    num_exclamations      : count of '!' characters
                            (urgency marker very common in phishing)

    contains_urgent       : 1 if any urgency keyword/phrase found
                            uses \b boundaries for single words
    contains_verify       : 1 if any verification keyword/phrase found
    contains_click        : 1 if any click/action keyword/phrase found
    contains_sensitive    : 1 if any sensitive-info keyword/phrase found

    repeated_words_ratio  : fraction of tokens that are repeated
                            spam often pads content with filler words

    DROPPED vs v1
    -------------
    uppercase_ratio : always 0.0 on lowercased text → removed

    SCALING NOTE
    ------------
    num_words and num_exclamations are on different scales than the
    binary features (0/1). Do NOT scale here — StandardScaler will be
    fitted on the training split only in Phase 3 to avoid data leakage.
    """
    text   = row   # already cleaned and lowercased
    tokens = text.split()

    num_words        = len(tokens)
    num_exclamations = text.count("!")

    contains_urgent   = _match_keywords(text, URGENT_SINGLE,   URGENT_PHRASES)
    contains_verify   = _match_keywords(text, VERIFY_SINGLE,   VERIFY_PHRASES)
    contains_click    = _match_keywords(text, CLICK_SINGLE,    CLICK_PHRASES)
    contains_sensitive= _match_keywords(text, SENSITIVE_SINGLE, SENSITIVE_PHRASES)
    repeated_ratio    = _repeated_words_ratio(text)

    return {
        "num_words"            : num_words,
        "num_exclamations"     : num_exclamations,
        "contains_urgent"      : contains_urgent,
        "contains_verify"      : contains_verify,
        "contains_click"       : contains_click,
        "contains_sensitive"   : contains_sensitive,
        "repeated_words_ratio" : repeated_ratio,
    }


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    print("=" * 55)
    print("  Phase 2: Feature Representation")
    print("=" * 55)

    # ── Load ──────────────────────────────────────────────
    print(f"\n[1/4] Loading '{INPUT_PATH}' …")
    df = pd.read_csv(INPUT_PATH)
    df["cleaned_text"] = df["cleaned_text"].fillna("")
    print(f"      Rows : {len(df):,}  |  Columns : {list(df.columns)}")

    texts  = df["cleaned_text"]
    labels = df["label"].values   # numpy array, shape (N,)

    # ── Part A: Text features ─────────────────────────────
    print("\n[2/4] Building text sequences …")
    tokenizer  = build_tokenizer(texts)
    X_text     = texts_to_padded_sequences(tokenizer, texts)

    # Sequence length stats (before padding — i.e. real email lengths)
    raw_lengths = texts.apply(lambda t: len(t.split()))
    print(f"\n      Sequence length stats (real token counts):")
    print(f"        mean   : {raw_lengths.mean():.1f}")
    print(f"        median : {raw_lengths.median():.1f}")
    print(f"        p95    : {raw_lengths.quantile(0.95):.0f}")
    print(f"        p99    : {raw_lengths.quantile(0.99):.0f}")
    print(f"        max    : {raw_lengths.max():,}")
    pct_covered = (raw_lengths <= MAX_LEN).mean() * 100
    print(f"\n      Emails fully within MAX_LEN={MAX_LEN}: {pct_covered:.1f}%")
    print(f"      X_text shape : {X_text.shape}")

    # ── Part B: Handcrafted features ──────────────────────
    print("\n[3/4] Extracting handcrafted features …")
    hand_df  = texts.apply(extract_handcrafted)
    hand_df  = pd.DataFrame(hand_df.tolist())
    X_hand   = hand_df.values.astype(np.float32)

    print(f"      X_hand shape : {X_hand.shape}")
    print(f"\n      Feature means (sanity check):")
    for col, val in zip(hand_df.columns, X_hand.mean(axis=0)):
        print(f"        {col:<25} : {val:.4f}")

    # ── Save ──────────────────────────────────────────────
    print(f"\n[4/4] Saving outputs to '{OUTPUT_DIR}/' …")

    np.save(OUTPUT_DIR / "X_text.npy",  X_text)
    np.save(OUTPUT_DIR / "X_hand.npy",  X_hand)
    np.save(OUTPUT_DIR / "y.npy",       labels)

    # Save tokenizer config so Phase 3 can reload it without re-fitting
    tok_config = tokenizer.to_json()
    with open(OUTPUT_DIR / "tokenizer.json", "w") as f:
        f.write(tok_config)

    # Save feature column names (useful for Phase 3 / debugging)
    with open(OUTPUT_DIR / "hand_feature_names.json", "w") as f:
        json.dump(list(hand_df.columns), f, indent=2)

    print(f"\n      ✔ X_text.npy          → {X_text.shape}  (int32)")
    print(f"      ✔ X_hand.npy          → {X_hand.shape}  (float32, 7 features, unscaled)")
    print(f"      ✔ y.npy               → {labels.shape}  (int)")
    print(f"      ✔ tokenizer.json      → reload in Phase 3 without re-fitting")
    print(f"      ✔ hand_feature_names.json")

    print("\n" + "=" * 55)
    print("  Phase 2 Complete.")
    print("  Outputs ready for Phase 3 (model training):")
    print("    • X_text : token sequences  → feeds Embedding layer")
    print("    • X_hand : 7 handcrafted features (unscaled)")
    print("               → StandardScaler applied in Phase 3")
    print("    • y      : labels")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()