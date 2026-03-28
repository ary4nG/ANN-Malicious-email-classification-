"""
Phase 1: Data Preprocessing Pipeline
======================================
Pipeline:
  Raw CSV  →  HTML clean  →  URL extraction  →  Text normalization  →  Save

Rules:
  - KEEP: !, $, @, numbers (strong phishing signals)
  - REMOVE: HTML tags, encoding artifacts, excessive whitespace
  - EXTRACT: URLs into separate column, remove from text
  - NORMALIZE: lowercase, single spaces

Note on dataset:
  phishing_email.csv is already pre-processed — URLs and HTML have been
  stripped upstream. The URL extraction step is retained in the pipeline
  so it works correctly on raw email data in production / future datasets.
"""

import re
import csv
import html
import unicodedata
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH  = Path("phishing_email.csv")
OUTPUT_PATH = Path("preprocessed_emails.csv")

# Regex patterns
URL_PATTERN = re.compile(
    r'(https?://[^\s<>"\']+|www\.[^\s<>"\']+)',
    re.IGNORECASE
)
HTML_TAG_PATTERN      = re.compile(r'<[^>]+>')
MULTI_SPACE_PATTERN   = re.compile(r'[ \t]+')
ENCODING_ARTIFACT_PATTERN = re.compile(
    r'&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);',
    re.IGNORECASE
)


# ─────────────────────────────────────────────
# STEP 2: Basic Cleaning
# ─────────────────────────────────────────────
def basic_clean(text: str) -> str:
    """
    Remove HTML tags, decode HTML entities, strip encoding artifacts.
    Does NOT remove !, $, @, numbers.
    """
    # Unescape HTML entities (e.g., &amp; → &)
    text = html.unescape(text)

    # Remove residual HTML entity patterns that didn't unescape
    text = ENCODING_ARTIFACT_PATTERN.sub(' ', text)

    # Remove HTML tags
    text = HTML_TAG_PATTERN.sub(' ', text)

    # Normalize unicode characters (NFKD handles ligatures, weird chars)
    text = unicodedata.normalize('NFKD', text)

    # Remove non-printable control characters (but keep newline for now)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cc' or ch == '\n')

    return text


# ─────────────────────────────────────────────
# STEP 3: URL Extraction
# ─────────────────────────────────────────────
def extract_urls(text: str) -> tuple[str, list[str]]:
    """
    Extract all URLs from text.
    Returns:
        text_no_urls : text with URLs replaced by the token <URL>
        urls         : list of extracted URLs
    """
    urls = URL_PATTERN.findall(text)
    # Replace each URL with <URL> placeholder token
    text_no_urls = URL_PATTERN.sub(' <URL> ', text)
    return text_no_urls, urls


# ─────────────────────────────────────────────
# STEP 4: Text Normalization
# ─────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """
    - Lowercase
    - Collapse multiple spaces / tabs into single space
    - Strip leading/trailing whitespace
    - Preserve: !, $, @, numbers, punctuation (phishing signals)
    """
    # Lowercase
    text = text.lower()

    # Collapse multiple spaces and tabs (but preserve newlines briefly)
    text = text.replace('\n', ' ')
    text = MULTI_SPACE_PATTERN.sub(' ', text)

    # Strip edges
    text = text.strip()

    return text


# ─────────────────────────────────────────────
# STEP 5: Tokenization Prep (lightweight)
# ─────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """
    Split on whitespace. Tokens retain punctuation attached to words.
    e.g. "urgent!!!" → ["urgent!!!"]
         "verify your account" → ["verify", "your", "account"]
    Full tokenization (with vocab building) happens in Phase 2/3.
    """
    return text.split()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def preprocess_row(text: str) -> dict:
    """Run full Phase 1 pipeline on a single email text."""

    # Step 2: Basic clean
    text = basic_clean(text)

    # Step 3: Extract URLs
    text, urls = extract_urls(text)

    # Step 4: Normalize
    text = normalize_text(text)

    # Step 5: Tokenization prep (optional preview)
    tokens = tokenize(text)

    return {
        "cleaned_text" : text,
        "token_count"  : len(tokens),   # kept for stats only; dropped from CSV
    }


def run_pipeline():
    print("=" * 55)
    print("  Phase 1: Data Preprocessing Pipeline")
    print("=" * 55)

    # ── Load ──────────────────────────────────────────────
    print(f"\n[1/3] Loading data from '{INPUT_PATH}' …")
    csv.field_size_limit(10**8)
    df = pd.read_csv(INPUT_PATH)
    print(f"      Rows loaded : {len(df):,}")
    print(f"      Columns     : {list(df.columns)}")
    print(f"      Label dist  : {df['label'].value_counts().to_dict()}")

    # ── Preprocess ────────────────────────────────────────
    print("\n[2/3] Running preprocessing …")
    results = df["text_combined"].fillna("").apply(preprocess_row)
    processed_df = pd.DataFrame(results.tolist())

    # Combine with original label
    output_df = pd.concat([processed_df, df["label"].reset_index(drop=True)], axis=1)

    # ── Stats preview ─────────────────────────────────────
    print(f"\n      ✔ Rows processed    : {len(output_df):,}")
    print(f"      ✔ Avg token count  : {output_df['token_count'].mean():.1f}")
    print(f"      ✔ Max token count  : {output_df['token_count'].max():,}")
    print(f"      ✔ Min token count  : {output_df['token_count'].min():,}")
    print(f"      ✔ Empty emails     : {(output_df['token_count'] == 0).sum():,}")

    # Sample output
    print("\n── Sample Output (row 0) ──────────────────────────")
    sample = output_df.iloc[0]
    print(f"  cleaned_text : {sample['cleaned_text'][:120]} …")
    print(f"  token_count  : {sample['token_count']}")
    print(f"  label        : {sample['label']}")

    # Drop debug column before saving
    output_df = output_df.drop(columns=["token_count"])

    # ── Save ──────────────────────────────────────────────
    print(f"\n[3/3] Saving to '{OUTPUT_PATH}' …")
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"      ✔ Saved {len(output_df):,} rows → {OUTPUT_PATH}")

    print("\n" + "=" * 55)
    print("  Phase 1 Complete. Output columns:")
    print("    • cleaned_text  — normalized text, URLs removed")
    print("    • label         — 0=legit, 1=malicious")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
