import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import torch
from transformers import T5EncoderModel, T5Tokenizer

FASTA_DIR = "data/"
FASTA_FILES = {
    "train": "train.fasta",
    "val": "val.fasta",
    "test": "test.fasta",
}
SAVE_DIR = "preprocessed_data/"
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
MAX_LEN = 112
USE_FP16 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Device: {device}")

print(f"⏳ Loading ProtT5 model: {MODEL_NAME}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL_NAME)

if USE_FP16 and device.type == "cuda":
    model = model.half()
    print("✅ FP16 mode enabled (GPU)")

model = model.to(device)
model.eval()
EMB_DIM = model.config.d_model
print(f"✅ Model ready. Embedding dim: {EMB_DIM}\n")


VALID_POS = {"allergen", "positive", "1", "pos", "true", "yes", "allergene"}
VALID_NEG = {"protein", "non-allergen", "non_allergen", "nonallergen", "negative", "0", "neg", "false", "no"}


def normalize_label(raw):
    """Normalize string labels into 'allergen' or 'protein'."""
    if raw is None:
        return None
    s = str(raw).lower().strip().replace("\t", "").replace(":", "").replace("=", "")
    if s in VALID_POS:
        return "allergen"
    if s in VALID_NEG:
        return "protein"
    if s.isdigit():
        return "allergen" if int(s) == 1 else "protein"
    return None


def parse_fasta_with_labels(path):
    """Parse FASTA file and extract (sequence, label)."""
    data, cur_seq, cur_label = [], "", None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_seq and cur_label:
                    data.append((cur_seq, cur_label))
                hdr = line.lower()
                if "non-allergen" in hdr or re.search(r"non[-_]?allergen", hdr):
                    cur_label = "protein"
                elif "allergen" in hdr:
                    cur_label = "allergen"
                else:
                    cur_label = None
                cur_seq = ""
            else:
                cur_seq += line
        if cur_seq and cur_label:
            data.append((cur_seq, cur_label))
    return data


@torch.no_grad()
def sequence_to_t5(sequence: str, max_len=MAX_LEN) -> np.ndarray:
    """Convert one sequence into (max_len x EMB_DIM) embedding matrix."""
    seq = re.sub(r"[^A-Za-z]", "", sequence)
    seq_spaced = " ".join(list(seq[:max_len]))

    inputs = tokenizer(seq_spaced, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if USE_FP16 and device.type == "cuda":
        inputs["input_ids"] = inputs["input_ids"].to(torch.long)

    output = model(**inputs).last_hidden_state.squeeze(0)
    output = output[1:-1]  # remove special tokens
    L, D = output.shape

    # pad or truncate to max_len
    if L < max_len:
        pad = torch.zeros((max_len - L, D), device=output.device, dtype=output.dtype)
        output = torch.cat([output, pad], dim=0)
    else:
        output = output[:max_len]

    return output.float().cpu().numpy()


def load_split(name, fname):
    """Load a single pre-split FASTA file (train/val/test) into a DataFrame.

    Note: train.fasta, val.fasta, and test.fasta are expected to already be
    isoform-deduplicated and split — no re-splitting or stratification is
    performed here. See data/README.md for the CD-HIT preprocessing this
    data underwent prior to being placed in this directory.
    """
    path = os.path.join(FASTA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file: {path}")

    rows = parse_fasta_with_labels(path)
    print(f"📄 {fname}: {len(rows)} records")

    df = pd.DataFrame(rows, columns=["sequence", "label"])
    df["sequence"] = df["sequence"].astype(str).str.replace(r"[^A-Za-z]", "", regex=True)
    df = df[df["sequence"].str.len() > 0]
    df["label"] = df["label"].apply(normalize_label)
    df = df[df["label"].isin(["allergen", "protein"])].copy()

    if df.empty:
        raise RuntimeError(f"❌ No valid allergen/protein labels found in {fname}!")

    df["label_encoded"] = df["label"].map({"allergen": 1, "protein": 0}).astype(np.float32)
    print(f"  🏷️  {name} label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def embed_split(dframe, name):
    """Embeds all sequences in a DataFrame split using ProtT5."""
    X, y = [], []
    print(f"\n🔄 Embedding {name} ({len(dframe)} seqs)...")
    for i, (_, row) in enumerate(dframe.iterrows(), 1):
        X.append(sequence_to_t5(row["sequence"]))
        y.append(row["label_encoded"])
        if i % 10 == 0 or i == len(dframe):
            print(f"  {i}/{len(dframe)}", end="\r")
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"✅ {name} → X:{X.shape}, y:{y.shape}")
    return tf.data.Dataset.from_tensor_slices((X, y))


def save_dataset(ds, name):
    """Saves a single TensorFlow dataset, backing up any existing copy."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name)
    if os.path.exists(path):
        backup = path + f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(path, backup)
        print(f"📦 Backup created for {name}")
    tf.data.experimental.save(ds, path)
    print(f"💾 Saved: {path}")


def main():
    print("=" * 60)
    print("🚀 ProtT5-based Preprocessing Pipeline")
    print("=" * 60)
    print("ℹ️  Using pre-split, isoform-deduplicated train/val/test FASTA files.")
    print("   No re-splitting or CD-HIT filtering is performed here.\n")

    for split_name, fname in FASTA_FILES.items():
        df = load_split(split_name, fname)
        ds = embed_split(df, split_name.capitalize())
        save_dataset(ds, f"{split_name}_dataset")

    print("\n✅ All done!")
    print(f"📁 Output dir: {os.path.abspath(SAVE_DIR)}")
    print(f"🎯 Model input shape: (None, {MAX_LEN}, {EMB_DIM})")
    print("=" * 60)


if __name__ == "__main__":
    main()
