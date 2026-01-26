"""
DeepAllergen UMAP Visualization
Creates Figure 3: Distribution of allergens and non-allergens
"""

import numpy as np
import matplotlib.pyplot as plt
import umap

import deepallergen_image
import tensorflow as tf
from tensorflow.keras.models import Model
from Bio import SeqIO
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re

# ============================================================================
# ProtT5 Setup
# ============================================================================
MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
MAX_LEN = 112
USE_FP16 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"Loading ProtT5 model: {MODEL_NAME}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
t5_model = T5EncoderModel.from_pretrained(MODEL_NAME)

if USE_FP16 and device.type == "cuda":
    t5_model = t5_model.half()

t5_model = t5_model.to(device)
t5_model.eval()
print(f"ProtT5 ready\n")


def load_fasta_data(fasta_path):
    """Load sequences and labels from FASTA file."""
    sequences = []
    labels = []

    print(f"Reading: {fasta_path}")

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequence = str(record.seq)
        sequences.append(sequence)

        header = record.description.lower()
        if 'allergen' in header and 'non' not in header:
            labels.append(1)
        else:
            labels.append(0)

    print(f"Loaded {len(sequences)} sequences")
    print(f"Allergens: {sum(labels)}, Non-allergens: {len(labels) - sum(labels)}")

    return sequences, np.array(labels)


@torch.no_grad()
def sequence_to_t5_embedding(sequence, max_len=MAX_LEN):
    """Convert sequence to ProtT5 embedding."""
    seq = re.sub(r"[^A-Za-z]", "", sequence)
    seq_spaced = " ".join(list(seq[:max_len]))

    inputs = tokenizer(seq_spaced, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if USE_FP16 and device.type == "cuda":
        inputs["input_ids"] = inputs["input_ids"].to(torch.long)

    output = t5_model(**inputs).last_hidden_state.squeeze(0)
    output = output[1:-1]
    L, D = output.shape

    if L < max_len:
        pad = torch.zeros((max_len - L, D), device=output.device, dtype=output.dtype)
        output = torch.cat([output, pad], dim=0)
    else:
        output = output[:max_len]

    return output.float().cpu().numpy()


def sequences_to_embeddings(sequences):
    """Convert sequences to embeddings."""
    print(f"Converting {len(sequences)} sequences to embeddings...")

    embeddings = []
    for i, seq in enumerate(sequences, 1):
        emb = sequence_to_t5_embedding(seq)
        embeddings.append(emb)

        if i % 10 == 0 or i == len(sequences):
            print(f"Progress: {i}/{len(sequences)}", end='\r')

    print()
    embeddings = np.stack(embeddings).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings


def load_deepallergen_model():
    """Load trained DeepAllergen model."""
    from model import SqueezeExcitation1DLayer, TransformerBlock, PositionEmbedding

    model_path = '/Users/feyzagok/PycharmProjects/deepallergen/saved_model/t5_model/DeepAllergen'

    custom_objects = {
        'SqueezeExcitation1DLayer': SqueezeExcitation1DLayer,
        'TransformerBlock': TransformerBlock,
        'PositionEmbedding': PositionEmbedding
    }

    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("Model loaded")
    return model


def get_features_from_layer(model, X, layer_index):
    """Extract features from a specific layer."""
    layer = model.layers[layer_index]
    intermediate_model = Model(inputs=model.input, outputs=layer.output)
    features = intermediate_model.predict(X, verbose=0)

    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    print(features.shape)
    return features


def compute_umap(features):
    """Compute UMAP embeddings."""
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    embeddings = reducer.fit_transform(features)
    return embeddings


def plot_umap(embeddings, labels, title, ax):
    """Plot UMAP with red=allergen, cyan=non-allergen."""
    allergen_mask = (labels == 1)
    non_allergen_mask = (labels == 0)

    ax.scatter(embeddings[non_allergen_mask, 0],
              embeddings[non_allergen_mask, 1],
              c='cyan', label='non-allergen', alpha=0.6, s=20)

    ax.scatter(embeddings[allergen_mask, 0],
              embeddings[allergen_mask, 1],
              c='red', label='allergen', alpha=0.6, s=20)

    ax.set_xlabel('Dimension1', fontsize=11)
    ax.set_ylabel('Dimension2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def create_figure3(model, X_test, y_test, save_path='figure3.png'):
    layer_configs = [
(65, "SE"),
(70, "Dense layer"),
(73, "Dense layer"),
(79, "Dense layer last"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    print("\n" + "="*70)
    print("CREATING FIGURE 3")
    print("="*70)

    for idx, (layer_index, title) in enumerate(layer_configs):
        layer_name = model.layers[layer_index].name
        print(f"\n[{idx+1}/4] Processing layer {layer_index}: {layer_name}")

        print(f"  Extracting features...")
        features = get_features_from_layer(model, X_test, layer_index)
        print(f"  Features shape: {features.shape}")

        print(f"  Computing UMAP...")
        embeddings = compute_umap(features)
        print(f"  UMAP shape: {embeddings.shape}")

        plot_umap(embeddings, y_test, title, axes[idx])
        print(f"  Done")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("\n" + "="*70)
    print(f"Figure saved: {save_path}")
    print("="*70)
    plt.show()


if __name__ == "__main__":

    print("\n" + "="*70)
    print("DeepAllergen UMAP Visualization")
    print("="*70)

    # Step 1: Load model
    print("\n[STEP 1] Loading model...")
    model = load_deepallergen_model()

    # Step 2: Load data
    print("\n[STEP 2] Loading data from FASTA...")
    test_fasta = '/Users/feyzagok/PycharmProjects/deepallergen/data/all.test.fasta'
    test_sequences, y_test = load_fasta_data(test_fasta)

    print("\n[STEP 3] Converting to embeddings...")
    print("This may take several minutes...")
    X_test = sequences_to_embeddings(test_sequences)

    print(f"\nFinal shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Allergens: {np.sum(y_test==1)}, Non-allergens: {np.sum(y_test==0)}")

    # Step 3: Create visualizations
    print("\n[STEP 4] Creating visualizations...")
    create_figure3(model, X_test, y_test, save_path='figure3_deepallergen.png')

    print("\nALL DONE!")