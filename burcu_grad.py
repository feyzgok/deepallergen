import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# =========================
# Grad-CAM Helpers (1D)
# =========================
def make_gradcam_1d(model, x, target_layer_name):
    """
    x: (1,112,1024)
    returns: cam (L,), pred_prob
    """
    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(x, training=False)  # conv_out: (1,L,C)
        score = pred[:, 0]                              # sigmoid

    grads = tape.gradient(score, conv_out)             # (1,L,C)
    weights = tf.reduce_mean(grads, axis=1)            # (1,C)
    cam = tf.reduce_sum(conv_out * tf.expand_dims(weights, axis=1), axis=-1)  # (1,L)
    cam = tf.nn.relu(cam)[0]                           # (L,)

    cam = cam.numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, float(pred[0, 0].numpy())

def resize_1d_to_len(cam, new_len=112):
    cam_t = tf.convert_to_tensor(cam[None, :, None], dtype=tf.float32)  # (1,L,1)
    cam_r = tf.image.resize(cam_t, size=(new_len, 1), method="bilinear")
    return cam_r[0, :, 0].numpy()

def top_windows(cam, window_len=15, k=2, min_gap=5):
    """En yüksek toplam skorlu k pencereyi seçer."""
    L = len(cam)
    scores = [(s, cam[s:s+window_len].sum()) for s in range(0, L - window_len + 1)]
    scores.sort(key=lambda x: x[1], reverse=True)

    chosen = []
    for s, _ in scores:
        e = s + window_len
        ok = True
        for (cs, ce) in chosen:
            if not (e + min_gap <= cs or s >= ce + min_gap):
                ok = False
                break
        if ok:
            chosen.append((s, e))
        if len(chosen) == k:
            break
    return chosen

def pick_conv1d_layers(model, prefer_last_k=3):
    """Modeldeki Conv1D layer isimlerini döndürür (son k tane)."""
    convs = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv1D)]
    if len(convs) == 0:
        raise ValueError("No Conv1D layers found.")
    return convs[-prefer_last_k:]

def multilayer_gradcam(model, x, layer_names):
    """returns cams: (n_layers,112), pred_prob"""
    cams = []
    pred_prob = None
    for ln in layer_names:
        cam, p = make_gradcam_1d(model, x, ln)
        cam_112 = cam if len(cam) == 112 else resize_1d_to_len(cam, 112)
        cams.append(cam_112)
        pred_prob = p
    return np.stack(cams, axis=0), pred_prob

def plot_fgh_multilayer(cams, layer_names, boxes, save_path, title=""):
    """
    cams: (n_layers,112)
    """
    fig, ax = plt.subplots(figsize=(14, 3.0))
    ax.imshow(cams, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xlabel("Position (0..111)")
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)
    ax.set_title(title)

    # kutuları tüm satırlara çiz
    for (s, e) in boxes:
        ax.add_patch(plt.Rectangle((s, -0.5), e - s, len(layer_names), fill=False, linewidth=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# =========================
# Paths / Settings
# =========================
MODEL_PATH = r"/Users/feyzagok/PycharmProjects/deepallergen/saved_model/t5_model/DeepAllergen"
TEST_DATASET_PATH = r"preprocessed_data_t5_testonly"

OUT_DIR = "gradcam_cases_multilayer"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_LEN = 15
K_BOXES = 2
N_LAYERS = 3  # kaç conv layer birleştirelim? (makale gibi katmanlı)

# =========================
# Load model & data
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

test_dataset = tf.data.experimental.load(TEST_DATASET_PATH).batch(1)

# hangi conv layerlar?
layer_names = pick_conv1d_layers(model, prefer_last_k=N_LAYERS)
print("✅ Using Conv1D layers for multi-layer Grad-CAM:", layer_names)

# =========================
# Pass 1: collect predictions for metrics + choose cases
# =========================
records = []  # (idx, y_true, y_pred, prob)
y_true_all, y_pred_all, y_prob_all = [], [], []

for i, (x, y) in enumerate(test_dataset, 1):
    prob = float(model.predict(x, verbose=0)[0][0])
    pred = int(prob > 0.5)
    yt = int(y.numpy()[0])

    y_true_all.append(yt)
    y_pred_all.append(pred)
    y_prob_all.append(prob)
    records.append((i, yt, pred, prob))

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_prob_all = np.array(y_prob_all)

# =========================
# Metrics
# =========================
accuracy = accuracy_score(y_true_all, y_pred_all)
precision = precision_score(y_true_all, y_pred_all, zero_division=0)
recall = recall_score(y_true_all, y_pred_all, zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

if len(np.unique(y_true_all)) > 1:
    auroc = roc_auc_score(y_true_all, y_prob_all)
    aupr = average_precision_score(y_true_all, y_prob_all)
else:
    auroc = float("nan")
    aupr = float("nan")

print("\nAdditional Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUROC: {auroc:.4f}")
print(f"AUPR: {aupr:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_all, y_pred_all))

# =========================
# Case selection (paper-like)
# Choose: TP_high, TN_low, FP_high, FN_low
# =========================
TP = [r for r in records if r[1] == 1 and r[2] == 1]
TN = [r for r in records if r[1] == 0 and r[2] == 0]
FP = [r for r in records if r[1] == 0 and r[2] == 1]
FN = [r for r in records if r[1] == 1 and r[2] == 0]

cases = {}

if TP:
    cases["TP_high"] = max(TP, key=lambda t: t[3])  # highest prob among TP
if TN:
    cases["TN_low"] = min(TN, key=lambda t: t[3])   # lowest prob among TN
if FP:
    cases["FP_high"] = max(FP, key=lambda t: t[3])  # highest prob among FP
if FN:
    cases["FN_low"] = min(FN, key=lambda t: t[3])   # lowest prob among FN

print("\nSelected cases:")
for k, (idx, yt, yp, prob) in cases.items():
    print(f"  {k}: sample={idx}, true={yt}, pred={yp}, p={prob:.3f}")

# =========================
# Pass 2: generate multi-layer Grad-CAM ONLY for selected cases
# =========================
target_indices = set(v[0] for v in cases.values())

for i, (x, y) in enumerate(test_dataset, 1):
    if i not in target_indices:
        continue

    yt = int(y.numpy()[0])
    prob = float(model.predict(x, verbose=0)[0][0])
    yp = int(prob > 0.5)

    cams, _ = multilayer_gradcam(model, x, layer_names)

    # kutuları en "geç" layer’a göre seç
    boxes = top_windows(cams[-1], window_len=WINDOW_LEN, k=K_BOXES, min_gap=5)

    # dosya adı
    fname = f"sample{i:04d}_true{yt}_pred{yp}_p{prob:.3f}_multilayer.png"
    fpath = os.path.join(OUT_DIR, fname)

    plot_fgh_multilayer(
        cams,
        layer_names,
        boxes,
        fpath,
        title=f"Multi-layer Grad-CAM | sample {i} | true={yt} pred={yp} p={prob:.3f}"
    )

print(f"\n✅ Multi-layer Grad-CAM figures saved to: {OUT_DIR}")