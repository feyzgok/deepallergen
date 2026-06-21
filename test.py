"""
DeepAllergen Test Script
==========================
Loads the trained model and evaluates it on the held-out test set,
printing accuracy, precision, recall, F1, AUROC, AUPR, and the
confusion matrix.

Usage:
    python src/test.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeepAllergen

EXPERIMENT_NAME = "edit_here"
WEIGHTS_PATH    = f"saved_model/model_model/best_{EXPERIMENT_NAME}.weights.h5"
DATA_PATH       = "preprocessed_data/"

clf = DeepAllergen(input_size=(112, 1024))
_ = clf(tf.zeros((1, 112, 1024)))  # build before loading weights
clf.load_weights(WEIGHTS_PATH)
print("✅ Weights loaded!")

element_spec = (
    tf.TensorSpec(shape=(112, 1024), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.float32)
)
test_ds = tf.data.Dataset.load(
    os.path.join(DATA_PATH, "test_dataset"), element_spec=element_spec
).batch(32)

y_true, y_pred, y_pred_proba = [], [], []
for x_batch, y_batch in test_ds:
    preds = clf.predict(x_batch, verbose=0)
    y_true.extend(y_batch.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())
    y_pred_proba.extend(preds.flatten())

y_true       = np.array(y_true)
y_pred       = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
auroc     = roc_auc_score(y_true, y_pred_proba)
aupr      = average_precision_score(y_true, y_pred_proba)

print(f"\n===== Test Results: {EXPERIMENT_NAME} =====")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUROC:     {auroc:.4f}")
print(f"AUPR:      {aupr:.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
