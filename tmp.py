import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# model yükle
model = tf.keras.models.load_model("saved_model/t5_model/DeepAllergen")

# test verisini yükle
test_dataset = tf.data.experimental.load("preprocessed_data_t5/test_dataset")
test_dataset = test_dataset.batch(32)

# gerçek etiket ve tahminleri topla
y_true = []
y_pred = []

for x, y in test_dataset:
    preds = model.predict(x)
    preds_binary = (preds > 0.5).astype(int)

    y_true.extend(y.numpy())
    y_pred.extend(preds_binary)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))

