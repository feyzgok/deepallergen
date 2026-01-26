import datetime
import time
from model import *
from utils import *
from tensorflow import keras
import tensorflow_addons as tfa

model_arch = "DeepAllergen"
path = ""

# Multi-GPU Strategy
strategy = tf.distribute.MirroredStrategy()
n_hardwares = strategy.num_replicas_in_sync
print(f"🖥️  Replicas: {n_hardwares}")

## TensorBoard Profiler
tf.profiler.experimental.server.start(6000)

# ======================= PATHS =======================
# ⚠️ preprocessing çıktı klasörü
saved_tf_dataset_path = "preprocessed_data/"
model_path = "saved_model/model/"
result_path = "output/results/" + model_arch + "/"
# =====================================================

os.makedirs(model_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

print("\n📂 Dataset'ler yükleniyor...")

# Dataset'leri yükle
train_dataset = tf.data.experimental.load(
    os.path.join(saved_tf_dataset_path, "train_dataset"),
    element_spec=(
        tf.TensorSpec(shape=(112, 1024), dtype=tf.float32),  # ⚠️ T5 shape
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

val_dataset = tf.data.experimental.load(
    os.path.join(saved_tf_dataset_path, "val_dataset"),
    element_spec=(
        tf.TensorSpec(shape=(112, 1024), dtype=tf.float32),  # ⚠️ T5 shape
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
)

print("✅ Dataset'ler yüklendi")

## Disable AutoShard
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

# Batch size ayarı
batch_size = 8 * n_hardwares
print(f"🎯 Batch size: {batch_size}")

train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

# Cache/prefetch optimizasyonu
train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)

print("✅ Dataset pipeline hazır\n")

# ======================= MODEL =======================
print("🏗️  Model oluşturuluyor...")

with strategy.scope():
    # ⚠️ CRITICAL: T5 input_size=(112, 1024)
    if model_arch == "DeepAllergen":
        model = DeepAllergen(
            input_size=(112, 1024),
            n_filters=128,
            n_head=8,
            kernel_size=11,
            n_ff_dims=128,
            n_classes=1
        )
    else:
        model = return_model(model_arch)

    # Loss & Optimizer
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=3e-4,
        weight_decay=1e-4,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

print("✅ Model compile edildi\n")
model.summary()

# ======================= CALLBACKS =======================
print("\n⚙️  Callbacks hazırlanıyor...")

# Cosine Learning Rate Scheduler
scheduler = CosineScheduler(
    max_update=84,  # 85 epoch - 1 (0-indexed)
    base_lr=3e-4,
    final_lr=1e-5,
    warmup_steps=10,
    warmup_begin_lr=1e-4
)
learning_rate_cb = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# Reduce LR on Plateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# TensorBoard
log_dir = path + "logs/fit/" + model_arch + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Model Checkpoint (en iyi modeli kaydet)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(model_path, model_arch + "_best.h5"),
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

print("✅ Callbacks hazır\n")

# ======================= TRAINING =======================
print("=" * 60)
print("🚀 Eğitim başlıyor...")
print("=" * 60)
print(f"📊 Epochs: 85")
print(f"📦 Batch size: {batch_size}")
print(f"🎯 Input shape: (None, 112, 1024)")
print("=" * 60 + "\n")

time1 = time.time()

history = model.fit(
    x=train_dataset,
    epochs=85,
    batch_size=batch_size,
    verbose=2,
    validation_data=val_dataset,
    callbacks=[
        tensorboard_cb,
        early_stop,
        learning_rate_cb,
        reduce_lr,
        checkpoint_cb
    ]
)

time2 = time.time()
training_time = time2 - time1

print("\n" + "=" * 60)
print("✅ Eğitim tamamlandı!")
print("=" * 60)
print(f"⏱️  Toplam süre: {training_time / 60:.2f} dakika")
print("=" * 60 + "\n")

# ======================= SAVE & EVALUATE =======================
print("💾 Model kaydediliyor...")
model.save(os.path.join(model_path, model_arch))
print(f"✅ Model kaydedildi: {os.path.join(model_path, model_arch)}")

print("\n📊 Final evaluation...")
result_dic = model.evaluate(val_dataset, batch_size=batch_size, return_dict=True)
result_dic["training_time"] = training_time

# Sonuçları kaydet
save_result(result_dic, result_path)

print("\n" + "=" * 60)
print("🎉 TÜM İŞLEMLER TAMAMLANDI!")
print("=" * 60)
print(f"📁 Model: {model_path}")
print(f"📁 Results: {result_path}")
print(f"📁 TensorBoard: {log_dir}")
print("\n💡 TensorBoard için: tensorboard --logdir={0}".format(log_dir))
print("=" * 60)