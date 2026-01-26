# Suppress warnings BEFORE imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import warnings

warnings.filterwarnings('ignore')

import datetime
import time
import json
import gc
import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

# Suppress TF logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from model import *
from utils import *

# ============================================================================
# AYARLAR
# ============================================================================
model_arch = "DeepAllergen"
path = ""

# Sabit hiperparametreler
FIXED_WEIGHT_DECAY = 1e-4
FIXED_CLIPNORM = 1.0
FIXED_LABEL_SMOOTHING = 0.05
FIXED_WARMUP_STEPS = 10

# Epoch sayıları
EPOCHS_TRIAL = 30  # Optuna trial'ları için
EPOCHS_FINAL = 85  # Final model için

# GPU setup
strategy = tf.distribute.MirroredStrategy()
n_hardwares = strategy.num_replicas_in_sync
print("Replicas:", n_hardwares)

# TensorBoard profiler
tf.profiler.experimental.server.start(6000)

# Paths
saved_tf_dataset_path = "preprocessed_data/"
model_path = "saved_model/defined_media_model/"
result_path = "output/defined_media/" + model_arch + "/"
optuna_results_path = os.path.join(result_path, "optuna_results/")

os.makedirs(model_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)
os.makedirs(optuna_results_path, exist_ok=True)

print("Dataset'ler yükleniyor...")

train_dataset = tf.data.experimental.load(os.path.join(saved_tf_dataset_path, "train_dataset"))
val_dataset = tf.data.experimental.load(os.path.join(saved_tf_dataset_path, "val_dataset"))

# Disable AutoShard
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_dataset = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)

# Batch size
batch_size = 8 * n_hardwares
print(f"Batch size: {batch_size}")

train_dataset = train_dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

print("Dataset'ler hazır!\n")

def objective(trial):
    """Sadece learning rate'i optimize et"""
    print(f"\n{'=' * 70}")
    print(f"TRIAL #{trial.number}")
    print(f"{'=' * 70}")

    # Sadece LR önerisi
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    print(f"Learning Rate    : {lr:.2e} [OPTİMİZE]")
    print(f"Weight Decay     : {FIXED_WEIGHT_DECAY} [SABİT]")
    print(f"Clip Norm        : {FIXED_CLIPNORM} [SABİT]")
    print(f"Label Smoothing  : {FIXED_LABEL_SMOOTHING} [SABİT]")
    print(f"Warmup Steps     : {FIXED_WARMUP_STEPS} [SABİT]\n")

    # Memory cleanup
    tf.keras.backend.clear_session()
    gc.collect()

    try:
        with strategy.scope():
            model = return_model(model_arch)

            optimizer = tfa.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=FIXED_WEIGHT_DECAY,
                clipnorm=FIXED_CLIPNORM
            )

            loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=FIXED_LABEL_SMOOTHING)

            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )

        # Callbacks
        pruning_callback = TFKerasPruningCallback(trial, monitor='val_loss')

        scheduler = CosineScheduler(
            max_update=EPOCHS_TRIAL - 1,
            base_lr=lr,
            final_lr=lr / 10.0,
            warmup_steps=FIXED_WARMUP_STEPS,
            warmup_begin_lr=lr / 10.0
        )
        learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )

        # Training
        history = model.fit(
            x=train_dataset,
            epochs=EPOCHS_TRIAL,
            batch_size=batch_size,
            verbose=0,
            validation_data=val_dataset,
            callbacks=[pruning_callback, learning_rate_callback, early_stop]
        )

        best_val_loss = min(history.history['val_loss'])
        best_val_auc = max(history.history.get('val_auc', [0.0]))

        print(f"Trial #{trial.number} tamamlandı!")
        print(f"  Val Loss: {best_val_loss:.4f}")
        print(f"  Val AUC : {best_val_auc:.4f}")

        # Cleanup
        del model
        tf.keras.backend.clear_session()
        gc.collect()

        return best_val_loss

    except Exception as e:
        print(f"HATA! Trial #{trial.number}: {e}")
        tf.keras.backend.clear_session()
        gc.collect()
        return float('inf')


def run_optuna(n_trials=30):
    print("\n" + "=" * 70)
    print("OPTUNA LEARNING RATE OPTİMİZASYONU")
    print("=" * 70)
    print(f"Trial sayısı: {n_trials}")
    print(f"Epoch/trial : {EPOCHS_TRIAL}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name=f"LR_Opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    end_time = time.time()

    print("\n" + "=" * 70)
    print("OPTİMİZASYON TAMAMLANDI!")
    print("=" * 70)
    print(f"Süre        : {(end_time - start_time) / 60:.2f} dakika")
    print(f"En iyi trial: #{study.best_trial.number}")
    print(f"En iyi loss : {study.best_value:.4f}")
    print(f"OPTIMAL LR  : {study.best_params['learning_rate']:.6e}")
    print("=" * 70)

    # Sonuçları kaydet
    results = {
        'best_trial': study.best_trial.number,
        'best_val_loss': study.best_value,
        'optimal_learning_rate': study.best_params['learning_rate'],
        'fixed_hyperparameters': {
            'weight_decay': FIXED_WEIGHT_DECAY,
            'clipnorm': FIXED_CLIPNORM,
            'label_smoothing': FIXED_LABEL_SMOOTHING,
            'warmup_steps': FIXED_WARMUP_STEPS
        },
        'n_trials': n_trials,
        'epochs_per_trial': EPOCHS_TRIAL,
        'time_minutes': (end_time - start_time) / 60
    }

    with open(os.path.join(optuna_results_path, "optimal_lr.json"), 'w') as f:
        json.dump(results, f, indent=4)

    # Görselleştirme
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(optuna_results_path, "lr_history.png"))
        print("Grafik kaydedildi!")
    except:
        print("Grafik kaydedilemedi (plotly gerekli)")

    return study

def train_final_model(optimal_lr):
    print("\n" + "=" * 70)
    print(f"FİNAL MODEL EĞİTİMİ ({EPOCHS_FINAL} EPOCH)")
    print("=" * 70)
    print(f"Optimal LR      : {optimal_lr:.6e}")
    print(f"Weight Decay    : {FIXED_WEIGHT_DECAY}")
    print(f"Clip Norm       : {FIXED_CLIPNORM}")
    print(f"Label Smoothing : {FIXED_LABEL_SMOOTHING}")
    print(f"Warmup Steps    : {FIXED_WARMUP_STEPS}\n")

    tf.keras.backend.clear_session()
    gc.collect()

    with strategy.scope():
        model = return_model(model_arch)

        optimizer = tfa.optimizers.AdamW(
            learning_rate=optimal_lr,
            weight_decay=FIXED_WEIGHT_DECAY,
            clipnorm=FIXED_CLIPNORM
        )

        loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=FIXED_LABEL_SMOOTHING)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

    model.summary()

    # Callbacks
    scheduler = CosineScheduler(
        max_update=EPOCHS_FINAL - 1,
        base_lr=optimal_lr,
        final_lr=optimal_lr / 10.0,
        warmup_steps=FIXED_WARMUP_STEPS,
        warmup_begin_lr=optimal_lr / 10.0
    )
    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True
    )

    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    log_dir = path + "logs/fit/" + model_arch + "_lr_opt_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = os.path.join(model_path, f"{model_arch}_lr_opt_best.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    print("Eğitim başlıyor...")
    time1 = time.time()

    history = model.fit(
        x=train_dataset,
        epochs=EPOCHS_FINAL,
        batch_size=batch_size,
        verbose=2,
        validation_data=val_dataset,
        callbacks=[
            tensorboard_callback,
            early_stop,
            learning_rate_callback,
            reduce_on_plateau,
            checkpoint_callback
        ]
    )

    time2 = time.time()
    training_time = time2 - time1

    print(f"\nEğitim tamamlandı! Süre: {training_time / 60:.2f} dakika")

    # Model kaydet
    model.save(os.path.join(model_path, model_arch + "_lr_optimized"))
    print(f"Model kaydedildi: {model_arch}_lr_optimized")

    # Evaluate
    result_dic = model.evaluate(val_dataset, batch_size=batch_size, return_dict=True)
    result_dic["training_time"] = training_time
    result_dic["optimal_learning_rate"] = optimal_lr
    result_dic["epochs_trained"] = len(history.history['loss'])

    save_result(result_dic, result_path)

    print("\n" + "=" * 70)
    print("FİNAL SONUÇLAR")
    print("=" * 70)
    print(f"Val Loss    : {result_dic['loss']:.4f}")
    print(f"Val Accuracy: {result_dic['accuracy']:.4f}")
    print(f"Val AUC     : {result_dic['auc']:.4f}")
    print(f"Süre        : {training_time / 60:.2f} dakika")
    print(f"Epoch       : {result_dic['epochs_trained']}")
    print(f"Optimal LR  : {optimal_lr:.6e}")
    print("=" * 70)

    return model, history, result_dic


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEEPALLERGEN - LEARNING RATE OPTİMİZASYONU")
    print("=" * 70)
    print(f"Model      : {model_arch}")
    print(f"GPUs       : {n_hardwares}")
    print(f"Batch Size : {batch_size}")
    print("=" * 70)

    # ADIM 1: Learning Rate Bul
    n_trials = 30  # İsterseniz artırın
    study = run_optuna(n_trials=n_trials)

    # ADIM 2: Final Model Eğit
    optimal_lr = study.best_params['learning_rate']
    final_model, history, results = train_final_model(optimal_lr)

    print("\n" + "=" * 70)
    print("TAMAMLANDI!")
    print(f"Optimal LR: {optimal_lr:.6e}")
    print("=" * 70)