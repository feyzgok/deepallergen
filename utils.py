import csv
import math
import os


## learning rate code adopted from https://d2l.ai/chapter_optimization/lr-scheduler.html
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.001, final_lr=0.0001,
                 warmup_steps=10, warmup_begin_lr=0.0001):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
        # Hata düzeltildi: base_lr'yi başlangıçta tanımlıyoruz
        self.base_lr = base_lr

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                   * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        # Hata düzeltildi: epoch > max_update durumu için kontrol eklendi
        elif epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                    self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
            return self.base_lr
        else:
            # max_update'den sonra final_lr'de sabit kalır
            return self.final_lr


def save_result(result_dic, path=''):
    """
    Sonuçları CSV dosyasına kaydet

    Args:
        result_dic (dict): Kaydedilecek metriklerin dictionary'si
        path (str): Dosya yolu (varsayılan: mevcut dizin)
    """
    # Hata düzeltildi: Path kontrolü ve dosya yolu birleştirme
    if path and not path.endswith('/'):
        path += '/'

    # Dizin yoksa oluştur
    if path and not os.path.exists(path):
        os.makedirs(path)

    filepath = path + "evaluation_metrics.csv"

    try:
        with open(filepath, "w", newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, result_dic.keys())
            writer.writeheader()
            writer.writerow(result_dic)
        print(f"Sonuçlar başarıyla kaydedildi: {filepath}")
    except Exception as e:
        print(f"Dosya kaydetme hatası: {e}")


# Test fonksiyonu (isteğe bağlı)
def test_scheduler():
    """Scheduler'ı test etmek için örnek fonksiyon"""
    scheduler = CosineScheduler(max_update=100, base_lr=0.001, final_lr=0.0001, warmup_steps=10)

    # Örnek epochs için learning rate'leri yazdır
    test_epochs = [0, 5, 10, 20, 50, 100, 110]
    for epoch in test_epochs:
        lr = scheduler(epoch)
        print(f"Epoch {epoch}: Learning Rate = {lr:.6f}")


if __name__ == "__main__":
    # Test için
    test_scheduler()

    # Save result test
    sample_results = {
        "accuracy": 0.95,
        "loss": 0.123,
        "precision": 0.94,
        "recall": 0.96
    }
    save_result(sample_results, "results/")