def dosya_oku(dosya_yolu: object):
    """
    Belirtilen path'den dosya okur ve içeriğini döndürür.

    Args:
        dosya_yolu (str): Okunacak dosyanın tam yolu

    Returns:
        str: Dosya içeriği veya hata mesajı
    """
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            icerik = dosya.read()
            return icerik
    except FileNotFoundError:
        return f"Hata: '{dosya_yolu}' dosyası bulunamadı."
    except PermissionError:
        return f"Hata: '{dosya_yolu}' dosyasına erişim izni yok."
    except UnicodeDecodeError:
        # UTF-8 ile okunamazsa, farklı encoding'ler dene
        try:
            with open(dosya_yolu, 'r', encoding='windows-1254') as dosya:
                return dosya.read()
        except:
            return f"Hata: '{dosya_yolu}' dosyası okunamadı (encoding sorunu)."
    except Exception as e:
        return f"Beklenmeyen hata: {str(e)}"


def loss_degerlerini_cikart(dosya_yolu):
    """
    Dosyadan loss ve val_loss değerlerini çıkarır.

    Args:
        dosya_yolu (str): Log dosyasının yolu

    Returns:
        tuple: (loss_listesi, val_loss_listesi)
    """
    import re

    loss_listesi = []
    val_loss_listesi = []

    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            for satir in dosya:
                # loss değerini bul (val_loss değil, sadece loss)
                loss_match = re.search(r'(?<!val_)loss:\s*([\d.]+)', satir)
                if loss_match:
                    loss_degeri = float(loss_match.group(1))
                    loss_listesi.append(loss_degeri)
                    print(f"Loss bulundu: {loss_degeri}")

                # val_loss değerini bul
                val_loss_match = re.search(r'val_loss:\s*([\d.]+)', satir)
                if val_loss_match:
                    val_loss_degeri = float(val_loss_match.group(1))
                    val_loss_listesi.append(val_loss_degeri)
                    print(f"Val_loss bulundu: {val_loss_degeri}")

    except Exception as e:
        print(f"Hata: {str(e)}")
        return [], []

    return loss_listesi, val_loss_listesi


def dosya_oku_satir_satir(dosya_yolu):
    """
    Dosyayı satır satır okur ve liste olarak döndürür.
    Büyük dosyalar için daha verimli.

    Args:
        dosya_yolu (str): Okunacak dosyanın tam yolu

    Returns:
        list: Dosya satırları listesi veya None (hata durumunda)
    """
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as dosya:
            satirlar = dosya.readlines()
            # Satır sonlarını temizle
            satirlar = [satir.strip() for satir in satirlar]
            return satirlar
    except Exception as e:
        print(f"Hata: {str(e)}")
        return None


def grafik_olustur(loss_listesi, val_loss_listesi):
    """
    Loss ve val_loss değerlerinden grafik oluşturur.

    Args:
        loss_listesi (list): Training loss değerleri
        val_loss_listesi (list): Validation loss değerleri
    """
    try:
        import matplotlib.pyplot as plt

        # Grafik boyutunu ayarla
        plt.figure(figsize=(12, 8))

        # Epoch sayılarını oluştur
        epochs_loss = range(1, len(loss_listesi) + 1)
        epochs_val_loss = range(1, len(val_loss_listesi) + 1)

        # Grafikleri çiz
        plt.plot(epochs_loss, loss_listesi, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
        plt.plot(epochs_val_loss, val_loss_listesi, 'r-', linewidth=2, label='Validation Loss', marker='s',
                 markersize=4)

        # Grafik özelliklerini ayarla
        plt.title('Training ve Validation Loss Değişimi', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Y ekseni için logaritmik ölçek (loss değerleri çok küçükse)
        plt.yscale('linear')  # 'log' yapabilirsin eğer değerler çok küçükse

        # Grafik aralığını otomatik ayarla
        plt.tight_layout()

        # Grafiği göster
        plt.show()

        # Grafiği kaydet
        plt.savefig('loss_grafigi.jpg', dpi=300, bbox_inches='tight')
        print("\nGrafik 'loss_grafigi.jpg' olarak kaydedildi!")

    except ImportError:
        print("\nHata: matplotlib kütüphanesi bulunamadı!")
        print("Kurulum için: pip install matplotlib")
    except Exception as e:
        print(f"\nGrafik oluştururken hata: {str(e)}")


# Kullanım örnekleri
if __name__ == "__main__":
    # Loss değerlerini çıkar
    dosya_yolu = input("Log dosyasının yolunu girin: ")

    print("=== LOSS VE VAL_LOSS DEĞERLERİ ÇIKARTILIYOR ===\n")
    loss_listesi, val_loss_listesi = loss_degerlerini_cikart(dosya_yolu)

    print(f"\n=== SONUÇLAR ===")
    print(f"Toplam {len(loss_listesi)} adet loss değeri bulundu:")
    print(f"Loss listesi: {loss_listesi}")

    print(f"\nToplam {len(val_loss_listesi)} adet val_loss değeri bulundu:")
    print(f"Val_loss listesi: {val_loss_listesi}")

    # Grafik oluştur (eğer veriler varsa)
    if loss_listesi or val_loss_listesi:
        print("\n=== GRAFİK OLUŞTURULUYOR ===")
        grafik_olustur(loss_listesi, val_loss_listesi)
    else:
        print("\nGrafik oluşturulamadı: Hiç loss değeri bulunamadı!")

    # Örnek kullanım için:
    # dosya_yolu = "training_log.txt"