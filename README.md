# Kredi Kartı Dolandırıcılık Analizi

Büyük Veri Analitiği dersi grup projesi (Proje 4, 5 kişi) - PySpark.

**Problem:** Çok büyük işlem verisinde az sayıdaki fraud işleminin tespiti.
**Araştırma sorusu:** Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?

Teknik rapor: [docs/RAPOR.md](docs/RAPOR.md) · Mimari: [docs/mimari.md](docs/mimari.md) ·
Sınav soruları: [docs/sinav_sorulari.md](docs/sinav_sorulari.md)

## Veri kaynağı

Kaggle - [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(`kartik2112/fraud-detection`). Sparkov ile üretilmiş simülasyon verisi; 2019-2020, ~1.000 kart,
~800 satıcı. `fraudTrain.csv` (1.296.675 satır) + `fraudTest.csv` (555.719 satır), fraud oranı ~%0.52.

`fraudTrain.csv` ve `fraudTest.csv` dosyalarını `data/` klasörüne koyun (git'e eklenmez).

## Kurulum

Gereksinimler: Python 3.10+ (3.12 ile test edildi), **Java 17** (Spark 3.5 için), Windows'ta ayrıca **winutils**.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Windows'a özel:
1. Java 17: `winget install EclipseAdoptium.Temurin.17.JDK` (JAVA_HOME otomatik ayarlanır)
2. Spark'ın Windows'ta dosya yazabilmesi için `winutils.exe` ve `hadoop.dll` (Hadoop 3.3.x):
   `C:\hadoop\bin\` altına koyun, `HADOOP_HOME=C:\hadoop` yapın.
   (`src/common.py` JAVA_HOME / HADOOP_HOME tanımlı değilse bu standart yerlere kendisi bakar.)

## Çalıştırma

Akış sırayla: **Öğrenci 1 → 3 → 2 → 4 → 5**. Her adım bir öncekinin çıktısını kullanır.

```powershell
# Hepsi sırayla (100K, 1M, 5M) - kaldığı yerden devam eder
.\run_all.ps1

# Ya da tek tek
.\.venv\Scripts\python.exe src\ogrenci1_veri_hazirlama.py       # -> data/scaled/
.\.venv\Scripts\python.exe src\ogrenci3_ozellik_uretimi.py      # -> data/features/
.\.venv\Scripts\python.exe src\ogrenci2_class_imbalance.py      # -> data/balanced/
.\.venv\Scripts\python.exe src\ogrenci4_model.py --size 1000000
.\.venv\Scripts\python.exe src\ogrenci5_olcek_analizi.py
```

Ortak seçenekler (her script): `--sizes`, `--driver_memory 8g`, `--plot_only` (Spark başlatmadan,
kayıtlı CSV'lerden grafikleri yeniden üretir), `--force` (bitmiş adımları da yeniden çalıştırır).

### Uzun çalıştırmalar (5M) - VS Code kapansa da devam etsin

VS Code'un terminalinde başlatılan işlem VS Code kapanınca ölür. Bağımsız bir pencerede başlatın:

```powershell
# Proje klasöründe, herhangi bir PowerShell penceresinde:
Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$PWD\run_all.ps1"

# Sadece bir öğrencinin adımı, ör. Öğrenci 2 yalnız 5M:
Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$PWD\run_all.ps1",'-Steps','2','-Sizes','5000000'

# İlerlemeyi izle (Ctrl+C sadece izlemeyi bırakır, çalışmayı durdurmaz)
Get-Content results\run_all.log -Wait -Tail 30
```

CMD'den eşdeğeri: `start "fraud" /min powershell -NoProfile -ExecutionPolicy Bypass -File run_all.ps1`

Kesintiye dayanıklılık: her deney biter bitmez sonucu CSV'ye eklenir; Spark çıktıları `_SUCCESS`
dosyasıyla kontrol edilir, yarım kalan klasörler silinip yeniden üretilir. Aynı komutu tekrar
çalıştırmak sadece eksik kalanları yapar. Her scriptin zaman damgalı logu `results/ogrenciN/run.log`.

## Klasör yapısı

```
data/            fraudTrain.csv, fraudTest.csv (Kaggle)
  scaled/        Öğrenci 1: fraud_<n>.parquet / .csv (100K, 1M, 5M)
  features/      Öğrenci 3: window özellikli veri
  balanced/      Öğrenci 2: dengelenmiş train + gerçek dağılımlı test, SECILEN_YONTEM.json
src/
  common.py      ortak: Spark, okuma, ölçekleme, train/test ayrımı, log, grafik stili
  ogrenci1_veri_hazirlama.py   ogrenci2_class_imbalance.py   ogrenci3_ozellik_uretimi.py
  ogrenci4_model.py            ogrenci5_olcek_analizi.py
results/ogrenciN/  her öğrencinin CSV + PNG çıktıları ve run.log
docs/            rapor, mimari şema, sınav soruları
run_all.ps1      tüm akış
```

## Katkı tablosu

| Öğrenci | Görev | Kod | Çıktı |
|---|---|---|---|
| Öğrenci 1 | Veri hazırlama, dağılım incelemesi, 100K/1M/5M ölçekleme, CSV vs Parquet karşılaştırması | `src/ogrenci1_veri_hazirlama.py`, `src/common.py` | `data/scaled/`, `results/ogrenci1/` |
| Öğrenci 2 | Class imbalance analizi, train/test ayrımı, dengeleme (none / class_weight / undersample / oversample × 1:1, 1:10), yöntem seçimi | `src/ogrenci2_class_imbalance.py` | `data/balanced/`, `results/ogrenci2/` |
| Öğrenci 3 | Spark window fonksiyonlarıyla müşteri / zaman bazlı davranış özellikleri | `src/ogrenci3_ozellik_uretimi.py` | `data/features/`, `results/ogrenci3/` |
| Öğrenci 4 | LR, Random Forest, GBT; precision / recall / F1 / AUPRC, confusion matrix, PR eğrisi, eşik analizi | `src/ogrenci4_model.py` | `results/ogrenci4/` |
| Öğrenci 5 | Boyut ve dengesizliğin eğitim süresi, bellek, throughput ve performansa etkisi | `src/ogrenci5_olcek_analizi.py` | `results/ogrenci5/` |
| Hep birlikte | Mimari şema, teknik rapor, README, sınav soruları | `docs/` | |
