# Kredi Kartı Dolandırıcılık Analizi

Büyük Veri Analitiği dersi grup projesi (Proje 4, 5 kişi) - PySpark.

**Problem:** Çok büyük işlem verisinde az sayıdaki fraud işleminin tespiti.
**Araştırma sorusu:** Çok veri olması, nadir fraud olaylarını bulmayı gerçekten kolaylaştırıyor mu?

Mimari: [docs/mimari.md](docs/mimari.md) · Teknik rapor ve sınav soruları: hazırlanıyor (`docs/`)

## Veri kaynağı

Kaggle - [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(`kartik2112/fraud-detection`). Sparkov ile üretilmiş simülasyon verisi; 2019-2020, 999 kart,
693 satıcı. `fraudTrain.csv` (1.296.675 satır) + `fraudTest.csv` (555.719 satır) = 1.852.394 satır,
9.651 fraud (%0.521).

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

Akış sırayla: **Öğrenci 1 → 3 → 2 → 4 → 5**. Her adım bir öncekinin çıktısını kullanır
(dosya adları ve sütunlar için aşağıdaki **Arayüz** bölümüne bakın).

```powershell
# Öğrenci 1 -> 3 -> 2 sırayla (100K, 1M, 5M) - kaldığı yerden devam eder
.\run_all.ps1

# Ya da tek tek
.\.venv\Scripts\python.exe src\ogrenci1_veri_hazirlama.py       # -> data/scaled/
.\.venv\Scripts\python.exe src\ogrenci3_ozellik_uretimi.py     # -> data/features/
.\.venv\Scripts\python.exe src\ogrenci2_class_imbalance.py      # -> data/balanced/
```

Öğrenci 2 varsayılan olarak `--input auto` ile çalışır: Öğrenci 3'ün `data/features/` çıktısı
hazırsa onu, değilse Öğrenci 1'in `data/scaled/` verisini temel özelliklerle kullanır. Öğrenci 3
teslim edince aynı komutu tekrar çalıştırmak yeterli: deneyler özellikli veriyle yeniden yapılır,
önceki sonuçlar CSV'de `input=scaled` olarak kalır.

Ortak seçenekler: `--sizes`, `--driver_memory 8g`, `--plot_only` (Spark başlatmadan,
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

## Arayüz (öğrenciler arası veri teslimi)

Herkes kendi kodunu yazar; adımlar birbirine sadece aşağıdaki dosyalarla bağlıdır.
Spark çıktı klasöründe `_SUCCESS` dosyası yoksa yazma yarıda kalmıştır, kullanmayın.

| Dosya | Üreten | Kullanan | İçerik |
|---|---|---|---|
| `data/scaled/fraud_<n>.parquet` (+ `.csv`) | Öğrenci 1 | Öğrenci 3 | n = 100000, 1000000, 5000000. Şema `common.SCALED_SCHEMA`: PII atılmış ham sütunlar + `orig_id` (orijinal satır kimliği; sentetik kopyalarda aynı) + `is_synthetic`. Fraud oranı her boyutta ~%0.52. |
| `data/features/fraud_<n>.parquet` | **Öğrenci 3** | Öğrenci 2 | `data/scaled` sütunlarının hepsi (**`orig_id` korunmalı**) + `common.MODEL_FEATURES`: `age, hour, day_of_week, distance_km, gender_m` (`common.add_basic_features`) ve `log_amt, is_night, txn_count_24h, card_avg_amt, amt_dev_from_avg, amt_ratio_to_avg, log_secs_since_prev`. Window'lar sadece geçmişe bakmalı. Sütun adları farklı olacaksa `common.BEHAVIOR_FEATURES` güncellenmeli. |
| `data/balanced/train_n<n>.parquet` | Öğrenci 2 | Öğrenci 4, 5 | Seçilen yöntemle dengelenmiş train (sadece train dengelendi). `class_weight` seçildiyse `class_weight` sütunu var: modelde `weightCol` olarak verin. |
| `data/balanced/test_n<n>.parquet` | Öğrenci 2 | Öğrenci 4, 5 | Gerçek dağılımda test. Dengelemeyin. |
| `data/balanced/SECILEN_YONTEM.json` | Öğrenci 2 | Öğrenci 4, 5 | Yöntem, oran, `weight_col`, `feature_cols` (modele girecek sayısal sütunlar; `category` ayrıca one-hot), boyut başına dosya yolları. |
| `results/ogrenci2/imbalance_results.csv` | Öğrenci 2 | Öğrenci 5 | Her (girdi, boyut, yöntem, oran) için train satır sayısı, eğitim süresi, P/R/F1/AUPRC. Dengesizliğin süreye etkisi için. |

Train/test ayrımı `common.group_split` ile `orig_id` hash'ine göre yapılır: 5M'deki sentetik
kopyalar orijinal satırla aynı tarafta kalır (test'e sızma olmaz). Kendi ayrımınızı yapacaksanız
da bu fonksiyonu kullanın.

## Klasör yapısı

```
data/            fraudTrain.csv, fraudTest.csv (Kaggle)
  scaled/        Öğrenci 1: fraud_<n>.parquet / .csv (100K, 1M, 5M)
  features/      Öğrenci 3: window özellikli veri
  balanced/      Öğrenci 2: dengelenmiş train + gerçek dağılımlı test, SECILEN_YONTEM.json
src/
  common.py                    ortak: Spark, okuma, ölçekleme, train/test ayrımı, log, grafik stili
  ogrenci1_veri_hazirlama.py   ogrenci3_ozellik_uretimi.py   ogrenci2_class_imbalance.py
notebooks/       keşif notebook'ları (Colab)
results/ogrenciN/  her öğrencinin CSV + PNG çıktıları ve run.log
docs/            rapor, mimari şema, sınav soruları
run_all.ps1      Öğrenci 1 -> 3 -> 2 akışı
```

## Katkı tablosu

| Öğrenci | Görev | Kod | Çıktı |
|---|---|---|---|
| Öğrenci 1 | Veri hazırlama, dağılım incelemesi, 100K/1M/5M ölçekleme, CSV vs Parquet karşılaştırması | `src/ogrenci1_veri_hazirlama.py`, `src/common.py` | `data/scaled/`, `results/ogrenci1/` |
| Öğrenci 2 | Class imbalance analizi, train/test ayrımı, dengeleme (none / class_weight / undersample / oversample × 1:1, 1:10), gerekçeli yöntem seçimi | `src/ogrenci2_class_imbalance.py` | `data/balanced/`, `results/ogrenci2/` (seçim gerekçesi: `secim_gerekcesi.md`) |
| Öğrenci 3 | Spark window fonksiyonlarıyla müşteri / zaman bazlı davranış özellikleri (sadece geçmişe bakan pencereler, sızıntı testi), süre ölçümü, fraud/normal özellik karşılaştırması | `src/ogrenci3_ozellik_uretimi.py` | `data/features/`, `results/ogrenci3/` |
| Öğrenci 4 | LR, Random Forest, GBT; precision / recall / F1 / AUPRC, confusion matrix, PR eğrisi, eşik analizi | | `results/ogrenci4/` |
| Öğrenci 5 | Boyut ve dengesizliğin eğitim süresi, bellek, throughput ve performansa etkisi | | `results/ogrenci5/` |
| Hep birlikte | Mimari şema, teknik rapor, README, sınav soruları | `docs/` | |
