# Öğrenci 5 — Ölçeklenebilirlik bulguları

Yalnızca üç geçerli ölçümü tamamlanan kombinasyonlar gösterilir; süreler ve metrikler medyandır.

## Veri büyüklüğü

### LR

| Boyut | Train | Test | Eğitim (sn) | Tahmin (sn) | RSS tepe (MB) | Satır/sn | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100K | 4568 | 20126 | 7.088 | 1.149 | 1885.5 | 17509.2 | 0.4609 | 0.2955 |
| 1M | 45390 | 200255 | 9.057 | 2.194 | 2374.9 | 91269.6 | 0.4881 | 0.2823 |
| 5M | 227437 | 1000966 | 11.351 | 5.371 | 7855.0 | 186353.4 | 0.4675 | 0.2878 |

100K → 5M: başlangıç verisi 50.00 kat, eğitim süresi 1.60 kat, tahmin süresi 4.67 kat, tepe RSS 4.17 kat; F1 değişimi +0.0066, AUPRC değişimi -0.0077.

### RF

| Boyut | Train | Test | Eğitim (sn) | Tahmin (sn) | RSS tepe (MB) | Satır/sn | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100K | 4568 | 20126 | 12.615 | 2.197 | 1922.4 | 9161.4 | 0.5363 | 0.7425 |
| 1M | 45390 | 200255 | 41.654 | 9.949 | 3232.5 | 20128.0 | 0.6644 | 0.8142 |
| 5M | 227437 | 1000966 | 167.236 | 32.910 | 6859.3 | 30414.8 | 0.6692 | 0.8287 |

100K → 5M: başlangıç verisi 50.00 kat, eğitim süresi 13.26 kat, tahmin süresi 14.98 kat, tepe RSS 3.57 kat; F1 değişimi +0.1329, AUPRC değişimi +0.0863.

## Dengesizlik oranı

- 100K LR: tamamlanan oranlar arasında en yüksek F1 1:10 (0.4911); AUPRC 0.2802.
- 100K RF: tamamlanan oranlar arasında en yüksek F1 original (0.7882); AUPRC 0.8067.
- 1M LR: tamamlanan oranlar arasında en yüksek F1 1:10 (0.4815); AUPRC 0.2879.
- 1M RF: tamamlanan oranlar arasında en yüksek F1 original (0.8032); AUPRC 0.8418.
- 5M LR: tamamlanan oranlar arasında en yüksek F1 1:50 (0.4775); AUPRC 0.3805.
- 5M RF: tamamlanan oranlar arasında en yüksek F1 original (0.8037); AUPRC 0.8374.

## Öğrenci 2 ile bağlam karşılaştırması

- 100K: Öğrenci 2 en yüksek F1 = 0.5075 (oversample, 1:10).
- 1M: Öğrenci 2 en yüksek F1 = 0.4881 (undersample, 1:10).
- 5M: Öğrenci 2 en yüksek F1 = 0.4683 (oversample, 1:10).

## Sınırlamalar

- Başlangıç büyüklüğü, gerçek eğitim satırı değildir; dengelenmiş train daha küçüktür.
- Kaggle kaynağı simülasyondur; 5M büyüklüğünde ek sentetik kopyalar bağımsız yeni gözlemler değildir. Boyutlar arasında davranış özellikleri ve test kümeleri de değişir.
- Oran deneyinde undersampling eğitim satırı sayısını da değiştirir; test her boyut içinde sabittir. Rastgele örneklemede hedef oran yaklaşık gerçekleşir.
- Okuma süresi, seçilen sütunların ilk cache+count ile okunmasını; hazırlama süresi split, örnekleme, repartition ve cache+count işlemlerini içerir. Eğitim özellik pipeline fit ve model fit; tahmin pipeline transform ve tahmin cache+count süresidir. F1/AUPRC hesabı ayrı ölçülür.
- RSS 0,2 sn örneklemeli JVM fiziksel belleğidir; tüm sistem belleği veya gerçek mutlak tepe garantisi değildir. JVM önceki ayırdığı belleği tutabilir.
- Colab oturumu ve disk önbelleği süreyi etkileyebilir. Farklı ortamların süreleri doğrudan eşdeğer sayılmaz; Öğrenci 2 süreleri bu nedenle kullanılmaz.
- Modellerin ayarları ve eşik sabittir; amaç en iyi modeli seçmek değildir.
- Öğrenci 1 CSV'sine göre en büyük boyutta ek sentetik payı: %62.95.


## Genel sonuç

- Daha fazla veri, özellikle Random Forest modelinde
  dolandırıcılık tespit başarısını artırabilmektedir.
  Ancak eğitim süresini ve hesaplama maliyetini de artırmaktadır.
- RF için 1:50 oranı, orijinal veriye yakın başarı sağlamıştır.
- Veri miktarı, model başarısı ve hesaplama maliyeti
  arasında uygun bir denge kurulmalıdır.

## Çalışma ortamı farklılığı

- İlk 29 deney: Intel Xeon, PySpark 3.5.5.
- Son deney (5M RF original): AMD EPYC, PySpark 4.0.3.
- Bu nedenle son deneyin süre ve bellek sonuçları
  diğer deneylerle doğrudan eşdeğer kabul edilmemelidir.
