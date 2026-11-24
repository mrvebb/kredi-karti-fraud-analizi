# Sınav soruları - Öğrenci 1 ve Öğrenci 2

Sayılar `results/ogrenci1/` ve `results/ogrenci2/` altındaki çıktılardan alınmıştır
(Öğrenci 2 sonuçları: `input=features`, Öğrenci 3'ün 14 özelliğiyle).

## Öğrenci 1 - Veri hazırlama ve format karşılaştırması

**1. 5M veri setini büyütürken sentetik kopyalara neden orijinal satırın `orig_id`'sini verdik?**

Train/test ayrımı `orig_id` hash'ine göre yapılıyor (`common.group_split`); bu sayede sentetik
kopya ile orijinali hep aynı tarafa düşüyor. Rastgele bölseydik modelin train'de gördüğü satırın
neredeyse aynısı testte karşısına çıkardı. Bu veri sızıntısı (data leakage) test sonuçlarını
olduğundan iyi gösterirdi.

**2. Parquet 5M'de filtre sorgusunda CSV'den yaklaşık 10 kat hızlı (0,11 sn / 1,03 sn). Peki 1M'de
tüm sütunları okurken neden CSV daha hızlı çıktı (0,67 sn / 1,17 sn)?**

Parquet sütun bazlıdır: sadece gereken sütunları okur ve filtreyi dosyadaki istatistiklerle okuma
aşamasına indirir (predicate pushdown). 1M Parquet ise tek dosya ve tek blok olduğu için Spark onu
tek çekirdekle okudu; 203 MB'lık CSV iki parçaya bölünüp paralel okundu. Format kadar dosya ve
partition düzeni de performansı belirler.

## Öğrenci 2 - Sınıf dengesizliği ve dengeleme

**3. Dengeleme yokken model 5M'de %99,6 accuracy alıyor. Bu neden yanıltıcı ve hangi metrikleri
kullandınız?**

Fraud oranı %0,52; her işleme "normal" diyen bir model bile %99,5 accuracy alır. Modelin recall'u
%28,9, yani fraud'ların %71'i kaçıyor. Bu yüzden precision, recall, F1 ve AUPRC kullandık;
dengeleme yöntemini F1'e göre seçtik (undersample 1:10, F1 0,468).

**4. Dengelemeyi neden sadece train setine uyguladınız ve davranış özellikleri neden dengelemeden
önce hesaplanmalı?**

Test seti, modelin sahada karşılaşacağı gerçek dağılımı temsil etmeli; dengelenmiş bir testte
precision olduğundan yüksek çıkar. Özellikler ise orijinal veride hesaplanmalı: oversample'ın
çoğalttığı satırlar "son 24 saatteki işlem sayısı" gibi sayımları şişirir, undersample kartın
gerçek geçmişini siler. Bu yüzden akış Öğrenci 1 → 3 → 2.
