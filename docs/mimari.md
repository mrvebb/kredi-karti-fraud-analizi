# Mimari

Tüm işlem tek makinede **PySpark (local[*])** ile yapılır; driver ve executor aynı JVM'dedir.
Her adım bir sonrakine diskteki Parquet dosyalarıyla veri aktarır. Böylece adımlar birbirinden
bağımsız çalıştırılabilir ve kesintiden sonra kaldığı yerden devam edilebilir.

```mermaid
flowchart LR
    K[("Kaggle CSV<br/>fraudTrain + fraudTest<br/>1.85M satır")] --> O1

    subgraph O1["Öğrenci 1 - Veri hazırlama"]
        direction TB
        A1["Birleştir, PII at<br/>(first, last, street, trans_num)"] --> A2["EDA: tutar, kategori, saat"]
        A2 --> A3["Ölçekle: 100K / 1M / 5M<br/>fraud oranı korunur<br/>5M: gürültülü sentetik büyütme"]
        A3 --> A4["Yaz: Parquet + CSV<br/>okuma/yazma/disk karşılaştırması"]
    end

    O1 -->|data/scaled/*.parquet| O3

    subgraph O3["Öğrenci 3 - Özellik üretimi"]
        direction TB
        B1["Window: partitionBy(cc_num)<br/>orderBy(unix_time)"] --> B2["txn_count_24h, card_avg_amt,<br/>amt_dev / ratio, secs_since_prev"]
        B2 --> B3["hour, is_night, distance_km, age"]
    end

    O3 -->|data/features/*.parquet| O2

    subgraph O2["Öğrenci 2 - Class imbalance"]
        direction TB
        C1["Train/test ayrımı<br/>orig_id hash, %80/%20"] --> C2["Dengele (sadece train):<br/>none, class_weight,<br/>undersample, oversample × 1:1, 1:10"]
        C2 --> C3["LR ile karşılaştır,<br/>en iyi yöntemi seç"]
    end

    O2 -->|data/balanced/<br/>train_n*, test_n*| O4
    O2 -->|data/balanced/| O5
    O2 -.->|imbalance_results.csv| O5

    subgraph O4["Öğrenci 4 - Model"]
        D1["LR, Random Forest, GBT"] --> D2["P / R / F1 / AUPRC<br/>confusion matrix, PR eğrisi,<br/>eşik analizi"]
    end

    subgraph O5["Öğrenci 5 - Ölçek analizi"]
        E1["100K / 1M / 5M ×<br/>orijinal / dengelenmiş"] --> E2["süre, throughput,<br/>tepe heap, performans"]
    end

    O4 --> R["Rapor<br/>docs/RAPOR.md"]
    O5 --> R
    O1 --> R
```

## Tasarım kararları

| Karar | Neden |
|---|---|
| Ara veriler Parquet | Sütun bazlı okuma, sıkıştırma ve şema korunur; CSV'ye göre hem daha küçük hem daha hızlı (Öğrenci 1 ölçümleri). |
| Train/test ayrımı `orig_id` hash'i ile | 5M'deki sentetik kopyalar orijinal satırla aynı tarafa düşer: test setine "train'deki satırın gürültülü kopyası" sızmaz. Hash etiketten bağımsız olduğu için sınıf oranı korunur. |
| Window özellikleri dengelemeden önce | Oversample ile çoğaltılan satırlar aynı kartın 24 saatlik işlem sayısını şişirir, undersample gerçek geçmişi siler. |
| Window'lar sadece geçmişe bakar | `rowsBetween(unboundedPreceding, -1)`, `rangeBetween(-86400, 0)`, `lag`: bir işlemin özelliği gelecekteki işlemleri görmez. Bu, gerçek zamanlı skorlamada da hesaplanabilir. |
| Özellik pipeline'ı (StringIndexer, OneHot, StandardScaler) sadece train'e fit | Test istatistikleri modele sızmaz. |
| Dengeleme sadece train'e | Test gerçek dağılımda kalır; metrikler sahadaki performansı yansıtır. |
| Sonuçlar satır satır CSV'ye, çıktılar `_SUCCESS` kontrolü ile | Uzun 5M çalıştırmaları kesilirse sadece o anki deney kaybolur; tekrar çalıştırınca eksikler tamamlanır. |
