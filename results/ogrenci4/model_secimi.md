# Öğrenci 4 - Model karşılaştırması (5M, girdi: features)

Train: Öğrenci 2'nin seçtiği **undersample 1:10** ile dengelenmiş veri (181,625 satır, 16,620 fraud). Test: gerçek dağılım (1,000,966 satır, 5,338 fraud, %0.5). Eşik, train'den ayrılan doğrulama kümesinde gerçek dağılıma göre düzeltilmiş F1'i en yüksek yapan değer; test seti eşik seçiminde kullanılmadı.

| Model | Eşik | Accuracy | Precision | Recall | F1 | AUPRC | AUROC | TP | FP | FN | Eğitim (sn) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naif: hep normal | – | 0.9947 | 0 | 0 | 0 | – | – | 0 | 0 | 5,338 | – |
| GBT | 0.500 | 0.9958 | 0.568 | 0.922 | 0.703 | 0.895 | 0.998 | 4,921 | 3,749 | 417 | 164.7 |
| GBT | 0.951 | 0.9984 | 0.929 | 0.763 | 0.838 | 0.895 | 0.998 | 4,075 | 313 | 1,263 | 164.7 |
| Logistic Regression | 0.500 | 0.9929 | 0.388 | 0.581 | 0.465 | 0.289 | 0.946 | 3,103 | 4,902 | 2,235 | 1.5 |
| Logistic Regression | 0.656 | 0.9938 | 0.426 | 0.496 | 0.459 | 0.289 | 0.946 | 2,650 | 3,566 | 2,688 | 1.5 |
| LR + özellik müh. | 0.500 | 0.9915 | 0.375 | 0.889 | 0.527 | 0.774 | 0.996 | 4,745 | 7,924 | 593 | 9.2 |
| LR + özellik müh. | 0.918 | 0.9972 | 0.752 | 0.704 | 0.727 | 0.774 | 0.996 | 3,758 | 1,239 | 1,580 | 9.2 |
| Random Forest | 0.500 | 0.9960 | 0.578 | 0.904 | 0.705 | 0.884 | 0.996 | 4,827 | 3,528 | 511 | 284.5 |
| Random Forest | 0.825 | 0.9982 | 0.883 | 0.763 | 0.819 | 0.884 | 0.996 | 4,072 | 538 | 1,266 | 284.5 |

## Yorum

- **Accuracy yanıltıcı:** hiçbir fraud'u yakalamayan naif model accuracy %99.5 alıyor. Bu yüzden karşılaştırmada AUPRC, F1, precision ve recall kullanıldı.
- **En iyi model: GBT** (AUPRC 0.895; ayarlı eşikte precision 0.929, recall 0.763, F1 0.838). Test setindeki 5,338 fraud'un 4,075 tanesi yakalandı, 313 yanlış alarm verildi.
- Random Forest: AUPRC 0.884, F1 0.819 (eğitim 284.5 sn).
- LR + özellik müh.: AUPRC 0.774, F1 0.727 (eğitim 9.2 sn).
- Logistic Regression: AUPRC 0.289, F1 0.459 (eğitim 1.5 sn).
- **LR'yi zayıflatan şey modelin kendisi değil, özellikler:** aynı Logistic Regression'a tutar, saat, yaş vb. 20 aralığa bölünmüş (one-hot) ve log_amt x kategori etkileşimi verilince AUPRC 0.289 → 0.774, F1 0.459 → 0.727. Doğrusal model bir sütuna tek ağırlık verir, yani 'tutar arttıkça risk hep artar (ya da hep azalır)' varsayar. Oysa fraud oranı tutarla inişli çıkışlı değişir (Sparkov 1M: 10-20$ arasında %0.7, 50-100$ arasında %0.01, 800-1000$ arasında %53, 1000$ üstünde %22). Böyle bir ilişkiyi LR ancak her aralığa ayrı ağırlık verilince öğrenebilir; ağaç modelleri bunu kendiliğinden yapar.
- **Eşik etkisi (GBT):** 0.5'te P=0.568 R=0.922 F1=0.703 (FP 3,749); eşik 0.951 iken P=0.929 R=0.763 F1=0.838 (FP 313). Dengelenmiş veriyle eğitilen model fraud olasılığını gerçekte olduğundan yüksek verir; eşiği yükseltmek yanlış alarmları azaltır. AUPRC eşikten bağımsızdır, değişmez.
- **Eşik etkisi (Random Forest):** 0.5'te P=0.578 R=0.904 F1=0.705 (FP 3,528); eşik 0.825 iken P=0.883 R=0.763 F1=0.819 (FP 538). Dengelenmiş veriyle eğitilen model fraud olasılığını gerçekte olduğundan yüksek verir; eşiği yükseltmek yanlış alarmları azaltır. AUPRC eşikten bağımsızdır, değişmez.
- **Eşik etkisi (LR + özellik müh.):** 0.5'te P=0.375 R=0.889 F1=0.527 (FP 7,924); eşik 0.918 iken P=0.752 R=0.704 F1=0.727 (FP 1,239). Dengelenmiş veriyle eğitilen model fraud olasılığını gerçekte olduğundan yüksek verir; eşiği yükseltmek yanlış alarmları azaltır. AUPRC eşikten bağımsızdır, değişmez.
- **Eşik etkisi (Logistic Regression):** 0.5'te P=0.388 R=0.581 F1=0.465 (FP 4,902); eşik 0.656 iken P=0.426 R=0.496 F1=0.459 (FP 3,566). Bu modelde eşik ayarı belirgin bir kazanç sağlamadı: sorun eşikte değil, modelin fraud'u normalden ayırma gücünde (düşük AUPRC). AUPRC eşikten bağımsızdır, değişmez.
- **GBT en önemli 5 özellik:** `amt` (0.33), `log_amt` (0.09), `age` (0.08), `hour` (0.06), `is_night` (0.05). Bunlardan 2 tanesi Öğrenci 3'ün ürettiği özellik.
- **Random Forest en önemli 5 özellik:** `amt` (0.21), `log_amt` (0.20), `amt_dev_from_avg` (0.14), `amt_ratio_to_avg` (0.09), `is_night` (0.06). Bunlardan 4 tanesi Öğrenci 3'ün ürettiği özellik.
- **Veri boyutuna göre (GBT):** 100K: AUPRC 0.524, 11.6 sn, 1M: AUPRC 0.890, 109.9 sn, 5M: AUPRC 0.895, 164.7 sn. (Boyut etkisinin ayrıntılı analizi: Öğrenci 5.)

## Ayar araması (hiperparametre)

RF ve GBT için birkaç ayar denendi; her biri train'den ayrılan doğrulama kümesinde, gerçek dağılıma göre düzeltilmiş AUPRC ile değerlendirildi ve en iyisi seçildi. Test seti bu seçimde kullanılmadı.

| Model | Ayar | Doğrulama AUPRC | Eğitim (sn) |
|---|---|---:|---:|
| Random Forest | numTrees=100, maxDepth=10 | 0.818 | 41.5 |
| Random Forest | numTrees=100, maxDepth=15 | 0.890 | 106.6 |
| Random Forest | numTrees=200, maxDepth=10 | 0.819 | 133.3 |
| Random Forest | numTrees=200, maxDepth=15 **(seçilen)** | 0.893 | 284.5 |
| GBT | maxIter=50, maxDepth=5 | 0.844 | 19.9 |
| GBT | maxIter=50, maxDepth=7 | 0.891 | 30.0 |
| GBT | maxIter=100, maxDepth=5 | 0.879 | 41.9 |
| GBT | maxIter=100, maxDepth=7 | 0.896 | 67.1 |
| GBT | maxIter=200, maxDepth=5 | 0.897 | 92.2 |
| GBT | maxIter=200, maxDepth=7 **(seçilen)** | 0.903 | 164.7 |
- Random Forest: varsayılan ayar (numTrees=100, maxDepth=10) doğrulama AUPRC 0.818 → seçilen (numTrees=200, maxDepth=15) 0.893; eğitim süresi 41 → 285 sn.
  En iyi ayar denenen aralığın en ucunda; daha fazla ağaç/derinlik biraz daha kazanç sağlayabilir ama eğitim süresi de artar.
- GBT: varsayılan ayar (maxIter=50, maxDepth=5) doğrulama AUPRC 0.844 → seçilen (maxIter=200, maxDepth=7) 0.903; eğitim süresi 20 → 165 sn.
  En iyi ayar denenen aralığın en ucunda; daha fazla ağaç/derinlik biraz daha kazanç sağlayabilir ama eğitim süresi de artar.
- Not: 100K veride doğrulama kümesi çok küçük (87 fraud); orada seçilen ayar şansa bağlı olabilir. Ayar sonuçlarının güvenilir olduğu boyutlar 1M ve 5M.

Grafikler: `metrics_by_model.png`, `pr_curves.png`, `threshold_analysis.png`, `confusion_matrices.png`, `feature_importance.png`, `training_time.png`.
