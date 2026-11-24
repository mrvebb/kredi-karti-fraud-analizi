# Öğrenci 2 - Dengeleme yöntemi seçimi (5M, girdi: features)

Seçim ölçütü: **F1** (test seti gerçek dağılımda, eşik 0.5). F1, precision ile recall'u birlikte dikkate aldığı için hem kaçırılan fraud'u hem de yanlış alarmı cezalandırır. En iyi değere 0.005 puandan yakın yöntemler eşit sayılır ve aralarından **en kısa sürede eğitilen** seçilir (gürültü düzeyindeki bir fark için daha pahalı yöntemi seçmemek için).

| Yöntem | Train satırı | Precision | Recall | F1 | AUPRC | Eğitim (sn) |
|---|---:|---:|---:|---:|---:|---:|
| oversample 1:10 | 4,374,097 | 0.391 | 0.585 | 0.468 | 0.287 | 69.5 |
| class_weight 1:10 | 3,999,034 | 0.390 | 0.586 | 0.468 | 0.287 | 37.3 |
| undersample 1:10 **(seçilen)** | 227,437 | 0.389 | 0.585 | 0.468 | 0.288 | 24.1 |
| none - | 3,999,034 | 0.873 | 0.289 | 0.435 | 0.448 | 47.0 |
| undersample 1:1 | 41,794 | 0.056 | 0.902 | 0.106 | 0.239 | 40.6 |
| class_weight 1:1 | 3,999,034 | 0.056 | 0.904 | 0.106 | 0.240 | 45.2 |
| oversample 1:1 | 7,958,070 | 0.056 | 0.904 | 0.106 | 0.239 | 204.8 |

## Gerekçe

- **Dengeleme yokken** accuracy %99.6 ama recall yalnızca %28.9: test setindeki fraud'ların kaçırılan payı %71.1. Yüksek accuracy, modelin neredeyse her şeye 'normal' demesinden geliyor.
- **Seçilen yöntem (undersample 1:10)** ile recall %28.9 → %58.5, precision %87.3 → %38.9, F1 0.435 → 0.468.
- **Eşitlik:** F1 açısından 3 yöntem birbirine 0.005 puandan yakın. Seçilen undersample 1:10 (0.468, 24.1 sn, 227,437 satır) bunların en hızlı eğitileni. Diğerleri: class_weight 1:10 (0.468, 37.3 sn, 3,999,034 satır), oversample 1:10 (0.468, 69.5 sn, 4,374,097 satır).
- En yüksek recall **class_weight 1:1** ile (%90.4), ama precision %5.6: fraud alarmlarında yanlış alarm payı %94.4. Bu yüzden tek başına recall'a göre seçilmedi.
- AUPRC yöntemler arasında 0.239–0.448 aralığında değişiyor; seçilen yöntemde 0.288.
- **AUPRC en yüksek dengeleme yokken** (0.448). Yani dengeleme modelin fraud'ları normallerden ayırma (sıralama) yeteneğini artırmıyor; sadece 0.5 eşiğinin nereye düştüğünü değiştiriyor. Baseline'ın recall'unun düşük (%28.9) olması bilgisizlikten değil, eşiğin dengesiz veri için fazla yüksek kalmasından. Dengelemenin pratik faydası, eşik ayarı yapmadan da kullanılabilir bir precision/recall dengesi vermesi.
- **Maliyet:** seçilen yöntemin train seti baseline'ın 0.06 katı, eğitim süresi 0.51 katı.
- Diğer boyutlarda seçilen yöntemin F1 sırası: 100K: 3/6, 1M: 1/6, 5M: 3/6.

Dengeleme sadece train setine uygulandı; test seti gerçek dağılımda bırakıldı. Dengelenmiş veri: `data/balanced/` (bkz. SECILEN_YONTEM.json).
