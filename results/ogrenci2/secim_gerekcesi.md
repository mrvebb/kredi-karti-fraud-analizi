# Öğrenci 2 - Dengeleme yöntemi seçimi (5M, girdi: scaled)

Seçim ölçütü: **F1** (test seti gerçek dağılımda, eşik 0.5). F1, precision ile recall'u birlikte dikkate aldığı için hem kaçırılan fraud'u hem de yanlış alarmı cezalandırır. En iyi değere 0.005 puandan yakın yöntemler eşit sayılır ve aralarından **en kısa sürede eğitilen** seçilir (gürültü düzeyindeki bir fark için daha pahalı yöntemi seçmemek için).

| Yöntem | Train satırı | Precision | Recall | F1 | AUPRC | Eğitim (sn) |
|---|---:|---:|---:|---:|---:|---:|
| oversample 1:10 | 4,375,493 | 0.349 | 0.517 | 0.417 | 0.223 | 5.2 |
| class_weight 1:10 | 3,999,034 | 0.349 | 0.517 | 0.417 | 0.223 | 3.5 |
| undersample 1:10 **(seçilen)** | 227,491 | 0.345 | 0.522 | 0.415 | 0.222 | 1.9 |
| undersample 1:1 | 41,608 | 0.034 | 0.744 | 0.065 | 0.154 | 3.1 |
| oversample 1:1 | 7,955,514 | 0.033 | 0.746 | 0.064 | 0.157 | 10.1 |
| class_weight 1:1 | 3,999,034 | 0.033 | 0.746 | 0.064 | 0.157 | 5.1 |
| none - | 3,999,034 | 0.051 | 0.004 | 0.007 | 0.243 | 4.3 |

## Gerekçe

- **Dengeleme yokken** accuracy %99.4 ama recall yalnızca %0.4: test setindeki fraud'ların kaçırılan payı %99.6. Yüksek accuracy, modelin neredeyse her şeye 'normal' demesinden geliyor.
- **Seçilen yöntem (undersample 1:10)** ile recall %0.4 → %52.2, precision %5.1 → %34.5, F1 0.007 → 0.415.
- **Eşitlik:** F1 açısından 3 yöntem birbirine 0.005 puandan yakın. Seçilen undersample 1:10 (0.415, 1.9 sn, 227,491 satır) bunların en hızlı eğitileni. Diğerleri: class_weight 1:10 (0.417, 3.5 sn, 3,999,034 satır), oversample 1:10 (0.417, 5.2 sn, 4,375,493 satır).
- En yüksek recall **class_weight 1:1** ile (%74.6), ama precision %3.3: fraud alarmlarında yanlış alarm payı %96.7. Bu yüzden tek başına recall'a göre seçilmedi.
- AUPRC yöntemler arasında 0.154–0.243 aralığında değişiyor; seçilen yöntemde 0.222.
- **AUPRC en yüksek dengeleme yokken** (0.243). Yani dengeleme modelin fraud'ları normallerden ayırma (sıralama) yeteneğini artırmıyor; sadece 0.5 eşiğinin nereye düştüğünü değiştiriyor. Baseline'ın recall'unun sıfıra yakın olması bilgisizlikten değil, eşiğin dengesiz veri için fazla yüksek kalmasından. Dengelemenin pratik faydası, eşik ayarı yapmadan da kullanılabilir bir precision/recall dengesi vermesi.
- **Maliyet:** seçilen yöntemin train seti baseline'ın 0.06 katı, eğitim süresi 0.45 katı.
- Diğer boyutlarda seçilen yöntemin F1 sırası: 100K: 2/6, 1M: 3/6, 5M: 3/6.

Dengeleme sadece train setine uygulandı; test seti gerçek dağılımda bırakıldı. Dengelenmiş veri: `data/balanced/` (bkz. SECILEN_YONTEM.json).
