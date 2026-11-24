"""
Öğrenci 3 - Spark Window Fonksiyonlarıyla Müşteri / Zaman Bazlı Davranış Özellikleri
==================================================================================

Soru: Ham bir işlem satırı (kart, tutar, saat) tek başına az şey söyler. 450 dolar bir müşteri
için sıradan, başka biri için çok şüphelidir. Her işlemi AYNI KARTIN GEÇMİŞİYLE ve ZAMANLA
kıyaslayan sütunlar üretirsek model fraud'u daha iyi ayırt edebilir mi?

Girdi : data/scaled/fraud_<n>.parquet      (Öğrenci 1, şema common.SCALED_SCHEMA)
Çıktı : data/features/fraud_<n>.parquet    (Öğrenci 2 bunu --input auto ile kendisi algılar)
        Girdideki TÜM sütunlar (orig_id dahil) + common.MODEL_FEATURES. Satır sayısı değişmez.

Üretilen sütunlar (hepsi double, null yok)
------------------------------------------
common.add_basic_features : age, hour, day_of_week, distance_km, gender_m
Bu script (common.BEHAVIOR_FEATURES):
    log_amt             log1p(amt). Tutar çok sağa çarpık (çoğu < 100$, bazıları > 1000$);
                        log alınca LR gibi doğrusal modeller büyük değerlere boğulmaz.
    is_night            saat 22:00-05:59 ise 1.0. Öğrenci 1'in EDA'sında fraud gece yoğunlaşıyor.
    txn_count_24h       aynı kartın son 24 saatteki ÖNCEKİ işlem sayısı (bu işlem hariç).
                        Ele geçirilen kartla kısa sürede art arda işlem yapılır.
    card_avg_amt        kartın bu işlemden ÖNCEKİ işlemlerinin ortalama tutarı (müşterinin "normal"i).
                        İlk işlemde geçmiş yok -> amt (yani "kendisi kadar normal").
    amt_dev_from_avg    amt - card_avg_amt        (normalden kaç dolar sapma)
    amt_ratio_to_avg    amt / card_avg_amt        (normalin kaç katı)
    log_secs_since_prev log1p(önceki işlemden bu yana geçen saniye). İlk işlemde 30 gün kabul edilir.

Window (pencere) mantığı
------------------------
Window.partitionBy("cc_num").orderBy(ts): "her kartın işlemlerini ayrı grup yap, zamana göre sırala".
Sonra her satır için grubun SADECE GEÇMİŞ kısmına bakılır:
    rangeBetween(-86400, 0)                 son 86400 sn (24 saat) + bu satır -> count - 1
    rowsBetween(unboundedPreceding, -1)     baştan bir ÖNCEKİ satıra kadar (bu satır hariç)
    F.lag(ts)                               bir önceki satır
rows* satır sayar, range* sıralama sütununun DEĞERİNE (saniye) bakar.

Neden sadece geçmiş? (veri sızıntısı / data leakage)
    Kartın ortalamasını TÜM veriden alsaydık, Ocak'taki işlemin özelliği Aralık'taki işlemleri
    "görmüş" olurdu. Banka işlemi onaylarken geleceği bilemez; model testte iyi, sahada kötü çalışır.
    --leak_test bunu kanıtlar: veri zamanda kesilip özellikler yeniden hesaplanır; kesimden önceki
    satırların özellikleri değişmemelidir. Etiket (is_fraud) hiçbir özelliğin hesabına girmez.

Neden dengelemeden ÖNCE? (akış 1 -> 3 -> 2)
    Oversample ile çoğaltılan fraud satırları aynı kartın 24 saatlik sayımını şişirir; undersample
    ise kartın gerçek geçmişini siler. Özellikler orijinal dağılımda üretilir, dengelemeyi Öğrenci 2 yapar.

Çıktılar (results/ogrenci3/)
----------------------------
    feature_timing.csv     boyut başına: okuma+hesaplama süresi, yazma süresi, satır, disk MB
    feature_summary.csv    boyut x sınıf: özellik ortalamaları (+ ilk işlem payı, <10 dk payı)
    verification.csv       boyut başına: satır sayısı, orig_id, MODEL_FEATURES, null kontrolü
    leak_test.csv          sızıntı testi sonucu
    *.png                  grafikler
    run.log                zaman damgalı log

Kesintiye dayanıklılık: _SUCCESS'li çıktılar atlanır (--force ile yeniden), yarım kalan klasör
silinip yeniden üretilir, her ölçüm biter bitmez CSV'ye eklenir.

Kullanım:
    python src/ogrenci3_ozellik_uretimi.py                          # 100K, 1M, 5M
    python src/ogrenci3_ozellik_uretimi.py --sizes 5000000 --driver_memory 8g
    python src/ogrenci3_ozellik_uretimi.py --plot_only              # Spark'sız, CSV'lerden grafikler
    python src/ogrenci3_ozellik_uretimi.py --verify_only            # sadece kontroller + özet tablo
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from pyspark import StorageLevel  # noqa: E402
from pyspark.sql import DataFrame, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

LABEL = C.LABEL_COL
TS = "trans_date_trans_time"
FIRST_TXN_GAP_SEC = 30 * 24 * 3600     # ilk işlemde "önceki işlemden geçen süre" = 30 gün
DAY_SEC = 24 * 3600

BASIC_FEATURES = ["age", "hour", "day_of_week", "distance_km", "gender_m"]   # add_basic_features

TIMING_KEYS = ["size"]
TIMING_COLS = ["size", "rows", "fraud", "synthetic_rows", "cards", "compute_sec", "write_sec",
               "total_sec", "rows_per_sec", "n_files", "disk_mb", "finished_at"]
SUMMARY_KEYS = ["size", "is_fraud"]
VERIFY_KEYS = ["size"]
log = None


# --------------------------------------------------------------------------- #
# Özellikler
# --------------------------------------------------------------------------- #
def add_behavior_features(df: DataFrame) -> DataFrame:
    """common.BEHAVIOR_FEATURES sütunlarını ekler. Girdi: data/scaled şeması.

    Not (eşit zaman): Aynı kartta aynı saniyede iki işlem varsa sıralamayı belirlemek için
    orig_id / is_synthetic / amt de sıraya eklenir (her çalıştırmada aynı sonuç). 24 saat
    penceresi ise saniye değerine baktığı için aynı saniyedeki diğer işlemleri de "önceki" sayar.
    """
    ts_sec = F.col(TS).cast("long")
    df = df.withColumn("_ts", ts_sec)

    card = Window.partitionBy("cc_num").orderBy("_ts", C.ID_COL, "is_synthetic", "amt")
    past = card.rowsBetween(Window.unboundedPreceding, -1)
    last_24h = Window.partitionBy("cc_num").orderBy("_ts").rangeBetween(-DAY_SEC, 0)

    hour = F.hour(F.col(TS))
    avg_past = F.avg("amt").over(past)                       # ilk işlemde null
    gap = F.col("_ts") - F.lag("_ts").over(card)             # ilk işlemde null

    df = (df
          .withColumn("log_amt", F.log1p("amt"))
          .withColumn("is_night", ((hour >= 22) | (hour < 6)).cast("double"))
          .withColumn("txn_count_24h", (F.count(F.lit(1)).over(last_24h) - 1).cast("double"))
          .withColumn("card_avg_amt", F.coalesce(avg_past, F.col("amt")))
          .withColumn("_gap", F.coalesce(gap, F.lit(FIRST_TXN_GAP_SEC))))
    return (df
            .withColumn("amt_dev_from_avg", F.col("amt") - F.col("card_avg_amt"))
            .withColumn("amt_ratio_to_avg",
                        F.when(F.col("card_avg_amt") > 0, F.col("amt") / F.col("card_avg_amt"))
                         .otherwise(F.lit(1.0)))
            .withColumn("log_secs_since_prev", F.log1p(F.col("_gap").cast("double")))
            .drop("_ts", "_gap"))


def build_features(df: DataFrame) -> DataFrame:
    """Girdinin tüm sütunları + MODEL_FEATURES (hepsi double). Satır silinmez/eklenmez."""
    out = add_behavior_features(C.add_basic_features(df))
    for c in C.MODEL_FEATURES:
        if c in BASIC_FEATURES or c in C.BEHAVIOR_FEATURES:
            out = out.withColumn(c, F.col(c).cast("double"))
    return out.select(*df.columns, *[c for c in C.MODEL_FEATURES if c not in df.columns])


# --------------------------------------------------------------------------- #
# Üretim (boyut başına)
# --------------------------------------------------------------------------- #
def generate(spark, size: int, out: Path, force: bool) -> bool:
    """data/features/fraud_<size>.parquet üretir. Üretildiyse True, atlandıysa False."""
    in_path, out_path = C.scaled_path(size, "parquet"), C.features_path(size)
    if not C.is_complete(in_path):
        log.error(f"{in_path} yok/yarım. Önce Öğrenci 1: python src/ogrenci1_veri_hazirlama.py "
                  f"--sizes {size}  (ya da Drive'dan data/scaled/ klasörünü alın)")
        return False
    if C.is_complete(out_path) and not force:
        log.info(f"{C.size_label(size)}: {out_path} hazır -> atlandı (yeniden üretmek için --force)")
        return False
    if C.remove_incomplete(out_path):
        log.warning(f"{C.size_label(size)}: yarım kalmış {out_path} silindi")

    log.info(f"{C.size_label(size)}: özellikler hesaplanıyor ({in_path.name})")
    src = spark.read.parquet(str(in_path))

    # Hesaplama ile yazmayı ayrı ölçmek için: Spark tembeldir (lazy), hesap ancak bir 'action'
    # gelince yapılır. Önce persist + count -> okuma + window hesabı; sonra yazma.
    with C.Timer() as t_comp:
        feats = build_features(src).persist(StorageLevel.MEMORY_AND_DISK)
        stats = feats.agg(F.count(F.lit(1)).alias("rows"), F.sum(LABEL).alias("fraud"),
                          F.sum("is_synthetic").alias("synthetic"),
                          F.countDistinct("cc_num").alias("cards")).first()
    n_files = max(1, math.ceil(size / 1_000_000))            # Öğrenci 1 ile aynı dosya düzeni
    with C.Timer() as t_write:
        feats.repartition(n_files).write.mode("overwrite").parquet(str(out_path))
    feats.unpersist()

    rows = int(stats["rows"])
    total = t_comp.sec + t_write.sec
    row = {"size": size, "rows": rows, "fraud": int(stats["fraud"]),
           "synthetic_rows": int(stats["synthetic"] or 0), "cards": int(stats["cards"]),
           "compute_sec": round(t_comp.sec, 2), "write_sec": round(t_write.sec, 2),
           "total_sec": round(total, 2), "rows_per_sec": int(rows / total) if total else 0,
           "n_files": n_files, "disk_mb": round(C.dir_size_bytes(out_path) / 1e6, 2),
           "finished_at": C.now_iso()}
    C.append_row(row, out / "feature_timing.csv", TIMING_COLS)
    log.info(f"{C.size_label(size)}: yazıldı -> {out_path} | hesaplama {t_comp.sec:.1f} sn, "
             f"yazma {t_write.sec:.1f} sn, {row['disk_mb']} MB")
    return True


# --------------------------------------------------------------------------- #
# Doğrulama
# --------------------------------------------------------------------------- #
def verify(spark, size: int, out: Path) -> bool:
    """Arayüz kuralları: satır sayısı aynı mı, girdi sütunları (orig_id) korunmuş mu,
    MODEL_FEATURES var mı, double mı, null/NaN var mı?"""
    in_path, out_path = C.scaled_path(size, "parquet"), C.features_path(size)
    if not (C.is_complete(in_path) and C.is_complete(out_path)):
        log.warning(f"{C.size_label(size)}: doğrulama atlandı (girdi ya da çıktı eksik)")
        return False
    src, feats = spark.read.parquet(str(in_path)), spark.read.parquet(str(out_path))
    types = dict(feats.dtypes)
    missing_in = [c for c in src.columns if c not in types]
    missing_feat = [c for c in C.MODEL_FEATURES if c not in types]
    # Eklediğimiz sütunlar double olmalı; amt/city_pop girdiden olduğu gibi gelir (sayısal olması yeter)
    added = BASIC_FEATURES + C.BEHAVIOR_FEATURES
    not_double = [c for c in added if c in types and types[c] != "double"]
    not_double += [c for c in C.MODEL_FEATURES if c not in added and c in types
                   and types[c] not in ("double", "float", "int", "bigint")]
    present = [c for c in C.MODEL_FEATURES if c in types]
    bad = feats.select([F.sum((F.col(c).isNull() | F.isnan(c)).cast("int")).alias(c)
                        for c in present]).first().asDict()
    bad = {k: int(v) for k, v in bad.items() if v}
    n_in, n_out = src.count(), feats.count()
    ok = (n_in == n_out and not missing_in and not missing_feat and not not_double and not bad)
    row = {"size": size, "input_rows": n_in, "output_rows": n_out, "rows_equal": n_in == n_out,
           "orig_id_kept": C.ID_COL in types, "missing_input_cols": ";".join(missing_in),
           "missing_model_features": ";".join(missing_feat), "not_double": ";".join(not_double),
           "null_or_nan": ";".join(f"{k}={v}" for k, v in bad.items()), "ok": ok,
           "checked_at": C.now_iso()}
    C.merge_csv([row], out / "verification.csv", VERIFY_KEYS)
    (log.info if ok else log.error)(
        f"{C.size_label(size)}: doğrulama {'TAMAM' if ok else 'HATA'} | satır {n_in:,} -> {n_out:,}, "
        f"orig_id={'var' if row['orig_id_kept'] else 'YOK'}, eksik={missing_in + missing_feat}, "
        f"double olmayan={not_double}, null/NaN={bad or 'yok'}")
    return ok


def leak_test(spark, size: int, out: Path) -> bool:
    """Sızıntı testi: veriyi medyan zamanda kes, özellikleri yeniden hesapla.
    Özellikler yalnızca geçmişe bakıyorsa kesimden önceki satırlar BİREBİR aynı kalmalı.
    Satırlar (orig_id, is_synthetic, ts, amt) ile eşleştirilir."""
    in_path = C.scaled_path(size, "parquet")
    if not C.is_complete(in_path):
        return False
    src = spark.read.parquet(str(in_path))
    cut = src.select(F.percentile_approx(F.col(TS).cast("long"), 0.5)).first()[0]
    key = [C.ID_COL, "is_synthetic", TS, "amt"]
    full = build_features(src).select(*key, *C.BEHAVIOR_FEATURES)
    part = build_features(src.filter(F.col(TS).cast("long") < cut)).select(*key, *C.BEHAVIOR_FEATURES)
    j = part.alias("a").join(full.alias("b"), key)
    cond = " OR ".join(f"abs(a.{c} - b.{c}) > 1e-9" for c in C.BEHAVIOR_FEATURES)
    n_cmp, n_diff = j.count(), j.filter(cond).count()
    ok = n_diff == 0
    cut_str = datetime.fromtimestamp(cut, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    C.merge_csv([{"size": size, "cut_time_utc": cut_str, "compared_rows": n_cmp, "changed_rows": n_diff, "ok": ok, "checked_at": C.now_iso()}],
                out / "leak_test.csv", ["size"])
    (log.info if ok else log.error)(
        f"{C.size_label(size)}: sızıntı testi {'TAMAM' if ok else 'HATA'} | kesimden önceki "
        f"{n_cmp:,} satırın {n_diff} tanesinin özelliği değişti")
    return ok


# --------------------------------------------------------------------------- #
# Özet tablo: fraud vs normal, boyut başına (ölçek etkilerini tartışmak için)
# --------------------------------------------------------------------------- #
def summarize(spark, size: int, out: Path):
    out_path = C.features_path(size)
    if not C.is_complete(out_path):
        return
    feats = spark.read.parquet(str(out_path))
    aggs = [F.count(F.lit(1)).alias("rows")]
    aggs += [F.avg(c).alias(c) for c in C.BEHAVIOR_FEATURES]
    aggs += [F.expr("percentile_approx(txn_count_24h, 0.5)").cast("double").alias("txn_count_24h_median"),
             # geçmişi olmayan (kartın veri setindeki ilk) işlem payı
             (100 * F.avg((F.col("log_secs_since_prev") == math.log1p(FIRST_TXN_GAP_SEC)).cast("double")))
             .alias("first_txn_pct"),
             # önceki işlemle arasında 10 dakikadan az olan işlem payı (5M sentetik kopya etkisi)
             (100 * F.avg((F.col("log_secs_since_prev") < math.log1p(600)).cast("double")))
             .alias("prev_under_10min_pct"),
             (100 * F.avg("is_synthetic")).alias("synthetic_pct")]
    rows = []
    for r in C.to_pandas(feats.groupBy(LABEL).agg(*aggs)).to_dict("records"):
        row = {"size": size, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}}
        rows.append(row)
    C.merge_csv(rows, out / "feature_summary.csv", SUMMARY_KEYS)
    log.info(f"{C.size_label(size)}: fraud/normal özellik özeti yazıldı")


# --------------------------------------------------------------------------- #
# Grafikler
# --------------------------------------------------------------------------- #
PLOT_FEATURES = [
    ("txn_count_24h", "Son 24 saatteki önceki işlem sayısı (ort.)"),
    ("amt_ratio_to_avg", "Tutar / kartın geçmiş ortalaması (ort.)"),
    ("log_secs_since_prev", "log(1 + önceki işlemden geçen sn) (ort.)"),
    ("is_night", "Gece (22-06) işlem payı"),
]


def plot_summary(out: Path):
    import pandas as pd
    path = out / "feature_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).sort_values(["size", "is_fraud"])
    sizes = sorted(df["size"].unique())
    plt = C.plt_setup()
    fig, axes = plt.subplots(1, len(PLOT_FEATURES), figsize=(4.2 * len(PLOT_FEATURES), 3.8))
    w = 0.38
    for ax, (feat, title) in zip(axes, PLOT_FEATURES):
        for k, (cls, name) in enumerate([(0, "Normal"), (1, "Fraud")]):
            vals = [df[(df["size"] == s) & (df["is_fraud"] == cls)][feat].mean() for s in sizes]
            xs = [i + (k - 0.5) * w for i in range(len(sizes))]
            bars = ax.bar(xs, vals, w - 0.03, color=C.SERIES[k], label=name)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                            fontsize=8, color=C.INK2, xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([C.size_label(s) for s in sizes])
        C.style_ax(ax)
        ax.set_title(title, loc="left", color=C.INK, fontsize=10)
        ax.margins(y=0.15)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Davranış özellikleri: fraud ve normal işlemler, veri boyutuna göre",
                 x=0.01, ha="left", color=C.INK, fontsize=12, fontweight="bold")
    C.save_fig(fig, out / "feature_by_class.png")


def plot_timing(out: Path):
    path = out / "feature_timing.csv"
    if not path.exists():
        return
    df = C.read_rows(path, TIMING_KEYS, ["total_sec"], TIMING_COLS)
    if df.empty:
        return
    df["size"] = df["size"].astype(int)
    df = df.sort_values("size")
    plt = C.plt_setup()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    labels = [C.size_label(s) for s in df["size"]]
    ax.bar(labels, df["compute_sec"], color=C.SERIES[0], width=0.55, label="Okuma + window hesabı")
    ax.bar(labels, df["write_sec"], bottom=df["compute_sec"], color=C.SERIES[1], width=0.55,
           label="Parquet yazma", edgecolor=C.SURFACE, linewidth=2)
    for i, (tot, rps) in enumerate(zip(df["total_sec"], df["rows_per_sec"])):
        ax.annotate(f"{tot:.1f} sn\n{int(rps):,} satır/sn", (i, tot), ha="center", va="bottom",
                    fontsize=8, color=C.INK2, xytext=(0, 3), textcoords="offset points")
    C.style_ax(ax, "Özellik üretme süresi")
    ax.set_ylabel("Süre (sn)", color=C.INK2)
    ax.margins(y=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    C.save_fig(fig, out / "feature_timing.png")


# --------------------------------------------------------------------------- #
def main():
    global log
    ap = argparse.ArgumentParser(description="Öğrenci 3 - window ile davranış özellikleri")
    ap.add_argument("--sizes", type=int, nargs="+", default=C.DEFAULT_SIZES)
    ap.add_argument("--driver_memory", default="8g")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "ogrenci3"))
    ap.add_argument("--force", action="store_true", help="Hazır çıktıları da yeniden üret")
    ap.add_argument("--leak_test_size", type=int, default=None,
                    help="Sızıntı testinin yapılacağı boyut (varsayılan: en küçük boyut; 0 = yapma)")
    ap.add_argument("--plot_only", action="store_true", help="Spark başlatmadan CSV'lerden grafikler")
    ap.add_argument("--verify_only", action="store_true", help="Üretmeden: doğrulama + özet + grafik")
    args = ap.parse_args()

    out = Path(args.out)
    log = C.setup_logging(out, "ogrenci3")
    sizes = sorted(args.sizes)
    if args.plot_only:
        C.safe_plot(plot_summary, out, log=log.warning)
        C.safe_plot(plot_timing, out, log=log.warning)
        log.info(f"--plot_only: grafikler {out} altına yazıldı")
        return

    log.info(f"BAŞLADI sizes={sizes} driver_memory={args.driver_memory} pid={os.getpid()}")
    spark = C.get_spark("ogrenci3-ozellik", args.driver_memory)
    all_ok = True
    try:
        for size in sizes:
            if not args.verify_only:
                generate(spark, size, out, args.force)
            all_ok &= verify(spark, size, out)
            summarize(spark, size, out)
        leak_size = sizes[0] if args.leak_test_size is None else args.leak_test_size
        if leak_size:
            all_ok &= leak_test(spark, leak_size, out)
    finally:
        spark.stop()
    C.safe_plot(plot_summary, out, log=log.warning)
    C.safe_plot(plot_timing, out, log=log.warning)
    if all_ok:
        log.info("BİTTİ: tüm kontroller tamam. Öğrenci 2'yi aynı komutla tekrar çalıştırın "
                 "(log'da input=features görünmeli).")
    else:
        log.error("BİTTİ ama kontrollerden en az biri başarısız (yukarıdaki HATA satırlarına bakın)")
        sys.exit(1)


if __name__ == "__main__":
    main()
