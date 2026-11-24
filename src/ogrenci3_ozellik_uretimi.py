"""
Öğrenci 3 - Spark ile Müşteri / Zaman Bazlı Davranış Özellikleri
================================================================

Girdi : data/scaled/fraud_<n>.parquet   (Öğrenci 1)
Çıktı : data/features/fraud_<n>.parquet (Öğrenci 2 -> dengeleme, Öğrenci 4 -> model)
        results/ogrenci3/feature_timing.csv, feature_by_class.csv, *.png

Window fonksiyonları (kart = cc_num, sıralama = unix_time). Hepsi SADECE GEÇMİŞE bakar:
bir işlemin özelliği, o işlemden sonraki işlemleri ya da herhangi bir etiketi görmez (sızıntı yok).

    txn_count_24h        : son 24 saatte aynı kartla yapılan ÖNCEKİ işlem sayısı
                           Window.rangeBetween(-86400, 0) üzerinde count - 1
    card_avg_amt         : kartın bu işlemden ÖNCEKİ tüm işlemlerinin ortalama tutarı
                           Window.rowsBetween(unboundedPreceding, -1); ilk işlemde = amt
    amt_dev_from_avg     : amt - card_avg_amt          (ortalamadan sapma, $)
    amt_ratio_to_avg     : amt / card_avg_amt          (kaç katı)
    log_secs_since_prev  : log1p(bir önceki işlemden bu yana geçen saniye), F.lag ile;
                           ilk işlemde 30 gün kabul edilir
Zaman / konum:
    hour, day_of_week, is_night (22:00-05:59), distance_km (müşteri-satıcı haversine),
    age, gender_m, log_amt

ÖNEMLİ: Bu adım dengelemeden ÖNCE ve orijinal (dengelenmemiş) veri üzerinde yapılır.
Oversample edilmiş satırlar txn_count_24h'yi şişirir, undersample gerçek geçmişi siler.

Kullanım:
    python src/ogrenci3_ozellik_uretimi.py                      # 100K, 1M, 5M
    python src/ogrenci3_ozellik_uretimi.py --sizes 100000 --force
    python src/ogrenci3_ozellik_uretimi.py --plot_only
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from pyspark.sql import DataFrame, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

LABEL = C.LABEL_COL
FIRST_TXN_GAP = 30 * 86400     # kartın ilk işleminde "önceki işlemden geçen süre"
TIMING_KEYS = ["size"]
TIMING_COLS = ["size", "rows", "cards", "feature_sec", "write_sec", "disk_mb", "finished_at"]
SUMMARY_FEATURES = ["amt", "is_night", "txn_count_24h", "card_avg_amt", "amt_dev_from_avg",
                    "amt_ratio_to_avg", "log_secs_since_prev", "distance_km", "age"]
log = None


def add_behavior_features(df: DataFrame) -> DataFrame:
    """Temel + window özelliklerini ekler. Girdi: ölçeklenmiş veri (common.SCALED_SCHEMA)."""
    df = C.add_basic_features(df)                  # age, hour, day_of_week, distance_km, gender_m
    by_card = Window.partitionBy("cc_num").orderBy("unix_time")
    last_24h = by_card.rangeBetween(-86400, 0)     # unix_time saniye cinsinden
    history = by_card.rowsBetween(Window.unboundedPreceding, -1)

    df = (df
          .withColumn("is_night", ((F.col("hour") >= 22) | (F.col("hour") < 6)).cast("double"))
          .withColumn("log_amt", F.log1p("amt"))
          # aynı saniyedeki eşzamanlı işlemler de "önceki" sayılır; işlemin kendisi çıkarılır
          .withColumn("txn_count_24h", (F.count("*").over(last_24h) - 1).cast("double"))
          .withColumn("_avg_prev", F.avg("amt").over(history))
          .withColumn("_prev_ts", F.lag("unix_time").over(by_card)))
    return (df
            .withColumn("card_avg_amt", F.coalesce("_avg_prev", "amt"))
            .withColumn("amt_dev_from_avg", F.col("amt") - F.col("card_avg_amt"))
            .withColumn("amt_ratio_to_avg", F.col("amt") / F.greatest("card_avg_amt", F.lit(0.01)))
            .withColumn("log_secs_since_prev",
                        F.log1p(F.greatest(F.lit(0.0),
                                           F.coalesce(F.col("unix_time") - F.col("_prev_ts"),
                                                      F.lit(FIRST_TXN_GAP)).cast("double"))))
            .drop("_avg_prev", "_prev_ts"))


def build_size(spark, size: int, out: Path, force: bool):
    path = C.features_path(size)
    done = C.done_set(out / "feature_timing.csv", TIMING_KEYS, ["write_sec"])
    if C.remove_incomplete(path):
        log.warning(f"  Yarım kalmış {path} silindi")
    if not force and C.is_complete(path) and (str(size),) in done:
        log.info(f"  {path.name} hazır -> atlandı")
        return
    src = C.scaled_path(size, "parquet")
    if not C.is_complete(src):
        raise FileNotFoundError(f"{src} yok/yarım. Önce: python src/ogrenci1_veri_hazirlama.py --sizes {size}")

    log.info(f"  BAŞLADI  {size:,} satır için window özellikleri")
    with C.Timer() as tf:
        feat = add_behavior_features(spark.read.parquet(str(src))).cache()
        n = feat.count()
    with C.Timer() as tw:
        (feat.repartition(max(1, math.ceil(size / 1_000_000)))
             .write.mode("overwrite").parquet(str(path)))
    cards = feat.select("cc_num").distinct().count()
    row = {"size": size, "rows": n, "cards": cards, "feature_sec": round(tf.sec, 2),
           "write_sec": round(tw.sec, 2), "disk_mb": round(C.dir_size_bytes(path) / 1e6, 2),
           "finished_at": C.now_iso()}
    C.append_row(row, out / "feature_timing.csv", TIMING_COLS)
    log.info(f"  BİTTİ    {n:,} satır, {cards} kart | özellik {tf.sec:.1f} sn, yazma {tw.sec:.1f} sn")

    # Sınıfa göre özellik ortalamaları: özellikler fraud'u ayırt ediyor mu?
    aggs = ([F.avg(c).alias(c) for c in SUMMARY_FEATURES]
            + [F.stddev(c).alias(f"{c}_std") for c in SUMMARY_FEATURES] + [F.count("*").alias("n")])
    pdf = feat.groupBy(LABEL).agg(*aggs).toPandas()
    pdf.insert(0, "size", size)
    import pandas as pd
    p = out / "feature_by_class.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["size"] != size]
    pd.concat([old, pdf], ignore_index=True).to_csv(p, index=False)
    feat.unpersist()


def plots(out: Path):
    import numpy as np
    import pandas as pd
    plt = C.plt_setup()
    p = out / "feature_by_class.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    sizes = sorted(d["size"].unique())

    # 1) Standartlaştırılmış ortalama farkı (Cohen's d): (ort_fraud - ort_normal) / birleşik std
    feats = SUMMARY_FEATURES
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=C.SURFACE)
    C.style_ax(ax, "Fraud ile normal arasındaki standartlaştırılmış fark (|d| büyük = ayırt edici)",
               xgrid=True, ygrid=False)
    y = np.arange(len(feats))
    h = 0.8 / len(sizes)
    for i, s in enumerate(sizes):
        sub = d[d["size"] == s].set_index(LABEL)
        dvals = []
        for f in feats:
            pooled = np.sqrt((sub.loc[0, f"{f}_std"] ** 2 + sub.loc[1, f"{f}_std"] ** 2) / 2)
            dvals.append((sub.loc[1, f] - sub.loc[0, f]) / pooled if pooled > 0 else np.nan)
        ax.barh(y + (i - (len(sizes) - 1) / 2) * h, dvals, h, color=C.SERIES[i % len(C.SERIES)],
                edgecolor=C.SURFACE, label=C.size_label(s))
    ax.axvline(0, color=C.INK2, lw=1)
    ax.set_yticks(y, feats)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d  (pozitif: fraud'da daha yüksek)", color=C.INK2)
    ax.legend(title="Veri boyutu", frameon=False, labelcolor=C.INK2)
    C.save_fig(fig, out / "feature_fraud_vs_normal.png")

    # 2) txn_count_24h boyuta göre: örnekleme/sentetik büyütme pencere sayımlarını değiştirir
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=C.SURFACE)
    C.style_ax(ax, "Son 24 saatteki işlem sayısı (ortalama) veri boyutuna göre")
    x = np.arange(len(sizes))
    for j, (lab, name) in enumerate(((0, "Normal"), (1, "Fraud"))):
        v = [d[(d["size"] == s) & (d[LABEL] == lab)]["txn_count_24h"].iloc[0] for s in sizes]
        bars = ax.bar(x + (j - 0.5) * 0.38, v, 0.38, color=C.SERIES[j], edgecolor=C.SURFACE, label=name)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, color=C.INK)
    ax.set_xticks(x, [C.size_label(s) for s in sizes])
    ax.legend(frameon=False, labelcolor=C.INK2)
    C.save_fig(fig, out / "txn_count_24h_by_size.png")
    log.info(f"Grafikler -> {out}")


def main():
    global log
    ap = argparse.ArgumentParser(description="Öğrenci 3 - window özellikleri")
    ap.add_argument("--sizes", type=int, nargs="+", default=C.DEFAULT_SIZES)
    ap.add_argument("--driver_memory", default="6g")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "ogrenci3"))
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else C.PROJECT_ROOT / out
    log = C.setup_logging(out, "ogrenci3")
    if args.plot_only:
        plots(out)
        return

    log.info(f"Öğrenci 3 başladı | sizes={args.sizes}")
    spark = C.get_spark("ogrenci3-ozellik", args.driver_memory)
    try:
        for size in sorted(args.sizes):
            log.info(f"===== {size:,} =====")
            build_size(spark, size, out, args.force)
        plots(out)
        log.info("Öğrenci 3 tamamlandı.")
    except BaseException as e:
        log.exception(f"Kesildi: {type(e).__name__}: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
