"""
Öğrenci 1 - Veri Hazırlama, Dağılım İncelemesi ve Format Karşılaştırması
=======================================================================

Akış
----
a) fraudTrain.csv + fraudTest.csv birleştirilir; index, PII (first, last, street) ve trans_num atılır
   (common.read_raw).
b) Dağılım incelemesi (ham verinin tamamı, ~1.85M satır)       -> results/ogrenci1/eda_*.csv / *.png
     - genel sınıf dağılımı, tutar dağılımı (sınıfa göre), kategoriye göre fraud, saate göre fraud
c) 100K / 1M / 5M ölçekleri (fraud oranı korunur; 5M için gürültülü sentetik büyütme)
     -> data/scaled/fraud_<n>.parquet ve fraud_<n>.csv
     Önce Parquet yazılır, CSV bu Parquet'ten üretilir: iki format her zaman AYNI satırları içerir.
d) Format karşılaştırması: yazma süresi, disk boyutu, okuma süreleri (3 sorgu x 3 tekrar, medyan)
     -> results/ogrenci1/write_stats.csv, read_stats.csv, format_*.png

Kesintiye dayanıklı: _SUCCESS'li çıktı klasörleri ve CSV'deki ölçümler atlanır (--force ile yeniden).

Kullanım:
    python src/ogrenci1_veri_hazirlama.py                       # 100K, 1M, 5M
    python src/ogrenci1_veri_hazirlama.py --sizes 100000 --read_repeats 1
    python src/ogrenci1_veri_hazirlama.py --plot_only
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

LABEL = C.LABEL_COL
FORMATS = ["parquet", "csv"]
WRITE_KEYS, WRITE_COLS = ["size", "format"], ["size", "format", "rows", "fraud", "fraud_pct",
                                              "synthetic_rows", "n_files", "write_sec", "disk_mb",
                                              "finished_at"]
READ_KEYS = ["size", "format", "query"]
READ_COLS = READ_KEYS + ["median_sec", "min_sec", "max_sec", "repeats", "finished_at"]
QUERIES = {
    # Tüm sütunları okumak zorunda: format avantajı en az
    "full_scan": "tüm sütunlarda max() -> her sütun okunur",
    # Sadece 3 sütun: Parquet sütun bazlı okuma (column pruning) avantajı
    "column_agg": "category bazında avg(amt), sum(is_fraud) -> 3 sütun",
    # Filtre: Parquet predicate pushdown (row-group istatistikleri)
    "filter": "is_fraud=1 AND amt>500 satır sayısı",
}
log = None


# --------------------------------------------------------------------------- #
# b) Dağılım incelemesi
# --------------------------------------------------------------------------- #
def eda(spark, raw, out: Path):
    lg = raw.withColumn("hour", F.hour("trans_date_trans_time"))
    n0, n1 = C.class_counts(raw)
    pd_rows = [{"total": n0 + n1, "normal": n0, "fraud": n1, "fraud_pct": round(100 * n1 / (n0 + n1), 4),
                "imbalance_ratio": round(n0 / n1, 1), "cards": raw.select("cc_num").distinct().count(),
                "merchants": raw.select("merchant").distinct().count(),
                "min_date": str(raw.agg(F.min("trans_date_trans_time")).first()[0]),
                "max_date": str(raw.agg(F.max("trans_date_trans_time")).first()[0])}]
    import pandas as pd
    pd.DataFrame(pd_rows).to_csv(out / "eda_summary.csv", index=False)
    log.info(f"  Ham veri: {n0 + n1:,} satır | fraud {n1:,} (%{100 * n1 / (n0 + n1):.3f}) | "
             f"{pd_rows[0]['cards']} kart, {pd_rows[0]['merchants']} satıcı")

    # Tutar: sınıfa göre özet + log10 histogram
    q = C.to_pandas(raw.groupBy(LABEL).agg(F.count("*").alias("n"), F.avg("amt").alias("mean"),
                                           F.expr("percentile_approx(amt, 0.5)").alias("median"),
                                           F.expr("percentile_approx(amt, 0.95)").alias("p95"),
                                           F.max("amt").alias("max")).orderBy(LABEL))
    q.to_csv(out / "eda_amount_by_class.csv", index=False)
    C.to_pandas(raw.withColumn("bin", F.floor(F.log10(F.greatest("amt", F.lit(0.01))) * 10) / 10)
                   .groupBy("bin", LABEL).count().orderBy("bin")) \
        .to_csv(out / "eda_amount_hist.csv", index=False)

    for col, name in (("category", "eda_category.csv"), ("hour", "eda_hour.csv")):
        C.to_pandas(lg.groupBy(col).agg(F.count("*").alias("n"), F.sum(LABEL).alias("fraud"))
                      .withColumn("fraud_pct", F.round(100 * F.col("fraud") / F.col("n"), 4))
                      .orderBy(col)).to_csv(out / name, index=False)
    log.info("  EDA tabloları yazıldı")


def eda_plots(out: Path):
    import numpy as np
    import pandas as pd
    plt = C.plt_setup()

    # 1) Tutar dağılımı (log ölçek, sınıf içi yoğunluk)
    h = pd.read_csv(out / "eda_amount_hist.csv")
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.SURFACE)
    C.style_ax(ax, "İşlem tutarı dağılımı: fraud'lar belirgin şekilde daha yüksek tutarlarda")
    for lab, name, col in ((0, "Normal", C.SERIES[0]), (1, "Fraud", C.SERIES[1])):
        s = h[h[LABEL] == lab].sort_values("bin")
        ax.plot(10 ** s["bin"], s["count"] / s["count"].sum() * 100, color=col, lw=2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Tutar ($, log ölçek)", color=C.INK2)
    ax.set_ylabel("Sınıf içi pay (%)", color=C.INK2)
    ax.legend(frameon=False, labelcolor=C.INK2)
    C.save_fig(fig, out / "eda_amount_distribution.png")

    # 2) Kategoriye göre fraud oranı
    c = pd.read_csv(out / "eda_category.csv").sort_values("fraud_pct")
    overall = c["fraud"].sum() / c["n"].sum() * 100
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C.SURFACE)
    C.style_ax(ax, "Kategoriye göre fraud oranı", xgrid=True, ygrid=False)
    ax.barh(c["category"], c["fraud_pct"], color=C.SERIES[0], edgecolor=C.SURFACE)
    ax.axvline(overall, color=C.INK2, ls="--", lw=1)
    ax.text(overall, len(c) - 0.4, f" genel %{overall:.2f}", color=C.INK2, fontsize=9, va="top")
    ax.set_xlabel("Fraud oranı (%)", color=C.INK2)
    C.save_fig(fig, out / "eda_fraud_by_category.png")

    # 3) Saate göre fraud oranı + işlem hacmi
    hr = pd.read_csv(out / "eda_hour.csv").sort_values("hour")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C.SURFACE)
    C.style_ax(ax, "Saate göre fraud oranı: gece 22:00-03:00 arası belirgin şekilde yüksek")
    ax.bar(hr["hour"], hr["fraud_pct"], color=C.SERIES[1], edgecolor=C.SURFACE)
    ax.set_xticks(np.arange(24))
    ax.set_xlabel("Saat", color=C.INK2)
    ax.set_ylabel("Fraud oranı (%)", color=C.INK2)
    ax2 = ax.twinx()
    ax2.plot(hr["hour"], hr["n"] / 1000, color=C.SERIES[0], lw=2, marker="o", ms=3)
    ax2.set_ylim(0, hr["n"].max() / 1000 * 1.1)   # 0'dan başlasın: hacim farkı abartılı görünmesin
    ax2.set_ylabel("İşlem sayısı (bin)", color=C.SERIES[0])
    ax2.tick_params(colors=C.INK2, labelsize=9)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    C.save_fig(fig, out / "eda_fraud_by_hour.png")


# --------------------------------------------------------------------------- #
# c) Ölçekleme + yazma
# --------------------------------------------------------------------------- #
def build_size(spark, size: int, raw_holder: dict, out: Path, force: bool):
    wpath = out / "write_stats.csv"
    done = C.done_set(wpath, WRITE_KEYS, ["write_sec"])
    pq = C.scaled_path(size, "parquet")
    for fmt in FORMATS:
        p = C.scaled_path(size, fmt)
        if C.remove_incomplete(p):
            log.warning(f"  Yarım kalmış {p} silindi")
    if force:
        done = set()

    # Parquet: kaynak (ölçeklenmiş) veri
    if not C.is_complete(pq) or (str(size), "parquet") not in done:
        if "raw" not in raw_holder:
            raw_holder["raw"] = C.read_raw(spark).cache()
        log.info(f"  [parquet] BAŞLADI  {size:,} satır ölçekleniyor")
        df = C.scale_dataset(raw_holder["raw"], size).cache()
        df.count()   # ölçekleme süresi yazma süresine karışmasın
        with C.Timer() as t:
            C.write_scaled(df, size, "parquet")
        df.unpersist()
        record_write(spark, size, "parquet", t.sec, wpath)

    # CSV: Parquet'ten üretilir (aynı satırlar)
    cp = C.scaled_path(size, "csv")
    if not C.is_complete(cp) or (str(size), "csv") not in done:
        log.info(f"  [csv] BAŞLADI  Parquet'ten CSV yazılıyor")
        src = C.load_scaled(spark, size, "parquet").cache()
        src.count()
        with C.Timer() as t:
            C.write_scaled(src, size, "csv")
        src.unpersist()
        record_write(spark, size, "csv", t.sec, wpath)


def record_write(spark, size, fmt, sec, wpath):
    p = C.scaled_path(size, fmt)
    df = C.load_scaled(spark, size, fmt)
    r = df.agg(F.count("*").alias("n"), F.sum(LABEL).alias("f"), F.sum("is_synthetic").alias("s")).first()
    n_files = len([f for f in p.iterdir() if f.name.startswith("part-")])
    row = {"size": size, "format": fmt, "rows": r["n"], "fraud": r["f"],
           "fraud_pct": round(100 * r["f"] / r["n"], 4), "synthetic_rows": r["s"], "n_files": n_files,
           "write_sec": round(sec, 2), "disk_mb": round(C.dir_size_bytes(p) / 1e6, 2),
           "finished_at": C.now_iso()}
    C.append_row(row, wpath, WRITE_COLS)
    log.info(f"  [{fmt}] BİTTİ    {r['n']:,} satır, fraud %{row['fraud_pct']:.3f}, sentetik {r['s']:,} | "
             f"yazma {sec:.1f} sn | {row['disk_mb']:.1f} MB")


# --------------------------------------------------------------------------- #
# d) Okuma karşılaştırması
# --------------------------------------------------------------------------- #
def run_query(spark, size, fmt, query):
    df = C.load_scaled(spark, size, fmt)   # her tekrarda yeni okuma (dosya listeleme dahil)
    if query == "full_scan":
        return df.agg(*[F.max(c).alias(c) for c in df.columns]).collect()
    if query == "column_agg":
        return df.groupBy("category").agg(F.avg("amt"), F.sum(LABEL)).collect()
    if query == "filter":
        return df.filter((F.col(LABEL) == 1) & (F.col("amt") > 500)).count()
    raise ValueError(query)


def read_bench(spark, size, out: Path, repeats: int, force: bool):
    rpath = out / "read_stats.csv"
    done = set() if force else C.done_set(rpath, READ_KEYS, ["median_sec"])
    for fmt in FORMATS:
        for q in QUERIES:
            if (str(size), fmt, q) in done:
                continue
            run_query(spark, size, fmt, q)   # ısınma (JIT, dosya önbelleği) - ölçülmez
            secs = []
            for _ in range(repeats):
                with C.Timer() as t:
                    run_query(spark, size, fmt, q)
                secs.append(t.sec)
            row = {"size": size, "format": fmt, "query": q, "median_sec": round(statistics.median(secs), 3),
                   "min_sec": round(min(secs), 3), "max_sec": round(max(secs), 3), "repeats": repeats,
                   "finished_at": C.now_iso()}
            C.append_row(row, rpath, READ_COLS)
            log.info(f"  okuma {fmt:7s} {q:10s}: medyan {row['median_sec']:.2f} sn")


def format_plots(out: Path):
    import numpy as np
    import pandas as pd
    plt = C.plt_setup()
    w = C.read_rows(out / "write_stats.csv", WRITE_KEYS, ["write_sec"])
    r = C.read_rows(out / "read_stats.csv", READ_KEYS, ["median_sec"])
    if w.empty:
        return
    w["size"] = w["size"].astype(int)
    sizes = sorted(w["size"].unique())
    x = np.arange(len(sizes))
    cols = {"parquet": C.SERIES[0], "csv": C.SERIES[1]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=C.SURFACE)
    for ax, (metric, title, unit) in zip(axes, [("disk_mb", "Disk boyutu", "MB"),
                                                ("write_sec", "Yazma süresi", "sn")]):
        C.style_ax(ax, title)
        for j, fmt in enumerate(FORMATS):
            s = w[w["format"] == fmt].set_index("size").reindex(sizes)[metric]
            bars = ax.bar(x + (j - 0.5) * 0.38, s, 0.38, color=cols[fmt], edgecolor=C.SURFACE, label=fmt)
            for b, v in zip(bars, s):
                if pd.notna(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}", ha="center", va="bottom",
                            fontsize=8, color=C.INK)
        ax.set_xticks(x, [C.size_label(s) for s in sizes])
        ax.set_ylabel(unit, color=C.INK2)
    axes[0].legend(frameon=False, labelcolor=C.INK2)
    C.save_fig(fig, out / "format_disk_write.png")

    if r.empty:
        return
    r["size"] = r["size"].astype(int)
    fig, axes = plt.subplots(1, len(QUERIES), figsize=(15, 4.8), facecolor=C.SURFACE)
    for ax, q in zip(axes, QUERIES):
        C.style_ax(ax, q)
        sub = r[r["query"] == q]
        for j, fmt in enumerate(FORMATS):
            s = sub[sub["format"] == fmt].assign(size=lambda d: d["size"].astype(int)) \
                .set_index("size").reindex(sizes)["median_sec"]
            ax.bar(x + (j - 0.5) * 0.38, s, 0.38, color=cols[fmt], edgecolor=C.SURFACE, label=fmt)
        ax.set_xticks(x, [C.size_label(s) for s in sizes])
        ax.set_ylabel("Okuma süresi (sn, medyan)", color=C.INK2)
        ax.text(0, -0.2, QUERIES[q], transform=ax.transAxes, fontsize=8, color=C.INK2)
    axes[0].legend(frameon=False, labelcolor=C.INK2)
    C.save_fig(fig, out / "format_read_times.png")
    log.info(f"[d] Format grafikleri -> {out}")


# --------------------------------------------------------------------------- #
def main():
    global log
    ap = argparse.ArgumentParser(description="Öğrenci 1 - veri hazırlama ve format karşılaştırması")
    ap.add_argument("--sizes", type=int, nargs="+", default=C.DEFAULT_SIZES)
    ap.add_argument("--driver_memory", default="6g")
    ap.add_argument("--read_repeats", type=int, default=3)
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "ogrenci1"))
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Mevcut çıktıları yeniden üret/ölç")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else C.PROJECT_ROOT / out
    log = C.setup_logging(out, "ogrenci1")

    if args.plot_only:
        eda_plots(out)
        format_plots(out)
        return

    log.info(f"Öğrenci 1 başladı | sizes={args.sizes}")
    spark = C.get_spark("ogrenci1-veri-hazirlama", args.driver_memory)
    holder: dict = {}
    try:
        if args.force or not (out / "eda_hour.csv").exists():
            log.info("[b] Dağılım incelemesi BAŞLADI")
            holder["raw"] = C.read_raw(spark).cache()
            eda(spark, holder["raw"], out)
        else:
            log.info("[b] EDA tabloları mevcut -> atlandı")
        C.safe_plot(eda_plots, out, log=log.warning)
        for size in sorted(args.sizes):
            log.info(f"===== [c] {size:,} =====")
            build_size(spark, size, holder, out, args.force)
        if "raw" in holder:
            holder["raw"].unpersist()
        for size in sorted(args.sizes):
            log.info(f"===== [d] okuma karşılaştırması {size:,} =====")
            read_bench(spark, size, out, args.read_repeats, args.force)
        C.safe_plot(format_plots, out, log=log.warning)
        log.info("Öğrenci 1 tamamlandı.")
    except BaseException as e:
        log.exception(f"Kesildi: {type(e).__name__}: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
