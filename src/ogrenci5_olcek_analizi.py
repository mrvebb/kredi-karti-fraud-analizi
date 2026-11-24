"""
Öğrenci 5 - Veri Büyüklüğü ve Sınıf Dengesizliğinin Süre / Performansa Etkisi
=============================================================================

Soru: Veri büyüdükçe (100K -> 1M -> 5M) eğitim süresi, bellek ve throughput nasıl değişiyor;
      performans (fraud yakalama) gerçekten artıyor mu? Dengeleme süreyi nasıl etkiliyor?

Girdi : data/features/fraud_<n>.parquet (Öğrenci 3)  -> "original": dengelenmemiş train
        data/balanced/train_n<n>.parquet  (Öğrenci 2)  -> "balanced": seçilen yöntemle dengelenmiş
        Her iki varyant da AYNI test setiyle (data/balanced/test_n<n>, gerçek dağılım) ölçülür.
        results/ogrenci2/imbalance_results.csv -> dengeleme yöntemlerinin süreye etkisi

Her (boyut, model, varyant) için ölçülenler -> results/ogrenci5/scale_results.csv
    read_sec       : parquet okuma + özellik pipeline'ı (fit+transform+cache)
    train_sec      : sadece model.fit() süresi (veri önceden cache'li)
    throughput     : train_rows / train_sec  (satır/sn)
    peak_heap_mb   : fit sırasında JVM heap kullanımının tepe değeri (0.2 sn'de bir örneklenir,
                     fit öncesi System.gc()); local modda driver = executor, tek JVM
    precision, recall, f1, auprc (eşik 0.5, test gerçek dağılımda)

Kesintiye dayanıklı: her deney bitince CSV'ye eklenir; tekrar çalıştırınca atlanır.

Kullanım:
    python src/ogrenci5_olcek_analizi.py                         # 100K, 1M, 5M x LR, RF x original, balanced
    python src/ogrenci5_olcek_analizi.py --models lr rf gbt
    python src/ogrenci5_olcek_analizi.py --plot_only
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from ogrenci2_class_imbalance import evaluate  # noqa: E402
from ogrenci4_model import MODEL_NAMES  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import (GBTClassifier, LogisticRegression,  # noqa: E402
                                       RandomForestClassifier)
from pyspark.sql import functions as F  # noqa: E402

LABEL = C.LABEL_COL
VARIANTS = ["original", "balanced"]
KEYS = ["size", "model", "variant"]
REQUIRED = ["auprc", "train_sec"]
COLS = KEYS + ["train_rows", "train_fraud", "test_rows", "test_fraud", "read_sec", "train_sec",
               "throughput_rows_per_sec", "peak_heap_mb", "heap_delta_mb", "precision", "recall", "f1",
               "auprc", "accuracy", "params", "balance_method", "finished_at"]
log = None


class HeapSampler:
    """Fit sırasında JVM heap kullanımını arka planda örnekler (tepe değer)."""
    def __init__(self, spark, interval=0.2):
        self.rt = spark._jvm.java.lang.Runtime.getRuntime()
        self.sys = spark._jvm.java.lang.System
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def used(self):
        return (self.rt.totalMemory() - self.rt.freeMemory()) / 1e6

    def __enter__(self):
        self.sys.gc()
        self.base = self.used()
        self.peak = self.base
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self.used())
            except Exception:   # py4j kapanırken
                return

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)
        self.peak = max(self.peak, self.used())
        return False


def make_model(name, args, wcol):
    common = dict(featuresCol="features", labelCol=LABEL)
    if name == "lr":
        m, p = LogisticRegression(maxIter=args.lr_iter, **common), f"maxIter={args.lr_iter}"
    elif name == "rf":
        m = RandomForestClassifier(numTrees=args.rf_trees, maxDepth=args.rf_depth, subsamplingRate=0.8,
                                   seed=C.SEED, **common)
        p = f"numTrees={args.rf_trees}, maxDepth={args.rf_depth}"
    else:
        m = GBTClassifier(maxIter=args.gbt_iter, maxDepth=args.gbt_depth, seed=C.SEED, **common)
        p = f"maxIter={args.gbt_iter}, maxDepth={args.gbt_depth}"
    if wcol:
        m.setWeightCol(wcol)
    return m, p


def load_variant(spark, size, variant, meta):
    """(train, test, weight_col) - test her iki varyantta da aynı."""
    info = meta["sizes"][str(size)]
    test = spark.read.parquet(info["test_path"])
    if variant == "balanced":
        return spark.read.parquet(info["train_path"]), test, meta.get("weight_col")
    train, _ = C.group_split(C.load_features(spark, size, log=log.info), 0.8)
    return train, test, None


def run(spark, args, out: Path):
    meta = json.loads((C.BALANCED_DIR / "SECILEN_YONTEM.json").read_text(encoding="utf-8"))
    method = f"{meta['method']} {meta['target_ratio']}"
    res_path = out / "scale_results.csv"
    done = set() if args.force else C.done_set(res_path, KEYS, REQUIRED)
    for size in sorted(args.sizes):
        if str(size) not in meta["sizes"]:
            log.warning(f"{size:,}: dengelenmiş veri yok (Öğrenci 2) -> atlandı")
            continue
        for variant in args.variants:
            todo = [m for m in args.models if (str(size), m, variant) not in done]
            if not todo:
                log.info(f"{size:,} {variant}: tamamlanmış -> atlandı")
                continue
            with C.Timer() as tr:
                train, test, wcol = load_variant(spark, size, variant, meta)
                keep = ["features", LABEL] + ([wcol] if wcol else [])
                fm = Pipeline(stages=C.build_feature_stages(meta["feature_cols"])).fit(train)
                train_f = fm.transform(train).select(*keep).cache()
                test_f = fm.transform(test).select("features", LABEL).cache()
                n_train, n_test = train_f.count(), test_f.count()
            n_fraud = train_f.filter(F.col(LABEL) == 1).count()
            t_fraud = test_f.filter(F.col(LABEL) == 1).count()
            for name in todo:
                tag = f"size={size:,} {MODEL_NAMES[name]} {variant}"
                log.info(f"BAŞLADI  {tag} | train={n_train:,} (fraud {n_fraud:,})")
                model_def, params = make_model(name, args, wcol)
                with HeapSampler(spark) as hs, C.Timer() as tt:
                    model = model_def.fit(train_f)
                m = evaluate(model.transform(test_f))
                row = {"size": size, "model": name, "variant": variant, "train_rows": n_train,
                       "train_fraud": n_fraud, "test_rows": n_test, "test_fraud": t_fraud,
                       "read_sec": round(tr.sec, 2), "train_sec": round(tt.sec, 2),
                       "throughput_rows_per_sec": round(n_train / tt.sec, 1),
                       "peak_heap_mb": round(hs.peak, 1), "heap_delta_mb": round(hs.peak - hs.base, 1),
                       **{k: m[k] for k in ("precision", "recall", "f1", "auprc", "accuracy")},
                       "params": params, "balance_method": method if variant == "balanced" else "none",
                       "finished_at": C.now_iso()}
                C.append_row(row, res_path, COLS)
                log.info(f"BİTTİ    {tag} | fit {tt.sec:.1f} sn, {row['throughput_rows_per_sec']:,.0f} satır/sn, "
                         f"heap tepe {hs.peak:,.0f} MB | P={m['precision']:.3f} R={m['recall']:.3f} "
                         f"F1={m['f1']:.3f} AUPRC={m['auprc']:.3f}")
            train_f.unpersist()
            test_f.unpersist()


# --------------------------------------------------------------------------- #
def plots(out: Path):
    import numpy as np
    import pandas as pd
    plt = C.plt_setup()
    r = C.read_rows(out / "scale_results.csv", KEYS, REQUIRED)
    if r.empty:
        log.warning("Grafik için sonuç yok")
        return
    r["size"] = r["size"].astype(int)
    models = [m for m in MODEL_NAMES if m in set(r["model"])]
    colors = {m: C.SERIES[i] for i, m in enumerate(MODEL_NAMES)}
    styles = {"original": "-", "balanced": "--"}
    sizes = sorted(r["size"].unique())

    def lines(ax, metric, logy=False, xcol="size"):
        for m in models:
            for v in VARIANTS:
                s = r[(r["model"] == m) & (r["variant"] == v)].sort_values(xcol)
                if s.empty:
                    continue
                ax.plot(s[xcol], s[metric], styles[v], color=colors[m], marker="o", lw=2,
                        label=f"{MODEL_NAMES[m]} ({'dengelenmiş' if v == 'balanced' else 'orijinal'})")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(sizes, [C.size_label(s) for s in sizes])
        ax.minorticks_off()

    # 1) Maliyet: süre, throughput, bellek
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=C.SURFACE)
    for ax, (metric, title, logy) in zip(axes, [
            ("train_sec", "Eğitim süresi (sn, log-log)", True),
            ("throughput_rows_per_sec", "Throughput (train satırı / sn)", True),
            ("peak_heap_mb", "Fit sırasında tepe JVM heap (MB)", False)]):
        C.style_ax(ax, title, xgrid=True)
        lines(ax, metric, logy)
        ax.set_xlabel("Veri boyutu", color=C.INK2)
    axes[0].legend(frameon=False, labelcolor=C.INK2, fontsize=8)
    fig.suptitle("Veri büyüdükçe maliyet", x=0.01, ha="left", color=C.INK, fontsize=13)
    C.save_fig(fig, out / "scale_cost.png")

    # 2) Performans: boyut arttıkça fraud yakalama
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=C.SURFACE)
    for ax, (metric, title) in zip(axes.flat, [("precision", "Precision"), ("recall", "Recall"),
                                               ("f1", "F1"), ("auprc", "AUPRC")]):
        C.style_ax(ax, title, xgrid=True)
        lines(ax, metric)
        ax.set_ylim(0, 1.02)
    axes[0, 0].legend(frameon=False, labelcolor=C.INK2, fontsize=8)
    fig.suptitle("Veri büyüdükçe performans (test gerçek dağılımda, eşik 0.5)", x=0.01, ha="left",
                 color=C.INK, fontsize=13)
    C.save_fig(fig, out / "scale_performance.png")

    # 3) Maliyet-fayda: F1 kazancı / süre artışı (en küçük boyuta göre)
    rows = []
    for (m, v), g in r.groupby(["model", "variant"]):
        g = g.sort_values("size")
        b = g.iloc[0]
        for _, x in g.iterrows():
            rows.append({"model": m, "variant": v, "size": x["size"],
                         "time_x": x["train_sec"] / b["train_sec"], "f1_gain": x["f1"] - b["f1"],
                         "auprc_gain": x["auprc"] - b["auprc"]})
    cb = pd.DataFrame(rows)
    cb.to_csv(out / "cost_benefit.csv", index=False)

    # Ölçeklenme üssü: train_sec ~ rows^k  (k=1 doğrusal)
    exps = []
    for (m, v), g in r.groupby(["model", "variant"]):
        if len(g) >= 2:
            k = np.polyfit(np.log(g["train_rows"].astype(float)), np.log(g["train_sec"].astype(float)), 1)[0]
            exps.append({"model": m, "variant": v, "scaling_exponent": round(k, 3)})
    pd.DataFrame(exps).to_csv(out / "scaling_exponent.csv", index=False)

    # 4) Dengesizliğin süreye etkisi (Öğrenci 2 sonuçları): train satırı arttıkça süre
    p2 = C.RESULTS_DIR / "ogrenci2" / "imbalance_results.csv"
    if p2.exists():
        d = C.read_rows(p2, ["size", "method", "target_ratio"], ["train_sec"])
        d["size"] = d["size"].astype(int)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=C.SURFACE)
        ax = axes[0]
        C.style_ax(ax, "Dengeleme yöntemi train setini değiştirir -> LR süresi", xgrid=True)
        mcol = {"none": C.INK2, "class_weight": C.SERIES[0], "undersample": C.SERIES[2],
                "oversample": C.SERIES[1]}
        for meth, g in d.groupby("method"):
            ax.scatter(g["train_rows"], g["train_sec"], color=mcol.get(meth, C.SERIES[3]), s=45,
                       label=meth, edgecolor="white", zorder=3)
            for _, x in g.iterrows():
                ax.annotate(f"{C.size_label(x['size'])} {x['target_ratio']}", (x["train_rows"], x["train_sec"]),
                            fontsize=7, color=C.INK2, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Train satır sayısı (dengeleme sonrası)", color=C.INK2)
        ax.set_ylabel("LR eğitim süresi (sn)", color=C.INK2)
        ax.legend(frameon=False, labelcolor=C.INK2, fontsize=9)

        ax = axes[1]
        C.style_ax(ax, "Train satırı, baseline'a (none) göre kaç kat?")
        base = d[d["method"] == "none"].set_index("size")["train_rows"]
        d = d[d["method"] != "none"].copy()
        d["rows_x"] = d.apply(lambda x: x["train_rows"] / base.get(x["size"], np.nan), axis=1)
        d["lab"] = d["method"] + " " + d["target_ratio"]
        labs = sorted(d["lab"].unique())
        xs = np.arange(len(labs))
        w = 0.8 / max(1, len(sizes))
        for i, s in enumerate(sorted(d["size"].unique())):
            sub = d[d["size"] == s].set_index("lab").reindex(labs)
            ax.bar(xs + (i - (len(sizes) - 1) / 2) * w, sub["rows_x"], w, color=C.SERIES[i],
                   edgecolor=C.SURFACE, label=C.size_label(s))
        ax.axhline(1, color=C.INK2, lw=1)
        ax.set_yscale("log")
        ax.set_xticks(xs, [l.replace(" ", "\n") for l in labs], fontsize=8.5)
        ax.legend(title="Veri boyutu", frameon=False, labelcolor=C.INK2, fontsize=9)
        C.save_fig(fig, out / "imbalance_time_effect.png")
    log.info(f"Grafikler -> {out}")


def main():
    global log
    ap = argparse.ArgumentParser(description="Öğrenci 5 - ölçek analizi")
    ap.add_argument("--sizes", type=int, nargs="+", default=C.DEFAULT_SIZES)
    ap.add_argument("--models", nargs="+", default=["lr", "rf"], choices=list(MODEL_NAMES))
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--lr_iter", type=int, default=100)
    ap.add_argument("--rf_trees", type=int, default=50)
    ap.add_argument("--rf_depth", type=int, default=8)
    ap.add_argument("--gbt_iter", type=int, default=30)
    ap.add_argument("--gbt_depth", type=int, default=5)
    ap.add_argument("--driver_memory", default="8g")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "ogrenci5"))
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else C.PROJECT_ROOT / out
    log = C.setup_logging(out, "ogrenci5")
    if args.plot_only:
        plots(out)
        return

    log.info(f"Öğrenci 5 başladı | sizes={args.sizes} models={args.models} variants={args.variants}")
    spark = C.get_spark("ogrenci5-olcek", args.driver_memory)
    try:
        run(spark, args, out)
        plots(out)
        log.info("Öğrenci 5 tamamlandı.")
    except BaseException as e:
        log.exception(f"Kesildi: {type(e).__name__}: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
