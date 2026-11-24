"""
Öğrenci 4 - Fraud Sınıflandırma Modelleri
=========================================

Girdi : data/balanced/train_n<n>.parquet (Öğrenci 2'nin seçtiği yöntemle dengelenmiş train)
        data/balanced/test_n<n>.parquet  (gerçek dağılımda test)
        data/balanced/SECILEN_YONTEM.json (yöntem, ağırlık sütunu, özellik listesi)
Çıktı : results/ogrenci4/

Modeller: Logistic Regression, Random Forest, Gradient-Boosted Trees (aynı özellik pipeline'ı,
          pipeline SADECE train'e fit edilir).
Metrikler: accuracy yerine precision, recall, F1, AUPRC (+ AUROC). Karşılaştırma için
          "her şeye normal de" sahte modeli: accuracy ~%99.5 ama hiç fraud yakalamaz.
Grafikler: metrik karşılaştırması, confusion matrix, PR eğrisi, eşik (threshold) analizi,
          özellik önemleri.

PR eğrisi / eşik analizi tüm tahminleri sürücüye toplamadan yapılır: fraud olasılığı Spark'ta
1000 kutuya bölünür, (kutu, etiket) sayıları pr_bins_<model>.csv'ye yazılır; eğri bu sayıların
kümülatif toplamından hesaplanır.

Kesintiye dayanıklı: her model bitince model_results.csv'ye eklenir; tekrar çalıştırınca atlanır.

Kullanım:
    python src/ogrenci4_model.py                         # 1M, LR + RF + GBT
    python src/ogrenci4_model.py --size 5000000 --driver_memory 10g
    python src/ogrenci4_model.py --plot_only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from ogrenci2_class_imbalance import evaluate  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import (GBTClassifier, LogisticRegression,  # noqa: E402
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

LABEL = C.LABEL_COL
N_BINS = 1000
MODEL_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "gbt": "GBT"}
KEYS = ["size", "model"]
REQUIRED = ["auprc", "train_sec"]
COLS = KEYS + ["train_rows", "test_rows", "test_fraud", "TP", "FP", "FN", "TN", "accuracy", "precision",
               "recall", "f1", "auprc", "auroc", "best_threshold", "best_f1", "p_at_best", "r_at_best",
               "train_sec", "predict_sec", "params", "finished_at"]
log = None


def make_model(name: str, args, weight_col: str | None):
    common = dict(featuresCol="features", labelCol=LABEL)
    if name == "lr":
        m = LogisticRegression(maxIter=args.lr_iter, **common)
        params = f"maxIter={args.lr_iter}"
    elif name == "rf":
        m = RandomForestClassifier(numTrees=args.rf_trees, maxDepth=args.rf_depth, seed=C.SEED,
                                   subsamplingRate=0.8, **common)
        params = f"numTrees={args.rf_trees}, maxDepth={args.rf_depth}"
    elif name == "gbt":
        m = GBTClassifier(maxIter=args.gbt_iter, maxDepth=args.gbt_depth, stepSize=0.1, seed=C.SEED, **common)
        params = f"maxIter={args.gbt_iter}, maxDepth={args.gbt_depth}"
    else:
        raise ValueError(name)
    if weight_col:
        m.setWeightCol(weight_col)
    return m, params


def prob_bins(pred):
    """fraud olasılığı -> 1000 kutu; (bin, is_fraud, count)"""
    p1 = vector_to_array("probability")[1]
    return (pred.withColumn("bin", F.least(F.floor(p1 * N_BINS), F.lit(N_BINS - 1)).cast("int"))
                .groupBy("bin", LABEL).count().toPandas())


def curve_from_bins(bins):
    """Kutulardan eşik başına precision / recall / F1. Eşik t: p >= t ise fraud."""
    import numpy as np
    import pandas as pd
    t = bins.pivot_table(index="bin", columns=LABEL, values="count", aggfunc="sum", fill_value=0)
    t = t.reindex(range(N_BINS), fill_value=0).reindex(columns=[0, 1], fill_value=0)
    # yüksek eşikten düşüğe kümülatif: bin >= k olanlar pozitif tahmin
    tp = t[1][::-1].cumsum()[::-1].to_numpy()
    fp = t[0][::-1].cumsum()[::-1].to_numpy()
    pos = t[1].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
        recall = tp / pos if pos else np.zeros_like(tp, dtype=float)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({"threshold": np.arange(N_BINS) / N_BINS, "precision": precision,
                         "recall": recall, "f1": f1, "tp": tp, "fp": fp})


def feature_names(feat_model, cols):
    idx = [s for s in feat_model.stages if s.__class__.__name__ == "StringIndexerModel"][0]
    return list(cols) + [f"category={l}" for l in idx.labels]


def run(spark, args, out: Path):
    meta = json.loads((C.BALANCED_DIR / "SECILEN_YONTEM.json").read_text(encoding="utf-8"))
    info = meta["sizes"].get(str(args.size))
    if not info:
        raise FileNotFoundError(f"{args.size} için dengelenmiş veri yok. Önce Öğrenci 2'yi çalıştırın.")
    for p in (info["train_path"], info["test_path"]):
        if not C.is_complete(p):
            raise FileNotFoundError(f"{p} yarım/eksik; Öğrenci 2'yi tekrar çalıştırın.")
    wcol = meta.get("weight_col")
    log.info(f"Girdi: {meta['method']} {meta['target_ratio']} ile dengelenmiş {args.size:,} | "
             f"ağırlık sütunu={wcol}")

    res_path = out / "model_results.csv"
    done = set() if args.force else C.done_set(res_path, KEYS, REQUIRED)
    todo = [m for m in args.models if (str(args.size), m) not in done]
    if not todo:
        log.info("Tüm modeller zaten tamamlanmış.")
        return

    train = spark.read.parquet(info["train_path"])
    test = spark.read.parquet(info["test_path"])
    keep = ["features", LABEL] + ([wcol] if wcol else [])
    feat_model = Pipeline(stages=C.build_feature_stages(meta["feature_cols"])).fit(train)
    train_f = feat_model.transform(train).select(*keep).cache()
    test_f = feat_model.transform(test).select("features", LABEL).cache()
    n_train, n_test = train_f.count(), test_f.count()
    test_fraud = test_f.filter(F.col(LABEL) == 1).count()
    log.info(f"train={n_train:,} satır | test={n_test:,} satır, fraud={test_fraud:,}")
    names = feature_names(feat_model, meta["feature_cols"])

    for i, name in enumerate(todo, 1):
        log.info(f"[{i}/{len(todo)}] BAŞLADI  {MODEL_NAMES[name]}")
        model_def, params = make_model(name, args, wcol)
        with C.Timer() as tt:
            model = model_def.fit(train_f)
        with C.Timer() as tp:
            pred = model.transform(test_f).cache()
            pred.count()
        m = evaluate(pred)
        auroc = BinaryClassificationEvaluator(labelCol=LABEL, metricName="areaUnderROC").evaluate(pred)
        bins = prob_bins(pred)
        bins.to_csv(out / f"pr_bins_{name}_n{args.size}.csv", index=False)
        cur = curve_from_bins(bins)
        best = cur.iloc[cur["f1"].idxmax()]
        pred.unpersist()

        if hasattr(model, "featureImportances"):
            imp = model.featureImportances.toArray()
            import pandas as pd
            pd.DataFrame({"feature": names[:len(imp)] + [f"f{j}" for j in range(len(names), len(imp))],
                          "importance": imp}).sort_values("importance", ascending=False) \
              .to_csv(out / f"feature_importance_{name}_n{args.size}.csv", index=False)

        row = {"size": args.size, "model": name, "train_rows": n_train, "test_rows": n_test,
               "test_fraud": test_fraud, **m, "auroc": round(auroc, 6),
               "best_threshold": round(float(best["threshold"]), 3), "best_f1": round(float(best["f1"]), 6),
               "p_at_best": round(float(best["precision"]), 6), "r_at_best": round(float(best["recall"]), 6),
               "train_sec": round(tt.sec, 2), "predict_sec": round(tp.sec, 2), "params": params,
               "finished_at": C.now_iso()}
        C.append_row(row, res_path, COLS)
        log.info(f"[{i}/{len(todo)}] BİTTİ    {MODEL_NAMES[name]} | P={m['precision']:.3f} R={m['recall']:.3f} "
                 f"F1={m['f1']:.3f} AUPRC={m['auprc']:.3f} | en iyi eşik {best['threshold']:.3f} -> "
                 f"F1={best['f1']:.3f} | fit {tt.sec:.1f} sn")
    train_f.unpersist()
    test_f.unpersist()


# --------------------------------------------------------------------------- #
# Grafikler (sadece CSV'lerden)
# --------------------------------------------------------------------------- #
def plots(out: Path, size: int):
    import numpy as np
    import pandas as pd
    plt = C.plt_setup()
    res = C.read_rows(out / "model_results.csv", KEYS, REQUIRED)
    res = res[res["size"] == str(size)]
    if res.empty:
        log.warning("Grafik için sonuç yok")
        return
    order = [m for m in MODEL_NAMES if m in set(res["model"])]
    res = res.set_index("model").loc[order]
    colors = {m: C.SERIES[i] for i, m in enumerate(MODEL_NAMES)}
    sl = C.size_label(size)

    # 1) Metrik karşılaştırması + "hep normal" sahte modeli
    metrics = ["accuracy", "precision", "recall", "f1", "auprc"]
    fraud_rate = res["test_fraud"].iloc[0] / res["test_rows"].iloc[0]
    dummy = {"accuracy": 1 - fraud_rate, "precision": 0, "recall": 0, "f1": 0, "auprc": fraud_rate}
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor=C.SURFACE)
    C.style_ax(ax, f"Model karşılaştırması ({sl}, test gerçek dağılımda, eşik 0.5)")
    x = np.arange(len(metrics))
    n = len(order) + 1
    w = 0.8 / n
    ax.bar(x - 0.4 + w / 2, [dummy[k] for k in metrics], w, color=C.GRID, edgecolor=C.SURFACE,
           label="'Hep normal' sahte model")
    for i, m in enumerate(order, 1):
        vals = [res.loc[m, k] for k in metrics]
        bars = ax.bar(x - 0.4 + w / 2 + i * w, vals, w, color=colors[m], edgecolor=C.SURFACE,
                      label=MODEL_NAMES[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [k.upper() if k == "auprc" else k.capitalize() for k in metrics])
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, labelcolor=C.INK2, fontsize=9, ncol=n, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    C.save_fig(fig, out / f"model_metrics_n{size}.png")

    # 2) Confusion matrix
    fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 4.4), facecolor=C.SURFACE)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, order):
        r = res.loc[m]
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=float)
        rown = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{int(v):,}\n(%{100 * rown[a, b]:.1f})", ha="center", va="center",
                    color="white" if rown[a, b] > 0.6 else C.INK, fontsize=10)
        ax.set_xticks([0, 1], ["Tahmin: normal", "Tahmin: fraud"])
        ax.set_yticks([0, 1], ["Gerçek: normal", "Gerçek: fraud"])
        ax.set_title(MODEL_NAMES[m], loc="left", color=C.INK, fontsize=12, fontweight="bold")
        ax.tick_params(colors=C.INK2, labelsize=9)
    fig.suptitle(f"Confusion matrix ({sl}, eşik 0.5; yüzdeler satır bazında)", x=0.01, ha="left",
                 color=C.INK, fontsize=12)
    C.save_fig(fig, out / f"confusion_matrix_n{size}.png")

    # 3) PR eğrisi + 4) eşik analizi
    curves = {m: curve_from_bins(pd.read_csv(out / f"pr_bins_{m}_n{size}.csv")) for m in order
              if (out / f"pr_bins_{m}_n{size}.csv").exists()}
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=C.SURFACE)
    C.style_ax(ax, f"Precision-Recall eğrisi ({sl})", xgrid=True)
    for m, c in curves.items():
        ax.plot(c["recall"], c["precision"], color=colors[m], lw=2,
                label=f"{MODEL_NAMES[m]} (AUPRC={res.loc[m, 'auprc']:.3f})")
        b = c.iloc[c["f1"].idxmax()]
        ax.scatter([b["recall"]], [b["precision"]], color=colors[m], s=40, zorder=3, edgecolor="white")
    ax.axhline(fraud_rate, color=C.INK2, ls="--", lw=1)
    ax.text(0.01, fraud_rate + 0.01, f"rastgele model (fraud oranı %{100 * fraud_rate:.2f})",
            color=C.INK2, fontsize=8)
    ax.set_xlabel("Recall", color=C.INK2)
    ax.set_ylabel("Precision", color=C.INK2)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, labelcolor=C.INK2, fontsize=9, loc="lower left")
    C.save_fig(fig, out / f"pr_curve_n{size}.png")

    fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4.5), facecolor=C.SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (m, c) in zip(axes, curves.items()):
        C.style_ax(ax, MODEL_NAMES[m], xgrid=True)
        for k, col in (("precision", C.SERIES[0]), ("recall", C.SERIES[1]), ("f1", C.SERIES[2])):
            ax.plot(c["threshold"], c[k], color=col, lw=2, label=k.capitalize() if k != "f1" else "F1")
        b = c.iloc[c["f1"].idxmax()]
        ax.axvline(b["threshold"], color=C.INK2, ls="--", lw=1)
        ax.axvline(0.5, color=C.GRID, lw=1)
        ax.text(b["threshold"], 0.03, f" en iyi F1\n eşik={b['threshold']:.2f}", color=C.INK2, fontsize=8)
        ax.set_xlabel("Karar eşiği", color=C.INK2)
        ax.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, labelcolor=C.INK2, fontsize=9)
    fig.suptitle(f"Eşik ayarı: precision ve recall arasındaki denge ({sl})", x=0.01, ha="left",
                 color=C.INK, fontsize=12)
    C.save_fig(fig, out / f"threshold_analysis_n{size}.png")

    # 5) Özellik önemleri (ağaç modelleri)
    imps = {m: pd.read_csv(out / f"feature_importance_{m}_n{size}.csv") for m in ("rf", "gbt")
            if (out / f"feature_importance_{m}_n{size}.csv").exists()}
    if imps:
        fig, axes = plt.subplots(1, len(imps), figsize=(6.5 * len(imps), 5.5), facecolor=C.SURFACE)
        axes = np.atleast_1d(axes)
        for ax, (m, d) in zip(axes, imps.items()):
            d = d.head(15)[::-1]
            C.style_ax(ax, f"{MODEL_NAMES[m]}: en önemli 15 özellik", xgrid=True, ygrid=False)
            ax.barh(d["feature"], d["importance"], color=colors[m], edgecolor=C.SURFACE)
            ax.tick_params(axis="y", labelsize=8.5)
        C.save_fig(fig, out / f"feature_importance_n{size}.png")
    log.info(f"Grafikler -> {out}")


def main():
    global log
    ap = argparse.ArgumentParser(description="Öğrenci 4 - fraud sınıflandırma modelleri")
    ap.add_argument("--size", type=int, default=1_000_000)
    ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    ap.add_argument("--lr_iter", type=int, default=100)
    ap.add_argument("--rf_trees", type=int, default=100)
    ap.add_argument("--rf_depth", type=int, default=10)
    ap.add_argument("--gbt_iter", type=int, default=50)
    ap.add_argument("--gbt_depth", type=int, default=6)
    ap.add_argument("--driver_memory", default="8g")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "ogrenci4"))
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else C.PROJECT_ROOT / out
    log = C.setup_logging(out, "ogrenci4")
    if args.plot_only:
        plots(out, args.size)
        return

    log.info(f"Öğrenci 4 başladı | size={args.size:,} models={args.models}")
    spark = C.get_spark("ogrenci4-model", args.driver_memory)
    try:
        run(spark, args, out)
        plots(out, args.size)
        log.info("Öğrenci 4 tamamlandı.")
    except BaseException as e:
        log.exception(f"Kesildi: {type(e).__name__}: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
