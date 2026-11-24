"""
Öğrenci 4 - Fraud Sınıflandırma Modelleri ve Değerlendirme
==========================================================

Soru: Dengesiz (fraud ~%0.5) veride hangi model fraud'u en iyi yakalar? Accuracy neden
yanıltıcıdır, precision / recall / F1 / AUPRC neyi gösterir? Karar eşiği 0.5 yerine
ayarlanınca ne değişir?

Girdi (Öğrenci 2'nin çıktısı, README > Arayüz):
    data/balanced/SECILEN_YONTEM.json   yöntem, feature_cols, weight_col
    data/balanced/train_n<n>.parquet    seçilen yöntemle dengelenmiş train
    data/balanced/test_n<n>.parquet     GERÇEK dağılımda test (dokunulmaz)

Akış (her veri boyutu için)
---------------------------
a) Train, orig_id hash'iyle ikiye bölünür: %80 fit, %20 doğrulama (eşik seçimi için).
   Test seti eşik seçiminde HİÇ kullanılmaz; sadece en sonda bir kez ölçülür.
b) Özellik pipeline'ı (category one-hot + VectorAssembler + StandardScaler) sadece fit kısmına fit.
c) Modeller: Logistic Regression, Random Forest, GBT (Spark ML). weight_col varsa weightCol olarak.
d) Test metrikleri iki eşikte: 0.5 ve doğrulamada F1'i en yüksek yapan eşik.
   Precision, Recall, F1, Accuracy, AUPRC, AUROC, confusion matrix (TP/FP/FN/TN).
   + Naif model ("her şeye normal de"): accuracy paradoksunu göstermek için.
e) PR eğrisi ve eşik analizi (eşiğe göre P/R/F1), özellik önemi (RF/GBT: featureImportances,
   LR: standartlaştırılmış katsayıların mutlak değeri).
f) Grafikler + model_secimi.md (sonuçlardan sayılarla otomatik yorum).

Eşik seçiminde öncel düzeltmesi
-------------------------------
Doğrulama kümesi dengelenmiş train'den geldiği için fraud oranı gerçekte olduğundan yüksektir
(ör. undersample 1:10'da ~%9, gerçekte ~%0.5). Orada seçilen eşik testte çok fazla yanlış alarm
verir. Bu yüzden doğrulamadaki her normal satır  w = (testte normal/fraud) / (train'de normal/fraud)
ağırlığıyla sayılır: precision = TP / (TP + w*FP). Yani eşik, gerçek dağılımda beklenen F1'e göre
seçilir. class_weight yönteminde satırlar atılmadığı için w ~ 1 olur.

Çıktılar (results/ogrenci4/)
----------------------------
    model_results.csv        (input, size, model, threshold_type) başına tüm metrikler + süreler
    pr_curve.csv             test PR / eşik eğrisi noktaları (eşik, TP, FP, FN, P, R, F1)
    feature_importance.csv   boyut x model x özellik
    metrics_by_model.png     pr_curves.png    threshold_analysis.png
    confusion_matrices.png   feature_importance.png    training_time.png
    model_secimi.md          sonuç ve yorum
    run.log

Kesintiye dayanıklılık: her (boyut, model) biter bitmez CSV'lere yazılır; tekrar çalıştırınca
bitmiş olanlar atlanır (--force ile yeniden).

Kullanım:
    python src/ogrenci4_modelleme.py                              # 100K, 1M, 5M; LR + RF + GBT
    python src/ogrenci4_modelleme.py --sizes 5000000 --driver_memory 8g
    python src/ogrenci4_modelleme.py --models lr gbt --gbt_iter 100
    python src/ogrenci4_modelleme.py --plot_only                  # Spark'sız, CSV'lerden grafikler
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import (GBTClassifier, LogisticRegression,  # noqa: E402
                                       RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import DataFrame  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

LABEL = C.LABEL_COL
PROB = "p_fraud"
N_BINS = 10_000            # eşik çözünürlüğü: 0.0001
MODELS = ["lr", "lr_fe", "rf", "gbt"]
MODEL_NAMES = {"lr": "Logistic Regression", "lr_fe": "LR + özellik müh.", "rf": "Random Forest",
               "gbt": "GBT", "naive": "Naif: hep normal"}
# lr_fe: LR doğrusal olduğu için "çok küçük VE çok büyük tutar şüpheli" gibi iki uçlu ilişkileri
# öğrenemez. Bu sütunlar 20 kutuya bölünüp one-hot yapılır (her aralığa ayrı ağırlık) ve
# log_amt x kategori etkileşimi eklenir (aynı tutar her kategoride farklı anlam taşır).
BIN_COLS = ["amt", "hour", "age", "amt_ratio_to_avg", "log_secs_since_prev", "txn_count_24h"]
N_BUCKETS = 20
KEYS = ["input", "size", "model", "threshold_type"]
REQUIRED = ["f1", "accuracy"]
RESULT_COLS = KEYS + ["method", "train_rows", "train_fraud", "val_rows", "val_fraud", "val_weight",
                      "test_rows", "test_fraud", "threshold", "TP", "FP", "FN", "TN",
                      "accuracy", "precision", "recall", "f1", "auprc", "auroc",
                      "val_f1", "train_sec", "predict_sec", "train_rows_per_sec", "finished_at"]
TUNE_COLS = ["input", "size", "model", "params", "val_auprc", "fit_sec", "finished_at"]
log = None


# --------------------------------------------------------------------------- #
# Girdi
# --------------------------------------------------------------------------- #
def read_meta() -> dict:
    path = C.BALANCED_DIR / "SECILEN_YONTEM.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} yok. Önce Öğrenci 2 çalıştırılmalı (README > Çalıştırma).")
    return json.loads(path.read_text(encoding="utf-8"))


def balanced_paths(size: int) -> tuple[Path, Path]:
    # JSON'daki yollar mutlak ve başka bilgisayarda üretilmiş olabilir; her zaman yerel klasörü kullan
    return C.BALANCED_DIR / f"train_n{size}.parquet", C.BALANCED_DIR / f"test_n{size}.parquet"


def check_input(meta: dict, sizes: list[int]):
    if meta.get("input") != "features":
        ready = all(C.is_complete(C.features_path(s)) for s in sizes)
        msg = (f"data/balanced/ Öğrenci 3'ün özellikleri OLMADAN üretilmiş (input={meta.get('input')}). "
               "Modeller sadece temel özellikleri görecek.")
        if ready:
            msg += (" data/features/ hazır: önce Öğrenci 2'yi tekrar çalıştırın "
                    "(.\\run_all.ps1 -Steps 2), sonra bu scripti.")
        log.warning(msg)
    for s in sizes:
        for p in balanced_paths(s):
            if not C.is_complete(p):
                raise FileNotFoundError(f"{p} yok/yarım (_SUCCESS yok). Öğrenci 2'yi bu boyutla çalıştırın.")


# --------------------------------------------------------------------------- #
# Metrikler
# --------------------------------------------------------------------------- #
def with_prob(pred: DataFrame) -> DataFrame:
    return pred.withColumn(PROB, vector_to_array("probability")[1])


def binned_counts(pred: DataFrame) -> "pd.DataFrame":
    """Olasılık 0.0001'lik kutulara bölünür; her kutuda fraud / normal sayısı (küçük tablo).
    Bütün eşik analizi bu tablodan yapılır, 1M satırlık test sürücüye çekilmez."""
    import pandas as pd
    b = F.least(F.floor(F.col(PROB) * N_BINS), F.lit(N_BINS - 1)).cast("int")
    rows = (pred.select(b.alias("bin"), F.col(LABEL).cast("int").alias("y"))
            .groupBy("bin").agg(F.sum("y").alias("pos"), F.count("*").alias("n")).collect())
    df = pd.DataFrame([r.asDict() for r in rows], columns=["bin", "pos", "n"])
    df["neg"] = df["n"] - df["pos"]
    return df.sort_values("bin").reset_index(drop=True)


def threshold_table(bins, neg_weight: float = 1.0):
    """Her eşik t = bin/N_BINS için (tahmin = p >= t) TP, FP, FN, TN, precision, recall, F1.
    neg_weight: normal satırların ağırlığı (doğrulamada öncel düzeltmesi)."""
    import numpy as np
    d = bins.sort_values("bin", ascending=False).reset_index(drop=True)
    P, N = d["pos"].sum(), d["neg"].sum()
    tp = d["pos"].cumsum().to_numpy(float)
    fp = d["neg"].cumsum().to_numpy(float)
    wfp = fp * neg_weight
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + wfp > 0, tp / (tp + wfp), 1.0)
        rec = tp / P if P else np.zeros_like(tp)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    out = d[["bin"]].copy()
    out["threshold"] = out["bin"] / N_BINS
    out["TP"], out["FP"] = tp.astype(int), fp.astype(int)
    out["FN"], out["TN"] = (P - tp).astype(int), (N - fp).astype(int)
    out["precision"], out["recall"], out["f1"] = prec, rec, f1
    return out


def best_threshold(val_bins, neg_weight: float) -> tuple[float, float]:
    t = threshold_table(val_bins, neg_weight)
    i = int(t["f1"].to_numpy().argmax())
    return float(t["threshold"].iloc[i]), float(t["f1"].iloc[i])


def weighted_ap(bins, neg_weight: float = 1.0) -> float:
    """Doğrulamada öncel düzeltmeli AUPRC (average precision): normal satırlar neg_weight ile sayılır.
    Ayar seçimi gerçek dağılımda beklenen sıralama başarısına göre yapılsın diye."""
    import numpy as np
    t = threshold_table(bins, neg_weight)
    rec = np.concatenate([[0.0], t["recall"].to_numpy()])
    return float(np.sum(np.diff(rec) * t["precision"].to_numpy()))


def metrics_at(pred: DataFrame, threshold: float) -> dict:
    yhat = (F.col(PROB) >= F.lit(threshold)).cast("int")
    cm = {(int(r["y"]), int(r["yhat"])): r["count"]
          for r in pred.select(F.col(LABEL).cast("int").alias("y"), yhat.alias("yhat"))
          .groupBy("y", "yhat").count().collect()}
    tp, fp, fn, tn = cm.get((1, 1), 0), cm.get((0, 1), 0), cm.get((1, 0), 0), cm.get((0, 0), 0)
    return confusion_metrics(tp, fp, fn, tn)


def confusion_metrics(tp, fp, fn, tn) -> dict:
    total = tp + fp + fn + tn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "accuracy": round((tp + tn) / total, 6),
            "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}


def area_metrics(pred: DataFrame) -> dict:
    ev = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol=PROB)
    return {"auprc": round(ev.setMetricName("areaUnderPR").evaluate(pred), 6),
            "auroc": round(ev.setMetricName("areaUnderROC").evaluate(pred), 6)}


# --------------------------------------------------------------------------- #
# Modeller
# --------------------------------------------------------------------------- #
# --tune: doğrulama kümesinde denenen ayarlar (test seti kullanılmaz)
GRID = {
    "rf": [{"numTrees": t, "maxDepth": d} for t in (100, 200) for d in (10, 15)],
    "gbt": [{"maxIter": i, "maxDepth": d} for i in (50, 100, 200) for d in (5, 7)],
}


def params_str(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())


def make_model(name: str, args, weight_col: str | None):
    common = dict(featuresCol="features", labelCol=LABEL)
    if name in ("lr", "lr_fe"):
        m = LogisticRegression(maxIter=args.lr_iter, regParam=args.lr_reg, **common)
    elif name == "rf":
        m = RandomForestClassifier(numTrees=args.rf_trees, maxDepth=args.rf_depth, seed=C.SEED,
                                   featureSubsetStrategy="sqrt", **common)
    elif name == "gbt":
        m = GBTClassifier(maxIter=args.gbt_iter, maxDepth=args.gbt_depth, stepSize=0.1, seed=C.SEED,
                          **common)
    else:
        raise ValueError(name)
    if weight_col:
        m.setWeightCol(weight_col)
    return m


def feature_names(df: DataFrame, numeric_cols: list[str], feat_model) -> list[str]:
    """VectorAssembler çıktısındaki sıraya göre özellik adları (one-hot: category=<değer>)."""
    try:
        attrs = df.schema["raw_features"].metadata["ml_attr"]["attrs"]
        names = sorted((a for group in attrs.values() for a in group), key=lambda a: a["idx"])
        out = [a["name"].replace("category_ohe_", "category=") for a in names]
        if out:
            return out
    except (KeyError, TypeError):
        pass
    labels = feat_model.stages[0].labels
    return numeric_cols + [f"category={c}" for c in labels]


def importances(model, name: str, names: list[str]) -> list[tuple[str, float]]:
    import numpy as np
    if name in ("lr", "lr_fe"):
        v = np.abs(model.coefficients.toArray())
    else:
        v = model.featureImportances.toArray()
    v = v / v.sum() if v.sum() > 0 else v
    return sorted(zip(names, v.tolist()), key=lambda x: -x[1])


def fe_stages(num_cols: list[str], cats: list[str]):
    """lr_fe için özellik pipeline'ı: temel pipeline + kutulanmış sütunlar (one-hot) + log_amt x kategori."""
    from pyspark.ml.feature import (OneHotEncoder, QuantileDiscretizer, StandardScaler, StringIndexer,
                                    VectorAssembler)
    bins = [c for c in BIN_COLS if c in num_cols]
    inter = [f"log_amt_x_{c}" for c in cats] if "log_amt" in num_cols else []
    return [
        StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep"),
        OneHotEncoder(inputCols=["category_idx"], outputCols=["category_ohe"]),
        *[QuantileDiscretizer(inputCol=c, outputCol=f"{c}_bin", numBuckets=N_BUCKETS, handleInvalid="keep")
          for c in bins],
        OneHotEncoder(inputCols=[f"{c}_bin" for c in bins], outputCols=[f"{c}_aralik" for c in bins]),
        VectorAssembler(inputCols=num_cols + ["category_ohe"] + [f"{c}_aralik" for c in bins] + inter,
                        outputCol="raw_features"),
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
    ]


def add_interactions(df: DataFrame, cats: list[str]) -> DataFrame:
    if "log_amt" not in df.columns:
        return df
    return df.select("*", *[(F.col("log_amt") * (F.col("category") == c).cast("double")).alias(f"log_amt_x_{c}")
                            for c in cats])


def featurize(stages, fit: DataFrame, val: DataFrame, test: DataFrame, num_cols, keep):
    """Pipeline'ı SADECE fit'e uydurur; (fit_f, val_f, test_f, özellik adları) döner."""
    feat_model = Pipeline(stages=stages).fit(fit)
    fit_t = feat_model.transform(fit)
    names = feature_names(fit_t, num_cols, feat_model)
    fit_f = fit_t.select(keep).cache()
    val_f = feat_model.transform(val).select("features", LABEL).cache()
    test_f = feat_model.transform(test).select("features", LABEL).cache()
    return fit_f, val_f, test_f, names


# --------------------------------------------------------------------------- #
# Bir veri boyutu
# --------------------------------------------------------------------------- #
def run_size(spark, size: int, models: list[str], meta: dict, args, out: Path) -> int:
    res_path = out / "model_results.csv"
    input_kind = meta.get("input", "?")
    done = set() if args.force else C.done_set(res_path, KEYS, REQUIRED)
    todo = [m for m in models if (input_kind, str(size), m, "tuned") not in done]
    if len(todo) < len(models):
        log.info(f"  {len(models) - len(todo)} model CSV'de zaten var -> atlanıyor")
    if not todo:
        return 0

    train_path, test_path = balanced_paths(size)
    train_all = spark.read.parquet(str(train_path))
    test = spark.read.parquet(str(test_path))
    wcol = meta.get("weight_col")
    if wcol and wcol not in train_all.columns:
        log.warning(f"  weight_col={wcol} train'de yok -> ağırlıksız eğitiliyor")
        wcol = None
    num_cols = [c for c in meta.get("feature_cols", C.MODEL_FEATURES) if c in train_all.columns]
    missing = set(meta.get("feature_cols", [])) - set(num_cols)
    if missing:
        log.warning(f"  feature_cols içinde olup veride olmayan sütunlar atlandı: {sorted(missing)}")

    # a) fit / doğrulama ayrımı (farklı seed: train'in hepsi ilk hash'te zaten <0.8 kovasında)
    fit, val = C.group_split(train_all, 0.8, seed=C.SEED + 1)

    # b) Özellik pipeline'ı sadece fit kısmına
    keep = ["features", LABEL] + ([wcol] if wcol else [])
    fit_f, val_f, test_f, names = featurize(C.build_feature_stages(num_cols), fit, val, test, num_cols, keep)
    sets = {"base": (fit_f, val_f, test_f, names)}
    if "lr_fe" in todo:
        cats = sorted(r[0] for r in fit.select("category").distinct().collect())
        sets["fe"] = featurize(fe_stages(num_cols, cats), add_interactions(fit, cats), add_interactions(val, cats),
                               add_interactions(test, cats), num_cols, keep)
    f0, f1_ = C.class_counts(fit_f)
    v0, v1 = C.class_counts(val_f)
    t0, t1 = C.class_counts(test_f)
    # öncel düzeltmesi: doğrulamadaki normal satırların gerçek dağılıma göre ağırlığı
    w_neg = (t0 / max(t1, 1)) / (v0 / max(v1, 1)) if v0 else 1.0
    log.info(f"  fit={f0 + f1_:,} (fraud {f1_:,}) | doğrulama={v0 + v1:,} (fraud {v1:,}) | "
             f"test={t0 + t1:,} (fraud {t1:,}, %{100 * t1 / (t0 + t1):.3f}) | "
             f"doğrulamada normal ağırlığı w={w_neg:.2f} | özellik sayısı={len(names)}")

    base = {"input": input_kind, "size": size, "method": f"{meta.get('method')} {meta.get('target_ratio')}",
            "train_rows": f0 + f1_, "train_fraud": f1_, "val_rows": v0 + v1, "val_fraud": v1,
            "val_weight": round(w_neg, 4), "test_rows": t0 + t1, "test_fraud": t1}

    # Naif model: her şeye "normal" (accuracy paradoksu)
    if args.force or (input_kind, str(size), "naive", "0.5") not in done:
        C.append_row({**base, "model": "naive", "threshold_type": "0.5", "threshold": 1.0,
                      **confusion_metrics(0, 0, t1, t0), "finished_at": C.now_iso()}, res_path, RESULT_COLS)

    for i, name in enumerate(todo, 1):
        log.info(f"  [{i}/{len(todo)}] BAŞLADI  size={size:,} {MODEL_NAMES[name]}")
        fit_f, val_f, test_f, names = sets["fe" if name == "lr_fe" else "base"]
        est = make_model(name, args, wcol)
        grid = GRID.get(name) if args.tune else None
        if grid:
            # Her ayarla fit kısmında eğit, doğrulamada öncel düzeltmeli AUPRC'ye göre en iyisini seç
            best = None
            for j, p in enumerate(grid, 1):
                with C.Timer() as tg:
                    m_ = est.copy(p and {est.getParam(k): v for k, v in p.items()}).fit(fit_f)
                vb = binned_counts(with_prob(m_.transform(val_f)).select(PROB, LABEL))
                ap = weighted_ap(vb, w_neg)
                C.append_row({"input": input_kind, "size": size, "model": name, "params": params_str(p),
                              "val_auprc": round(ap, 6), "fit_sec": round(tg.sec, 2), "finished_at": C.now_iso()},
                             out / "tuning_results.csv", TUNE_COLS)
                log.info(f"      ayar {j}/{len(grid)} {params_str(p):28s} doğrulama AUPRC={ap:.4f} ({tg.sec:.1f} sn)")
                if best is None or ap > best[0]:
                    best = (ap, p, tg.sec, m_)
            log.info(f"      SEÇİLEN: {params_str(best[1])} (doğrulama AUPRC={best[0]:.4f})")
            model, fit_sec = best[3], best[2]
        else:
            with C.Timer() as tt:
                model = est.fit(fit_f)
            fit_sec = tt.sec
        with C.Timer() as tp:
            val_p = with_prob(model.transform(val_f)).select(PROB, LABEL).cache()
            test_p = with_prob(model.transform(test_f)).select(PROB, LABEL).cache()
            test_p.count(), val_p.count()
        thr, val_f1 = best_threshold(binned_counts(val_p), w_neg)
        areas = area_metrics(test_p)
        test_bins = binned_counts(test_p)

        # e) eğri + önem: sonuç satırından ÖNCE yazılır (sonuç satırı "bitti" işaretidir)
        curve = threshold_table(test_bins)
        curve = curve.assign(input=input_kind, size=size, model=name)
        C.merge_csv(curve.to_dict("records"), out / "pr_curve.csv", ["input", "size", "model", "bin"])
        imp = [{"input": input_kind, "size": size, "model": name, "feature": f, "importance": round(v, 6)}
               for f, v in importances(model, name, names)]
        C.merge_csv(imp, out / "feature_importance.csv", ["input", "size", "model", "feature"])

        common_row = {**base, "model": name, **areas, "train_sec": round(fit_sec, 2),
                      "predict_sec": round(tp.sec, 2),
                      "train_rows_per_sec": round((f0 + f1_) / max(fit_sec, 1e-9)), "finished_at": C.now_iso()}
        for ttype, t in (("0.5", 0.5), ("tuned", thr)):
            m = metrics_at(test_p, t)
            C.append_row({**common_row, "threshold_type": ttype, "threshold": round(t, 4),
                          "val_f1": round(val_f1, 6) if ttype == "tuned" else None, **m},
                         res_path, RESULT_COLS)
            log.info(f"  [{i}/{len(todo)}] {MODEL_NAMES[name]:20s} eşik={t:.4f}: P={m['precision']:.3f} "
                     f"R={m['recall']:.3f} F1={m['f1']:.3f} | TP={m['TP']} FP={m['FP']} FN={m['FN']}")
        log.info(f"  [{i}/{len(todo)}] BİTTİ    {MODEL_NAMES[name]} | AUPRC={areas['auprc']:.3f} "
                 f"AUROC={areas['auroc']:.3f} | fit {fit_sec:.1f} sn, tahmin {tp.sec:.1f} sn | "
                 f"en önemli: {', '.join(f for f, _ in importances(model, name, names)[:3])}")
        val_p.unpersist()
        test_p.unpersist()

    for fs in sets.values():
        for d in fs[:3]:
            d.unpersist()
    return len(todo)


# --------------------------------------------------------------------------- #
# Sonuç okuma
# --------------------------------------------------------------------------- #
def read_results(out: Path, input_kind: str | None = None):
    import pandas as pd
    res = C.read_rows(out / "model_results.csv", KEYS, REQUIRED, RESULT_COLS)
    if res.empty:
        return res, pd.DataFrame(), pd.DataFrame()
    res["size"] = res["size"].astype(int)
    if input_kind is None:
        input_kind = "features" if (res["input"] == "features").any() else res["input"].iloc[-1]
    res = res[res["input"] == input_kind].reset_index(drop=True)
    curve = pd.read_csv(out / "pr_curve.csv") if (out / "pr_curve.csv").exists() else pd.DataFrame()
    imp = pd.read_csv(out / "feature_importance.csv") if (out / "feature_importance.csv").exists() \
        else pd.DataFrame()
    if not curve.empty:
        curve = curve[curve["input"] == input_kind]
    if not imp.empty:
        imp = imp[imp["input"] == input_kind]
    return res, curve, imp


# --------------------------------------------------------------------------- #
# f) Grafikler
# --------------------------------------------------------------------------- #
def make_plots(res, curve, imp, out: Path):
    import numpy as np
    plt = C.plt_setup()
    sizes = sorted(res["size"].unique())
    big = sizes[-1]
    models = [m for m in MODELS if m in set(res["model"])]
    colors = {m: C.SERIES[i] for i, m in enumerate(MODELS)}
    size_col = {s: C.SERIES[i % len(C.SERIES)] for i, s in enumerate(sizes)}
    tuned = res[res["threshold_type"] == "tuned"]

    # 1) Model x boyut: P / R / F1 / AUPRC (ayarlı eşik)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=C.SURFACE)
    x = np.arange(len(models))
    w = 0.8 / len(sizes)
    for ax, (metric, title) in zip(axes.flat, [("precision", "Precision"), ("recall", "Recall"),
                                               ("f1", "F1"), ("auprc", "AUPRC")]):
        C.style_ax(ax, title)
        for i, s in enumerate(sizes):
            sub = tuned[tuned["size"] == s].set_index("model").reindex(models)
            bars = ax.bar(x + (i - (len(sizes) - 1) / 2) * w, sub[metric].fillna(0), w, color=size_col[s],
                          edgecolor=C.SURFACE, linewidth=1.5, label=C.size_label(s))
            if s == big:
                for b, v in zip(bars, sub[metric].fillna(0)):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=8, color=C.INK)
        ax.set_xticks(x, [MODEL_NAMES[m] for m in models], fontsize=9)
        ax.set_ylim(0, 1.08)
    axes[0, 0].legend(title="Veri boyutu", frameon=False, fontsize=9, title_fontsize=9, labelcolor=C.INK2)
    fig.suptitle("Modellerin test metrikleri (gerçek dağılım, doğrulamada ayarlanan eşik)",
                 x=0.01, ha="left", color=C.INK, fontsize=13)
    C.save_fig(fig, out / "metrics_by_model.png")

    if not curve.empty:
        cb = curve[curve["size"] == big]
        fraud_rate = tuned[tuned["size"] == big]["test_fraud"].iloc[0] / tuned[tuned["size"] == big]["test_rows"].iloc[0]

        # 2) PR eğrileri
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=C.SURFACE)
        C.style_ax(ax, f"Precision-Recall eğrisi ({C.size_label(big)}, test)", xgrid=True)
        for m in models:
            c = cb[cb["model"] == m].sort_values("threshold", ascending=False)
            r = tuned[(tuned["size"] == big) & (tuned["model"] == m)].iloc[0]
            ax.plot(c["recall"], c["precision"], color=colors[m], lw=2,
                    label=f"{MODEL_NAMES[m]} (AUPRC {r['auprc']:.3f})")
            ax.scatter([r["recall"]], [r["precision"]], color=colors[m], s=60, zorder=5,
                       edgecolor=C.INK, linewidth=0.8)
        ax.axhline(fraud_rate, color=C.INK2, ls="--", lw=1)
        ax.text(0.99, fraud_rate + 0.015, f"rastgele model = fraud oranı (%{100 * fraud_rate:.2f})",
                color=C.INK2, fontsize=8, ha="right")
        ax.set_xlabel("Recall (yakalanan fraud payı)", color=C.INK2)
        ax.set_ylabel("Precision (alarmların doğru olan payı)", color=C.INK2)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=9, labelcolor=C.INK2, loc="lower left", bbox_to_anchor=(0, 0.04))
        ax.text(0.99, 0.99, "nokta = ayarlı eşik", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=C.INK2)
        C.save_fig(fig, out / "pr_curves.png")

        # 3) Eşik analizi
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), facecolor=C.SURFACE,
                                 squeeze=False)
        for ax, m in zip(axes[0], models):
            c = cb[cb["model"] == m].sort_values("threshold")
            r = tuned[(tuned["size"] == big) & (tuned["model"] == m)].iloc[0]
            C.style_ax(ax, MODEL_NAMES[m], xgrid=True)
            for metric, col, lab in (("precision", C.SERIES[0], "Precision"), ("recall", C.SERIES[1], "Recall"),
                                     ("f1", C.SERIES[2], "F1")):
                ax.plot(c["threshold"], c[metric], color=col, lw=2, label=lab)
            ax.axvline(0.5, color=C.INK2, ls=":", lw=1)
            ax.axvline(r["threshold"], color=C.INK, ls="--", lw=1)
            ax.text(r["threshold"], 1.02, f" ayarlı {r['threshold']:.3f}", fontsize=8, color=C.INK,
                    ha="right" if r["threshold"] > 0.7 else "left")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.08)
            ax.set_xlabel("Karar eşiği", color=C.INK2)
        h_, l_ = axes[0][0].get_legend_handles_labels()
        fig.legend(h_, l_, frameon=False, fontsize=9, labelcolor=C.INK2, loc="upper right", ncol=3)
        fig.suptitle(f"Eşik analizi ({C.size_label(big)}, test): eşik yükseldikçe precision artar, recall düşer"
                     " · noktalı = 0.5", x=0.01, ha="left", color=C.INK, fontsize=12)
        C.save_fig(fig, out / "threshold_analysis.png")

    # 4) Confusion matrix (en büyük boyut; üst: 0.5, alt: ayarlı eşik)
    rb = res[res["size"] == big]
    fig, axes = plt.subplots(2, len(models), figsize=(4.2 * len(models), 7.6), facecolor=C.SURFACE,
                             squeeze=False)
    for row, ttype in enumerate(("0.5", "tuned")):
        for ax, m in zip(axes[row], models):
            r = rb[(rb["model"] == m) & (rb["threshold_type"] == ttype)].iloc[0]
            cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=float)
            norm = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for (a, b), v in np.ndenumerate(cm):
                ax.text(b, a, f"{int(v):,}\n(%{100 * norm[a, b]:.1f})", ha="center", va="center",
                        fontsize=9, color="white" if norm[a, b] > 0.6 else C.INK)
            ax.set_xticks([0, 1], ["Tahmin: normal", "Tahmin: fraud"], fontsize=8)
            ax.set_yticks([0, 1], ["Gerçek: normal", "Gerçek: fraud"], fontsize=8)
            ax.set_title(f"{MODEL_NAMES[m]} · eşik {r['threshold']:.3f}", fontsize=10, color=C.INK, loc="left")
    fig.suptitle(f"Confusion matrix ({C.size_label(big)}, test) · üst: eşik 0.5, alt: ayarlı eşik · "
                 "yüzdeler satır bazında", x=0.01, ha="left", color=C.INK, fontsize=12)
    C.save_fig(fig, out / "confusion_matrices.png")

    # 5) Özellik önemi (en büyük boyut, ağaç modelleri)
    tree = [m for m in ("gbt", "rf") if m in models]
    if not imp.empty and tree:
        fig, axes = plt.subplots(1, len(tree), figsize=(6.5 * len(tree), 6), facecolor=C.SURFACE, squeeze=False)
        for ax, m in zip(axes[0], tree):
            d = imp[(imp["size"] == big) & (imp["model"] == m)].nlargest(15, "importance")[::-1]
            C.style_ax(ax, f"{MODEL_NAMES[m]}: en önemli 15 özellik", xgrid=True, ygrid=False)
            behav = set(C.BEHAVIOR_FEATURES)
            ax.barh(d["feature"], d["importance"],
                    color=[C.SERIES[1] if f in behav else C.SERIES[0] for f in d["feature"]])
            ax.tick_params(axis="y", labelsize=8.5)
            ax.set_xlabel("Önem (toplam = 1)", color=C.INK2)
        axes[0][0].text(0.99, 0.02, "turuncu = Öğrenci 3'ün ürettiği özellikler", transform=axes[0][0].transAxes,
                        ha="right", fontsize=8, color=C.SERIES[1])
        C.save_fig(fig, out / "feature_importance.png")

    # 6) Eğitim süresi
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.SURFACE)
    C.style_ax(ax, "Model eğitim süresi (dengelenmiş train)", xgrid=True, ygrid=False)
    y = np.arange(len(models))
    h = 0.8 / len(sizes)
    for i, s in enumerate(sizes):
        sub = tuned[tuned["size"] == s].set_index("model").reindex(models)
        ax.barh(y + (i - (len(sizes) - 1) / 2) * h, sub["train_sec"], h, color=size_col[s],
                edgecolor=C.SURFACE, label=f"{C.size_label(s)} ({int(sub['train_rows'].max()):,} satır)")
    ax.set_yticks(y, [MODEL_NAMES[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("Eğitim süresi (sn)", color=C.INK2)
    ax.legend(title="Veri boyutu (train satırı)", frameon=False, fontsize=9, title_fontsize=9, labelcolor=C.INK2)
    C.save_fig(fig, out / "training_time.png")
    log.info(f"[f] Grafikler -> {out}")


# --------------------------------------------------------------------------- #
# Yorum (sayılarla)
# --------------------------------------------------------------------------- #
def write_summary(res, imp, out: Path):
    big = res["size"].max()
    rb = res[res["size"] == big]
    tuned = rb[rb["threshold_type"] == "tuned"].sort_values("auprc", ascending=False)
    half = rb[rb["threshold_type"] == "0.5"].set_index("model")
    if tuned.empty:
        return
    best = tuned.iloc[0]
    naive = rb[rb["model"] == "naive"]
    pct = lambda v: f"%{100 * v:.1f}"  # noqa: E731
    name = lambda m: MODEL_NAMES.get(m, m)  # noqa: E731

    L = [f"# Öğrenci 4 - Model karşılaştırması ({C.size_label(big)}, girdi: {best['input']})", "",
         f"Train: Öğrenci 2'nin seçtiği **{best['method']}** ile dengelenmiş veri "
         f"({int(best['train_rows']):,} satır, {int(best['train_fraud']):,} fraud). "
         f"Test: gerçek dağılım ({int(best['test_rows']):,} satır, {int(best['test_fraud']):,} fraud, "
         f"{pct(best['test_fraud'] / best['test_rows'])}). Eşik, train'den ayrılan doğrulama kümesinde "
         "gerçek dağılıma göre düzeltilmiş F1'i en yüksek yapan değer; test seti eşik seçiminde kullanılmadı.", "",
         "| Model | Eşik | Accuracy | Precision | Recall | F1 | AUPRC | AUROC | TP | FP | FN | Eğitim (sn) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in naive.iterrows():
        L.append(f"| {name('naive')} | – | {r['accuracy']:.4f} | 0 | 0 | 0 | – | – | 0 | 0 | {int(r['FN']):,} | – |")
    for _, r in rb[rb["model"] != "naive"].sort_values(["model", "threshold_type"]).iterrows():
        L.append(f"| {name(r['model'])} | {r['threshold']:.3f} | {r['accuracy']:.4f} | {r['precision']:.3f} | "
                 f"{r['recall']:.3f} | {r['f1']:.3f} | {r['auprc']:.3f} | {r['auroc']:.3f} | {int(r['TP']):,} | "
                 f"{int(r['FP']):,} | {int(r['FN']):,} | {r['train_sec']:.1f} |")
    L += ["", "## Yorum", ""]
    if not naive.empty:
        L.append(f"- **Accuracy yanıltıcı:** hiçbir fraud'u yakalamayan naif model accuracy "
                 f"{pct(naive.iloc[0]['accuracy'])} alıyor. Bu yüzden karşılaştırmada AUPRC, F1, precision ve "
                 "recall kullanıldı.")
    L.append(f"- **En iyi model: {name(best['model'])}** (AUPRC {best['auprc']:.3f}; ayarlı eşikte "
             f"precision {best['precision']:.3f}, recall {best['recall']:.3f}, F1 {best['f1']:.3f}). "
             f"Test setindeki {int(best['test_fraud']):,} fraud'un {int(best['TP']):,} tanesi yakalandı, "
             f"{int(best['FP']):,} yanlış alarm verildi.")
    for _, r in tuned.iloc[1:].iterrows():
        L.append(f"- {name(r['model'])}: AUPRC {r['auprc']:.3f}, F1 {r['f1']:.3f} "
                 f"(eğitim {r['train_sec']:.1f} sn).")
    best_f1 = tuned.sort_values("f1", ascending=False).iloc[0]
    if best_f1["model"] != best["model"]:
        L.append(f"- Not: ayarlı eşikte F1'i en yüksek model **{name(best_f1['model'])}** (F1 {best_f1['f1']:.3f}). "
                 "AUPRC tüm eşikleri özetler, F1 tek bir çalışma noktasını; sıralama bu yüzden farklı olabilir.")
    t_ = tuned.set_index("model")
    if {"lr", "lr_fe"} <= set(t_.index):
        a, b = t_.loc["lr"], t_.loc["lr_fe"]
        L.append(f"- **LR'yi zayıflatan şey modelin kendisi değil, özellikler:** aynı Logistic Regression'a "
                 f"tutar, saat, yaş vb. {N_BUCKETS} aralığa bölünmüş (one-hot) ve log_amt x kategori etkileşimi "
                 f"verilince AUPRC {a['auprc']:.3f} → {b['auprc']:.3f}, F1 {a['f1']:.3f} → {b['f1']:.3f}. Doğrusal model "
                 "bir sütuna tek ağırlık verir, yani 'tutar arttıkça risk hep artar (ya da hep azalır)' varsayar. Oysa "
                 "fraud oranı tutarla inişli çıkışlı değişir (Sparkov 1M: 10-20$ arasında %0.7, 50-100$ arasında %0.01, "
                 "800-1000$ arasında %53, 1000$ üstünde %22). Böyle bir ilişkiyi LR ancak her aralığa ayrı ağırlık "
                 "verilince öğrenebilir; ağaç modelleri bunu kendiliğinden yapar.")
    trees = [m for m in ("gbt", "rf") if m in t_.index]
    if "lr" in t_.index and trees and "lr_fe" not in t_.index:
        bt = max(trees, key=lambda m: t_.loc[m, "auprc"])
        a_lr, a_bt = t_.loc["lr", "auprc"], t_.loc[bt, "auprc"]
        if a_bt > a_lr:
            L.append(f"- LR → {name(bt)}: AUPRC {a_lr:.3f} → {a_bt:.3f}. Ağaç modelleri 'yüksek tutar × gece × "
                     "kategori' gibi doğrusal olmayan etkileşimleri kendiliğinden yakalar; LR her özelliğe tek "
                     "bir ağırlık verir.")
        else:
            L.append(f"- LR, ağaç modellerinden geride değil (LR AUPRC {a_lr:.3f}, en iyi ağaç {a_bt:.3f}): bu veride "
                     "fraud sinyali büyük ölçüde doğrusal ayrılabiliyor (ör. tutar ve gece). Ağaç parametreleri "
                     "(--gbt_iter, --rf_depth) artırılarak tekrar denenebilir.")
    for m in tuned["model"]:
        if m in half.index:
            h, t = half.loc[m], tuned.set_index("model").loc[m]
            L.append(f"- **Eşik etkisi ({name(m)}):** 0.5'te P={h['precision']:.3f} R={h['recall']:.3f} "
                     f"F1={h['f1']:.3f} (FP {int(h['FP']):,}); eşik {t['threshold']:.3f} iken P={t['precision']:.3f} "
                     f"R={t['recall']:.3f} F1={t['f1']:.3f} (FP {int(t['FP']):,}). "
                     + ("Dengelenmiş veriyle eğitilen model fraud olasılığını gerçekte olduğundan yüksek verir; "
                        "eşiği yükseltmek yanlış alarmları azaltır. " if t['f1'] > h['f1'] + 0.01 else
                        "Bu modelde eşik ayarı belirgin bir kazanç sağlamadı: sorun eşikte değil, modelin "
                        "fraud'u normalden ayırma gücünde (düşük AUPRC). ")
                     + "AUPRC eşikten bağımsızdır, değişmez.")
    if not imp.empty:
        for m in ("gbt", "rf"):
            d = imp[(imp["size"] == big) & (imp["model"] == m)].nlargest(5, "importance")
            if not d.empty:
                behav = [f for f in d["feature"] if f in C.BEHAVIOR_FEATURES]
                L.append(f"- **{name(m)} en önemli 5 özellik:** "
                         + ", ".join(f"`{f}` ({v:.2f})" for f, v in zip(d["feature"], d["importance"]))
                         + (f". Bunlardan {len(behav)} tanesi Öğrenci 3'ün ürettiği özellik." if behav else "."))
    sizes = sorted(res["size"].unique())
    if len(sizes) > 1:
        tr = res[(res["threshold_type"] == "tuned") & (res["model"] == best["model"])].sort_values("size")
        L.append(f"- **Veri boyutuna göre ({name(best['model'])}):** "
                 + ", ".join(f"{C.size_label(s)}: AUPRC {a:.3f}, {t:.1f} sn"
                             for s, a, t in zip(tr["size"], tr["auprc"], tr["train_sec"]))
                 + ". (Boyut etkisinin ayrıntılı analizi: Öğrenci 5.)")
    tpath = out / "tuning_results.csv"
    if tpath.exists():
        import pandas as pd
        tu = pd.read_csv(tpath)
        tu = tu[(tu["input"] == best["input"]) & (tu["size"] == big)].drop_duplicates(["model", "params"], keep="last")
        if not tu.empty:
            L += ["", "## Ayar araması (hiperparametre)", "",
                  "RF ve GBT için birkaç ayar denendi; her biri train'den ayrılan doğrulama kümesinde, gerçek dağılıma "
                  "göre düzeltilmiş AUPRC ile değerlendirildi ve en iyisi seçildi. Test seti bu seçimde kullanılmadı.", "",
                  "| Model | Ayar | Doğrulama AUPRC | Eğitim (sn) |", "|---|---|---:|---:|"]
            for m in ("rf", "gbt"):
                g = tu[tu["model"] == m]
                if g.empty:
                    continue
                top = g["val_auprc"].max()
                for _, r in g.iterrows():
                    mark = " **(seçilen)**" if r["val_auprc"] == top else ""
                    L.append(f"| {name(m)} | {r['params']}{mark} | {r['val_auprc']:.3f} | {r['fit_sec']:.1f} |")
            for m in ("rf", "gbt"):
                g = tu[tu["model"] == m].reset_index(drop=True)
                if len(g) > 1:
                    d0, b_ = g.iloc[0], g.loc[g["val_auprc"].idxmax()]
                    L.append(f"- {name(m)}: varsayılan ayar ({d0['params']}) doğrulama AUPRC {d0['val_auprc']:.3f} → "
                             f"seçilen ({b_['params']}) {b_['val_auprc']:.3f}; eğitim süresi {d0['fit_sec']:.0f} → "
                             f"{b_['fit_sec']:.0f} sn.")
                    if b_.name == len(g) - 1:
                        L.append(f"  En iyi ayar denenen aralığın en ucunda; daha fazla ağaç/derinlik biraz daha "
                                 "kazanç sağlayabilir ama eğitim süresi de artar.")
            small = sorted(res["size"].unique())[0]
            if small != big:
                vf = int(res[res["size"] == small]["val_fraud"].max())
                L.append(f"- Not: {C.size_label(small)} veride doğrulama kümesi çok küçük ({vf} fraud); orada "
                         "seçilen ayar şansa bağlı olabilir. Ayar sonuçlarının güvenilir olduğu boyutlar 1M ve 5M.")
    L += ["", "Grafikler: `metrics_by_model.png`, `pr_curves.png`, `threshold_analysis.png`, "
          "`confusion_matrices.png`, `feature_importance.png`, `training_time.png`."]
    (out / "model_secimi.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info(f"[f] Yorum -> {out / 'model_secimi.md'}")


# --------------------------------------------------------------------------- #
def main():
    global log
    ap = argparse.ArgumentParser(description="Öğrenci 4 - fraud sınıflandırma modelleri")
    ap.add_argument("--sizes", type=int, nargs="+", default=C.DEFAULT_SIZES)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--driver_memory", default="8g")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "ogrenci4"))
    ap.add_argument("--lr_iter", type=int, default=100)
    ap.add_argument("--lr_reg", type=float, default=0.0)
    ap.add_argument("--rf_trees", type=int, default=100)
    ap.add_argument("--rf_depth", type=int, default=10)
    ap.add_argument("--gbt_iter", type=int, default=50)
    ap.add_argument("--gbt_depth", type=int, default=5)
    ap.add_argument("--tune", action="store_true",
                    help="RF ve GBT için GRID'deki ayarları dene, doğrulama AUPRC'sine göre en iyisini seç")
    ap.add_argument("--plot_only", action="store_true", help="Spark başlatmadan CSV'lerden grafik + yorum")
    ap.add_argument("--force", action="store_true", help="CSV'de olan modelleri de yeniden eğit")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = C.PROJECT_ROOT / out
    log = C.setup_logging(out, "ogrenci4")

    if not args.plot_only:
        sizes = sorted(args.sizes)
        meta = read_meta()
        check_input(meta, sizes)
        log.info(f"Öğrenci 4 başladı | sizes={sizes} models={args.models} | dengeleme: {meta.get('method')} "
                 f"{meta.get('target_ratio')} | input={meta.get('input')} | weight_col={meta.get('weight_col')} "
                 f"| pid={os.getpid()}")
        spark = C.get_spark("ogrenci4-modelleme", args.driver_memory)
        try:
            for size in sizes:
                log.info(f"===== {size:,} kayıt =====")
                with C.Timer() as t:
                    n = run_size(spark, size, args.models, meta, args, out)
                log.info(f"===== {size:,} bitti: {n} model eğitildi, {t.sec / 60:.1f} dk =====")
        except BaseException as e:   # noqa: BLE001  (KeyboardInterrupt dahil: log'da kalsın)
            log.exception(f"Çalıştırma kesildi: {type(e).__name__}: {e}")
            raise
        finally:
            spark.stop()

    res, curve, imp = read_results(out)
    if res.empty:
        log.error(f"{out / 'model_results.csv'} boş ya da yok; önce modelleri çalıştırın.")
        sys.exit(1)
    C.safe_plot(make_plots, res, curve, imp, out, log=log.warning)
    C.safe_plot(write_summary, res, imp, out, log=log.warning)
    log.info("Öğrenci 4 tamamlandı.")


if __name__ == "__main__":
    main()
