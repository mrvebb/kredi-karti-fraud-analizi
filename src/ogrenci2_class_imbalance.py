"""
Öğrenci 2 - Class Imbalance Analizi ve Dengeleme
================================================

Soru: Fraud oranı ~%0.5 olan veride dengesizlik modeli nasıl etkiler, hangi dengeleme
yöntemi uygun? Veri büyüdükçe (100K -> 1M -> 5M) bu sorun kendiliğinden çözülür mü?

Girdi (--input, varsayılan auto):
    features : data/features/fraud_<n>.parquet  (Öğrenci 3 - window özellikleri dengelemeden ÖNCE
               üretilmiş; sütunlar common.MODEL_FEATURES)
    scaled   : data/scaled/fraud_<n>.parquet    (Öğrenci 1) + temel özellikler (common.NUMERIC_FEATURES)
    auto     : tüm boyutlar için Öğrenci 3 çıktısı hazırsa features, değilse scaled.
    Sonuç CSV'sinde "input" sütunu var: Öğrenci 3 sonradan teslim edince aynı komut özellikli
    veriyle deneyleri yeniden yapar, eski (scaled) sonuçlar silinmez, karışmaz.

Akış (her veri boyutu için)
---------------------------
a) Sınıf dağılımı (tüm veri / train / test)      -> results/ogrenci2/class_distribution.csv
b) Train/test ayrımı (~%80/%20, sınıf oranı korunur, common.group_split).
   * Dengeleme SADECE train setine uygulanır; test gerçek dağılımda kalır.
   * Özellik pipeline'ı (StringIndexer/OneHot/StandardScaler) SADECE train'e fit edilir.
c) Yöntemler : none (baseline), class_weight, undersample, oversample
d) Hedef oran: --target_ratios 1 10  ->  fraud:normal = 1:1 ve 1:10
e) Aynı Logistic Regression ile metrikler     -> results/ogrenci2/imbalance_results.csv
f) Grafikler                                   -> results/ogrenci2/*.png
g) En büyük boyutta --select_metric'e göre en iyi yöntem seçilir; gerekçe sayılarla
   results/ogrenci2/secim_gerekcesi.md'ye yazılır. HER boyut için dengelenmiş train +
   gerçek dağılımlı test yazılır -> data/balanced/train_n<n>.parquet, test_n<n>.parquet,
   SECILEN_YONTEM.json   (Öğrenci 4 ve Öğrenci 5 için)

Kesintiye dayanıklılık
----------------------
* Her (size, method, target_ratio) deneyi biter bitmez imbalance_results.csv'ye bir satır eklenir.
* Tekrar çalıştırınca CSV'de olan deneyler atlanır (resume). Hepsini yeniden koşmak: --force
* data/balanced/ altında eksiksiz (_SUCCESS'li) ve aynı yöntemle yazılmış çıktı yeniden yazılmaz.
* Zaman damgalı log: konsol + <out>/run.log

Kullanım:
    python src/ogrenci2_class_imbalance.py                       # 100K, 1M, 5M
    python src/ogrenci2_class_imbalance.py --sizes 5000000 --driver_memory 8g
    python src/ogrenci2_class_imbalance.py --plot_only           # Spark'sız, CSV'den grafikler
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
from pyspark.ml.classification import LogisticRegression  # noqa: E402
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.sql import DataFrame  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

LABEL = C.LABEL_COL
METHODS = ["none", "class_weight", "undersample", "oversample"]
WEIGHT_COL = "class_weight"
KEYS = ["input", "size", "method", "target_ratio"]
REQUIRED = ["auprc", "train_sec"]
RESULT_COLS = KEYS + ["train_rows", "train_normal", "train_fraud", "test_rows", "test_fraud",
                      "TP", "FP", "FN", "TN", "accuracy", "precision", "recall", "f1", "auprc",
                      "balance_sec", "train_sec", "finished_at"]
log = None


def read_results(path: Path, input_kind: str | None = None):
    """Sonuçlar; input_kind verilirse sadece o girdiyle yapılan deneyler.
    input_kind="auto": features sonuçları varsa onlar, yoksa scaled."""
    df = C.read_rows(path, KEYS, REQUIRED, RESULT_COLS)
    if df.empty:
        return df
    df["size"] = df["size"].astype(int)
    if input_kind == "auto":
        input_kind = "features" if (df["input"] == "features").any() else "scaled"
    return df[df["input"] == input_kind].reset_index(drop=True) if input_kind else df


def resolve_input(input_kind: str, sizes: list[int]) -> str:
    """auto -> tüm boyutlar için Öğrenci 3 çıktısı eksiksizse features, değilse scaled.
    (Boyutlar arasında karışık girdi olmasın diye tek karar verilir.)"""
    if input_kind != "auto":
        return input_kind
    ready = [s for s in sizes if C.is_complete(C.features_path(s))]
    if len(ready) == len(sizes):
        return "features"
    if ready:
        log.warning(f"Öğrenci 3 çıktısı sadece {ready} için hazır; tutarlılık için hepsinde scaled kullanılıyor")
    return "scaled"


# --------------------------------------------------------------------------- #
# a) Sınıf dağılımı
# --------------------------------------------------------------------------- #
def distribution_row(df: DataFrame, size: int, split: str) -> dict:
    n0, n1 = C.class_counts(df)
    total = n0 + n1
    return {"size": size, "split": split, "total": total, "normal": n0, "fraud": n1,
            "fraud_pct": round(100 * n1 / total, 4) if total else 0.0,
            "imbalance_ratio": round(n0 / n1, 2) if n1 else float("inf")}


# --------------------------------------------------------------------------- #
# c/d) Dengeleme - sadece train'e uygulanır
# --------------------------------------------------------------------------- #
def ratio_label(ratio: int | None) -> str:
    return "-" if ratio is None else f"1:{ratio}"


def balance(train: DataFrame, method: str, ratio: int | None, n_normal: int, n_fraud: int,
            seed: int = C.SEED) -> tuple[DataFrame, str | None]:
    """Dengelenmiş train ve (varsa) ağırlık sütunu adı.

    ratio = hedef normal/fraud oranı (1 -> 1:1, 10 -> 1:10).
    """
    is_fraud = F.col(LABEL) == 1
    if method == "none":
        return train, None

    if method == "class_weight":
        # fraud ağırlığı = normal/fraud (1:1). 1:10 için etkin oran normal:fraud = 10:1 olacak şekilde
        w = n_normal / (n_fraud * ratio)
        return train.withColumn(WEIGHT_COL, F.when(is_fraud, F.lit(w)).otherwise(F.lit(1.0))), WEIGHT_COL

    fraud, normal = train.filter(is_fraud), train.filter(~is_fraud)
    if method == "undersample":
        # Tüm fraud'lar kalır, normaller fraud*ratio kadar rastgele seçilir
        frac = min(1.0, n_fraud * ratio / n_normal)
        return fraud.unionByName(normal.sample(False, frac, seed)), None

    if method == "oversample":
        # Tüm normaller kalır, fraud'lar iadeli örneklemeyle normal/ratio sayısına çıkarılır
        target = n_normal / ratio
        extra = max(0.0, target - n_fraud)
        if extra == 0:
            return train, None
        dup = fraud.sample(True, extra / n_fraud, seed)
        return normal.unionByName(fraud).unionByName(dup), None

    raise ValueError(f"Bilinmeyen yöntem: {method}")


# --------------------------------------------------------------------------- #
# e) Değerlendirme (Öğrenci 4 ve 5 de kullanır)
# --------------------------------------------------------------------------- #
def evaluate(pred: DataFrame) -> dict:
    cm = {(int(r[LABEL]), int(r["prediction"])): r["count"]
          for r in pred.groupBy(LABEL, "prediction").count().collect()}
    tp, fp = cm.get((1, 1), 0), cm.get((0, 1), 0)
    fn, tn = cm.get((1, 0), 0), cm.get((0, 0), 0)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auprc = BinaryClassificationEvaluator(labelCol=LABEL, rawPredictionCol="rawPrediction",
                                          metricName="areaUnderPR").evaluate(pred)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": round((tp + tn) / total, 6), "precision": round(precision, 6),
            "recall": round(recall, 6), "f1": round(f1, 6), "auprc": round(auprc, 6)}


# --------------------------------------------------------------------------- #
# Veri hazırlama + bir veri boyutu için tüm deneyler
# --------------------------------------------------------------------------- #
def experiments(methods: list[str], ratios: list[int]) -> list[tuple[str, int | None]]:
    return [(m, r) for m in methods for r in ([None] if m == "none" else ratios)]


def feature_cols(input_kind: str) -> list[str]:
    return C.MODEL_FEATURES if input_kind == "features" else C.NUMERIC_FEATURES


def load_input(spark, size: int, input_kind: str, holder: dict) -> DataFrame:
    if input_kind == "features":
        return C.load_features(spark, size, log=log.info)
    if not C.is_complete(C.scaled_path(size, "parquet")) and "raw" not in holder:
        holder["raw"] = C.read_raw(spark).cache()
    return C.add_basic_features(
        C.load_or_build(spark, size, "parquet", holder.get("raw"), persist=True, log=log.info))


def prepare_size(spark, size: int, input_kind: str, holder: dict, out: Path) -> dict:
    """Girdiyi okur, train/test'e böler. Sonuç holder'da tutulur (aynı boyut tekrar okunmaz)."""
    if holder.get("data", {}).get("size") == size:
        return holder["data"]
    release(holder)
    df = load_input(spark, size, input_kind, holder).cache()

    # b) Train/test ayrımı (kopyalar aynı tarafta, sınıf oranı korunur)
    train, test = C.group_split(df, 0.8)
    train, test = train.cache(), test.cache()
    dist = [distribution_row(df, size, "all"), distribution_row(train, size, "train"),
            distribution_row(test, size, "test")]
    C.merge_csv(dist, out / "class_distribution.csv", ["size", "split"])
    for d in dist:
        log.info(f"  {d['split']:5s}: {d['total']:>10,} satır | fraud {d['fraud']:>7,} "
                 f"(%{d['fraud_pct']:.3f}) | imbalance 1:{d['imbalance_ratio']:.0f}")
    holder["data"] = {"size": size, "df": df, "train": train, "test": test, "dist": dist,
                      "n_normal": dist[1]["normal"], "n_fraud": dist[1]["fraud"]}
    return holder["data"]


def release(holder: dict):
    data = holder.pop("data", None)
    if data:
        for k in ("df", "train", "test"):
            data[k].unpersist()


def run_size(spark, size: int, methods: list[str], ratios: list[int], input_kind: str, holder: dict,
             out: Path, max_iter: int, force: bool = False) -> int:
    """Bu boyuttaki EKSİK deneyleri çalıştırır; çalıştırılan deney sayısını döner."""
    res_path = out / "imbalance_results.csv"
    done = set() if force else C.done_set(res_path, KEYS, REQUIRED)
    todo = [(m, r) for m, r in experiments(methods, ratios)
            if (input_kind, str(size), m, ratio_label(r)) not in done]
    skipped = len(experiments(methods, ratios)) - len(todo)
    if skipped:
        log.info(f"  {skipped} deney CSV'de zaten var -> atlanıyor")
    if not todo:
        log.info("  Bu boyutta eksik deney yok.")
        return 0

    data = prepare_size(spark, size, input_kind, holder, out)
    n_normal, n_fraud, dist = data["n_normal"], data["n_fraud"], data["dist"]

    # Özellik pipeline'ı SADECE train'e fit edilir (scaler test istatistiklerini görmez)
    log.info("  Özellik pipeline'ı train'e fit ediliyor")
    feat_model = Pipeline(stages=C.build_feature_stages(feature_cols(input_kind))).fit(data["train"])
    train_f = feat_model.transform(data["train"]).select("features", LABEL).cache()
    test_f = feat_model.transform(data["test"]).select("features", LABEL).cache()
    train_f.count(), test_f.count()

    for i, (method, ratio) in enumerate(todo, 1):
        tag = f"{method} {ratio_label(ratio)}"
        log.info(f"  [{i}/{len(todo)}] BAŞLADI  size={size:,} {tag}")
        with C.Timer() as tb:
            bal, wcol = balance(train_f, method, ratio, n_normal, n_fraud)
            bal = bal.cache()
            b0, b1 = C.class_counts(bal)
        lr = LogisticRegression(featuresCol="features", labelCol=LABEL, maxIter=max_iter)
        if wcol:
            lr.setWeightCol(wcol)
        with C.Timer() as tt:
            model = lr.fit(bal)
        metrics = evaluate(model.transform(test_f))
        bal.unpersist()
        row = {"input": input_kind, "size": size, "method": method, "target_ratio": ratio_label(ratio),
               "train_rows": b0 + b1, "train_normal": b0, "train_fraud": b1,
               "test_rows": dist[2]["total"], "test_fraud": dist[2]["fraud"], **metrics,
               "balance_sec": round(tb.sec, 2), "train_sec": round(tt.sec, 2),
               "finished_at": C.now_iso()}
        # Her deneyden hemen sonra diske ekle: kesinti olursa sadece o anki deney kaybolur
        C.append_row(row, res_path, RESULT_COLS)
        log.info(f"  [{i}/{len(todo)}] BİTTİ    size={size:,} {tag} | train={b0 + b1:,} | "
                 f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                 f"AUPRC={metrics['auprc']:.3f} | denge {tb.sec:.1f} sn, fit {tt.sec:.1f} sn")

    for d in (train_f, test_f):
        d.unpersist()
    return len(todo)


# --------------------------------------------------------------------------- #
# g) Seçilen yöntemle dengelenmiş veriyi kaydet (her boyut)
# --------------------------------------------------------------------------- #
TIE_TOL = 0.005   # metrik farkı bundan küçükse yöntemler "eşit" sayılır


def choose_method(results, select_metric: str, size: int, tol: float = TIE_TOL):
    """En iyi metrik değerine tol'dan yakın yöntemler eşit sayılır; aralarından en kısa
    eğitim süreli (en ucuz) olan seçilir. Gürültü düzeyindeki farkla pahalı yöntem seçilmesin."""
    cand = results[(results["size"] == size) & (results["method"] != "none")]
    if cand.empty:
        return None
    top = cand[select_metric].max()
    tied = cand[cand[select_metric] >= top - tol]
    best = tied.sort_values(["train_sec", select_metric], ascending=[True, False]).iloc[0]
    return best["method"], int(best["target_ratio"].split(":")[1]), float(best[select_metric])


def save_balanced(spark, results, select_metric: str, sizes: list[int], input_kind: str,
                  holder: dict, out: Path, force: bool = False):
    choice = choose_method(results, select_metric, sizes[-1])
    if choice is None:
        log.warning("[g] Dengeleme sonucu yok -> data/balanced/ atlandı")
        return None
    method, ratio, score = choice
    log.info(f"[g] {sizes[-1]:,} boyutunda en iyi yöntem ({select_metric}): {method} 1:{ratio} "
             f"({select_metric}={score:.4f}) -> tüm boyutlara uygulanıyor")

    write_justification(results, select_metric, sizes[-1], method, ratio, input_kind, out)
    meta_path = C.BALANCED_DIR / "SECILEN_YONTEM.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    same_choice = (old.get("method") == method and old.get("target_ratio") == f"1:{ratio}"
                   and old.get("input") == input_kind)
    per_size = old.get("sizes", {}) if same_choice else {}
    C.BALANCED_DIR.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        train_path = C.BALANCED_DIR / f"train_n{size}.parquet"
        test_path = C.BALANCED_DIR / f"test_n{size}.parquet"
        if (not force and same_choice and str(size) in per_size
                and C.is_complete(train_path) and C.is_complete(test_path)):
            log.info(f"[g] {size:,}: dengelenmiş veri zaten eksiksiz -> atlandı")
            continue
        # NOT: window özellikleri bu adımdan ÖNCE (Öğrenci 3) orijinal veri üzerinde hesaplandı.
        log.info(f"[g] {size:,}: BAŞLADI  dengelenmiş train/test yazılıyor")
        data = prepare_size(spark, size, input_kind, holder, out)
        bal, _ = balance(data["train"], method, ratio, data["n_normal"], data["n_fraud"])
        with C.Timer() as t:
            bal.write.mode("overwrite").parquet(str(train_path))
            data["test"].write.mode("overwrite").parquet(str(test_path))
        b0, b1 = C.class_counts(spark.read.parquet(str(train_path)))
        per_size[str(size)] = {"train_path": str(train_path), "test_path": str(test_path),
                               "train_normal": b0, "train_fraud": b1,
                               "test_rows": data["dist"][2]["total"], "test_fraud": data["dist"][2]["fraud"]}
        log.info(f"[g] {size:,}: BİTTİ    train normal={b0:,} fraud={b1:,} | {t.sec:.1f} sn")
        _write_meta(meta_path, method, ratio, select_metric, score, sizes[-1], results, per_size, input_kind)

    _write_meta(meta_path, method, ratio, select_metric, score, sizes[-1], results, per_size, input_kind)
    return method, ratio


def write_justification(results, metric: str, sel_size: int, method: str, ratio: int, input_kind: str,
                        out: Path):
    """Yöntem seçiminin gerekçesini sonuçlardaki sayılarla markdown olarak yazar."""
    r = results[results["size"] == sel_size].copy()
    base = r[r["method"] == "none"].iloc[0] if (r["method"] == "none").any() else None
    sel = r[(r["method"] == method) & (r["target_ratio"] == f"1:{ratio}")].iloc[0]
    cand = r[r["method"] != "none"]
    name = lambda x: f"{x['method']} {x['target_ratio']}"  # noqa: E731
    pct = lambda v: f"%{100 * v:.1f}"  # noqa: E731

    top = cand[metric].max()
    tied = cand[cand[metric] >= top - TIE_TOL]
    lines = [f"# Öğrenci 2 - Dengeleme yöntemi seçimi ({C.size_label(sel_size)}, girdi: {input_kind})", "",
             f"Seçim ölçütü: **{metric.upper()}** (test seti gerçek dağılımda, eşik 0.5). "
             "F1, precision ile recall'u birlikte dikkate aldığı için hem kaçırılan fraud'u hem de "
             f"yanlış alarmı cezalandırır. En iyi değere {TIE_TOL} puandan yakın yöntemler eşit "
             "sayılır ve aralarından **en kısa sürede eğitilen** seçilir (gürültü düzeyindeki bir fark "
             "için daha pahalı yöntemi seçmemek için).", "",
             "| Yöntem | Train satırı | Precision | Recall | F1 | AUPRC | Eğitim (sn) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for _, x in r.sort_values(metric, ascending=False).iterrows():
        mark = " **(seçilen)**" if (x["method"], x["target_ratio"]) == (method, f"1:{ratio}") else ""
        lines.append(f"| {name(x)}{mark} | {int(x['train_rows']):,} | {x['precision']:.3f} | {x['recall']:.3f} "
                     f"| {x['f1']:.3f} | {x['auprc']:.3f} | {x['train_sec']:.1f} |")
    lines += ["", "## Gerekçe", ""]
    if base is not None:
        lines.append(f"- **Dengeleme yokken** accuracy {pct(base['accuracy'])} ama recall yalnızca "
                     f"{pct(base['recall'])}: test setindeki fraud'ların kaçırılan payı {pct(1 - base['recall'])}. "
                     "Yüksek accuracy, modelin neredeyse her şeye 'normal' demesinden geliyor.")
        lines.append(f"- **Seçilen yöntem ({method} 1:{ratio})** ile recall {pct(base['recall'])} → "
                     f"{pct(sel['recall'])}, precision {pct(base['precision'])} → {pct(sel['precision'])}, "
                     f"F1 {base['f1']:.3f} → {sel['f1']:.3f}.")
    if len(tied) > 1:
        others = ", ".join(f"{name(x)} ({x[metric]:.3f}, {x['train_sec']:.1f} sn, {int(x['train_rows']):,} satır)"
                           for _, x in tied.iterrows()
                           if (x["method"], x["target_ratio"]) != (method, f"1:{ratio}"))
        lines.append(f"- **Eşitlik:** {metric.upper()} açısından {len(tied)} yöntem birbirine {TIE_TOL} puandan "
                     f"yakın. Seçilen {method} 1:{ratio} ({sel[metric]:.3f}, {sel['train_sec']:.1f} sn, "
                     f"{int(sel['train_rows']):,} satır) bunların en hızlı eğitileni. Diğerleri: {others}.")
    best_rec = cand.sort_values("recall", ascending=False).iloc[0]
    if (best_rec["method"], best_rec["target_ratio"]) != (method, f"1:{ratio}"):
        lines.append(f"- En yüksek recall **{name(best_rec)}** ile ({pct(best_rec['recall'])}), ama precision "
                     f"{pct(best_rec['precision'])}: fraud alarmlarında yanlış alarm payı "
                     f"{pct(1 - best_rec['precision'])}. Bu yüzden tek başına recall'a göre seçilmedi.")
    lo, hi = cand["auprc"].min(), cand["auprc"].max()
    if base is not None:
        lo, hi = min(lo, base["auprc"]), max(hi, base["auprc"])
    if hi - lo < 0.05:
        lines.append(f"- AUPRC tüm yöntemlerde birbirine yakın ({lo:.3f}–{hi:.3f}). Dengeleme, modelin fraud'u "
                     "sıralama yeteneğini pek değiştirmiyor; asıl etkisi karar eşiğini kaydırmak. Bu yüzden "
                     "Öğrenci 4'ün eşik ayarı da benzer bir kazanç sağlayabilir.")
    else:
        lines.append(f"- AUPRC yöntemler arasında {lo:.3f}–{hi:.3f} aralığında değişiyor; seçilen yöntemde "
                     f"{sel['auprc']:.3f}.")
    if base is not None and base["auprc"] >= cand["auprc"].max():
        lines.append(f"- **AUPRC en yüksek dengeleme yokken** ({base['auprc']:.3f}). Yani dengeleme modelin "
                     "fraud'ları normallerden ayırma (sıralama) yeteneğini artırmıyor; sadece 0.5 eşiğinin "
                     "nereye düştüğünü değiştiriyor. Baseline'ın recall'unun düşük (%"
                     f"{base['recall'] * 100:.1f}) olması bilgisizlikten değil, eşiğin dengesiz veri için fazla "
                     "yüksek kalmasından. Dengelemenin pratik faydası, eşik ayarı yapmadan da kullanılabilir bir "
                     "precision/recall dengesi vermesi.")
    if base is not None:
        lines.append(f"- **Maliyet:** seçilen yöntemin train seti baseline'ın {sel['train_rows'] / base['train_rows']:.2f} "
                     f"katı, eğitim süresi {sel['train_sec'] / max(base['train_sec'], 1e-9):.2f} katı.")
    # Seçim diğer boyutlarda da tutarlı mı?
    ranks = []
    for s, g in results.groupby("size"):
        g = g[g["method"] != "none"].sort_values(metric, ascending=False).reset_index(drop=True)
        hit = g[(g["method"] == method) & (g["target_ratio"] == f"1:{ratio}")]
        if not hit.empty:
            ranks.append(f"{C.size_label(s)}: {hit.index[0] + 1}/{len(g)}")
    if ranks:
        lines.append(f"- Diğer boyutlarda seçilen yöntemin {metric.upper()} sırası: {', '.join(ranks)}.")
    lines += ["", "Dengeleme sadece train setine uygulandı; test seti gerçek dağılımda bırakıldı. "
              "Dengelenmiş veri: `data/balanced/` (bkz. SECILEN_YONTEM.json)."]
    (out / "secim_gerekcesi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"[g] Seçim gerekçesi -> {out / 'secim_gerekcesi.md'}")


def _write_meta(path, method, ratio, metric, score, sel_size, results, per_size, input_kind):
    sub = results[results["size"] == sel_size][["method", "target_ratio", "precision", "recall",
                                                "f1", "auprc", "train_rows", "train_sec"]]
    meta = {"method": method, "target_ratio": f"1:{ratio}", "select_metric": metric, "score": score,
            "selected_on_size": int(sel_size), "input": input_kind,
            "feature_cols": feature_cols(input_kind),
            "weight_col": WEIGHT_COL if method == "class_weight" else None,
            "sizes": per_size,
            "comparison_on_selected_size": sub.to_dict(orient="records"),
            "note": "Test seti dengelenmedi (gerçek dağılım). Window özellikleri dengelemeden önce "
                    "(Öğrenci 3) hesaplandı. Train/test ayrımı orig_id hash'ine göre: sentetik "
                    "kopyalar orijinaliyle aynı tarafta."}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)   # atomik: yarım JSON kalmaz


# --------------------------------------------------------------------------- #
# f) Grafikler
# --------------------------------------------------------------------------- #
def make_plots(res, out: Path):
    import numpy as np
    import pandas as pd
    plt = C.plt_setup()

    res = res.copy()
    res["label"] = res.apply(lambda r: r["method"] if r["target_ratio"] == "-"
                             else f"{r['method']}\n{r['target_ratio']}", axis=1)
    ratios = sorted({r for r in res["target_ratio"] if r != "-"}, key=lambda s: int(s.split(":")[1]))
    order = ["none"] + [f"{m}\n{r}" for m in METHODS[1:] for r in ratios]
    order = [o for o in order if o in set(res["label"])]
    sizes = sorted(res["size"].unique())
    colors = {s: C.SERIES[i % len(C.SERIES)] for i, s in enumerate(sizes)}
    width = 0.8 / len(sizes)
    x = np.arange(len(order))

    # 1) Precision / Recall / F1 / AUPRC
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor=C.SURFACE)
    for ax, (metric, title) in zip(axes.flat, [("precision", "Precision"), ("recall", "Recall"),
                                               ("f1", "F1"), ("auprc", "AUPRC")]):
        C.style_ax(ax, title)
        for i, s in enumerate(sizes):
            sub = res[res["size"] == s].set_index("label").reindex(order)
            ax.bar(x + (i - (len(sizes) - 1) / 2) * width, sub[metric].fillna(0), width,
                   color=colors[s], edgecolor=C.SURFACE, linewidth=1.5, label=C.size_label(s))
        ax.set_xticks(x, order, fontsize=8.5)
        ax.set_ylim(0, 1.05)
    # Lejant başlık satırında, sağda: çubukların üstüne binmesin
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Veri boyutu", frameon=False, fontsize=9, title_fontsize=9,
               labelcolor=C.INK2, loc="upper right", ncol=len(sizes), bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Dengeleme yöntemine ve veri boyutuna göre test metrikleri "
                 "(test seti gerçek dağılımda, eşik 0.5)", x=0.01, ha="left", color=C.INK, fontsize=13)
    C.save_fig(fig, out / "metrics_by_method.png")

    # 2) Eğitim süresi (log x)
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C.SURFACE)
    C.style_ax(ax, "Yönteme göre Logistic Regression eğitim süresi", xgrid=True, ygrid=False)
    y = np.arange(len(order))
    h = 0.8 / len(sizes)
    for i, s in enumerate(sizes):
        sub = res[res["size"] == s].set_index("label").reindex(order)
        ax.barh(y + (i - (len(sizes) - 1) / 2) * h, sub["train_sec"], h, color=colors[s],
                edgecolor=C.SURFACE, linewidth=1.5, label=C.size_label(s))
    ax.set_xscale("log")
    ax.set_yticks(y, [o.replace("\n", " ") for o in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Eğitim süresi (sn, log ölçek)", color=C.INK2)
    ax.legend(title="Veri boyutu", frameon=False, fontsize=9, title_fontsize=9, labelcolor=C.INK2)
    C.save_fig(fig, out / "training_time.png")

    # 3) Baseline: accuracy vs recall
    base = res[res["method"] == "none"].sort_values("size")
    if not base.empty:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=C.SURFACE)
        C.style_ax(ax, "Baseline (dengeleme yok): accuracy yüksek, recall düşük")
        xb = np.arange(len(base))
        for j, (metric, name, col) in enumerate([("accuracy", "Accuracy", C.SERIES[0]),
                                                 ("recall", "Recall (fraud yakalama)", C.SERIES[1])]):
            bars = ax.bar(xb + (j - 0.5) * 0.38, base[metric], 0.38, color=col,
                          edgecolor=C.SURFACE, linewidth=1.5, label=name)
            for b, v in zip(bars, base[metric]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=9, color=C.INK)
        ax.set_xticks(xb, [C.size_label(s) for s in base["size"]])
        ax.set_ylim(0, 1.3)  # üstte lejant için boşluk: değer etiketleriyle çakışmasın
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.legend(frameon=False, fontsize=9, labelcolor=C.INK2, loc="upper center", ncol=2)
        C.save_fig(fig, out / "baseline_accuracy_vs_recall.png")

    # 4) Veri büyüse de fraud ORANI değişmiyor, sadece fraud SAYISI artıyor
    dpath = out / "class_distribution.csv"
    if dpath.exists():
        d = pd.read_csv(dpath)
        d = d[d["split"] == "all"].sort_values("size")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=C.SURFACE)
        C.style_ax(ax, "Veri büyüdükçe fraud sayısı artıyor, oranı sabit kalıyor")
        xs = np.arange(len(d))
        bars = ax.bar(xs, d["fraud"], 0.55, color=C.SERIES[1], edgecolor=C.SURFACE, label="Fraud sayısı")
        for b, v in zip(bars, d["fraud"]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs, [C.size_label(s) for s in d["size"]])
        ax.set_ylabel("Fraud işlem sayısı", color=C.INK2)
        ax2 = ax.twinx()
        ax2.plot(xs, d["fraud_pct"], color=C.SERIES[0], marker="o", lw=2, label="Fraud oranı (%)")
        for xi, v in zip(xs, d["fraud_pct"]):
            ax2.text(xi, v, f" %{v:.3f}", color=C.SERIES[0], fontsize=9, va="bottom")
        ax2.set_ylim(0, max(1.0, d["fraud_pct"].max() * 2))
        ax2.set_ylabel("Fraud oranı (%)", color=C.SERIES[0])
        ax2.spines["top"].set_visible(False)
        C.save_fig(fig, out / "class_distribution_by_size.png")
    log.info(f"[f] Grafikler -> {out}")


def plot_input_comparison(res_path: Path, out: Path):
    """Temel özellikler (input=scaled) ile Öğrenci 3'ün özellikleri (input=features) karşılaştırması.
    İki girdinin de sonuçları CSV'de varsa: boyut başına en iyi F1 (dengeleme yöntemleri arasında)
    ve dengeleme yokken recall."""
    import numpy as np
    res = read_results(res_path)
    if res.empty or not {"scaled", "features"} <= set(res["input"]):
        return
    plt = C.plt_setup()
    names = {"scaled": "Temel özellikler (7)", "features": "Öğrenci 3'ün özellikleriyle (14)"}
    sizes = sorted(res["size"].unique())
    x = np.arange(len(sizes))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=C.SURFACE)
    panels = [("En iyi F1 (dengeleme yöntemleri arasında)", res[res["method"] != "none"], "f1", "max"),
              ("Dengeleme yokken recall", res[res["method"] == "none"], "recall", "first")]
    for ax, (title, sub, metric, agg) in zip(axes, panels):
        C.style_ax(ax, title)
        for j, kind in enumerate(["scaled", "features"]):
            vals = (sub[sub["input"] == kind].groupby("size")[metric].agg(agg)
                    .reindex(sizes).fillna(0).to_numpy())
            bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=C.SERIES[j], edgecolor=C.SURFACE,
                          linewidth=1.5, label=names[kind])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=9, color=C.INK)
        ax.set_xticks(x, [C.size_label(s) for s in sizes])
        ax.set_ylim(0, 0.75)  # üstte lejant için boşluk
        ax.set_yticks(np.arange(0, 0.61, 0.1))
    axes[0].legend(frameon=False, fontsize=9, labelcolor=C.INK2, loc="upper left", ncol=2)
    C.save_fig(fig, out / "input_comparison.png")


# --------------------------------------------------------------------------- #
def main():
    global log
    ap = argparse.ArgumentParser(description="Öğrenci 2 - class imbalance analizi")
    ap.add_argument("--sizes", type=int, nargs="+", default=C.DEFAULT_SIZES)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--target_ratios", type=int, nargs="+", default=[1, 10],
                    help="normal/fraud hedef oranı: 1 -> 1:1, 10 -> 1:10")
    ap.add_argument("--input", default="auto", choices=["auto", "features", "scaled"],
                    help="auto: Öğrenci 3 çıktısı hazırsa features, değilse scaled (data/scaled + temel özellikler)")
    ap.add_argument("--driver_memory", default="8g")
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "ogrenci2"))
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--select_metric", default="f1", choices=["f1", "auprc", "recall", "precision"])
    ap.add_argument("--no_save", action="store_true", help="data/balanced/ altına yazma")
    ap.add_argument("--plot_only", action="store_true",
                    help="Spark başlatmadan, mevcut imbalance_results.csv'den sadece grafikleri üret")
    ap.add_argument("--force", action="store_true",
                    help="CSV'de olan deneyleri de yeniden çalıştır (resume'u kapat)")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = C.PROJECT_ROOT / out
    log = C.setup_logging(out, "ogrenci2")
    res_path = out / "imbalance_results.csv"

    if args.plot_only:
        results = read_results(res_path, args.input)
        if results.empty:
            log.error(f"{res_path} boş ya da yok; önce deneyleri çalıştırın.")
            sys.exit(1)
        log.info(f"--plot_only: {len(results)} deney sonucu okundu (girdi: {results['input'].iloc[0]})")
        make_plots(results, out)
        plot_input_comparison(res_path, out)
        return

    sizes = sorted(args.sizes)
    input_kind = resolve_input(args.input, sizes)
    log.info(f"Öğrenci 2 başladı | sizes={sizes} methods={args.methods} "
             f"ratios={args.target_ratios} input={input_kind} pid={os.getpid()}")
    if input_kind == "scaled":
        log.info("Girdi: data/scaled (Öğrenci 1) + temel özellikler. Öğrenci 3 data/features/ çıktısını "
                 "teslim edince aynı komut özellikli veriyle yeniden çalışır.")
    spark = C.get_spark("ogrenci2-class-imbalance", args.driver_memory)
    holder: dict = {}
    try:
        for size in sizes:
            log.info(f"===== {size:,} kayıt =====")
            with C.Timer() as t:
                n = run_size(spark, size, args.methods, args.target_ratios, input_kind, holder,
                             out, args.max_iter, args.force)
            log.info(f"===== {size:,} bitti: {n} deney çalıştırıldı, {t.sec / 60:.1f} dk =====")

        results = read_results(res_path, input_kind)
        C.safe_plot(make_plots, results, out, log=log.warning)
        C.safe_plot(plot_input_comparison, res_path, out, log=log.warning)
        if not args.no_save and set(args.methods) - {"none"}:
            save_balanced(spark, results, args.select_metric, sizes, input_kind, holder, out, args.force)
        log.info("Öğrenci 2 tamamlandı.")
    except BaseException as e:   # KeyboardInterrupt dahil: nerede kesildiği log'da kalsın
        log.exception(f"Çalıştırma kesildi: {type(e).__name__}: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
