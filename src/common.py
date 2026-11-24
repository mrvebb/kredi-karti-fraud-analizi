"""
Ortak yardımcı modül - tüm öğrencilerin scriptleri bu modülü kullanır.

İçerik
-------
- get_spark()              : SparkSession oluşturma (driver memory parametreli)
- read_raw()               : fraudTrain.csv + fraudTest.csv birleştirip okuma
- add_basic_features()     : yaş, saat, haftanın günü, haversine mesafe, cinsiyet
- build_feature_stages()   : category -> StringIndexer + OneHotEncoder, VectorAssembler, StandardScaler
- scale_dataset()          : hedef boyuta ölçekleme (fraud oranı korunur)
- group_split()            : train/test ayrımı (aynı orijinal satırın kopyaları aynı tarafta kalır)
- write_scaled() / load_scaled() / load_or_build()
- dir_size_bytes(), Timer

Komut satırı (ölçeklenmiş veri setlerini üretir):
    python src/common.py --sizes 100000 1000000 5000000 --driver_memory 6g
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Windows'ta Spark'ın doğru Python'u kullanması için (aksi halde "Python worker failed to connect")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def _windows_env():
    """Windows: JAVA_HOME / HADOOP_HOME tanımlı değilse yaygın kurulum yerlerinde ara.
    hadoop.dll'in JVM tarafından bulunması için HADOOP_HOME\\bin PATH'e eklenir
    (aksi halde Parquet/CSV yazarken UnsatisfiedLinkError: NativeIO$Windows.access0)."""
    if os.name != "nt":
        return
    if not os.environ.get("JAVA_HOME"):
        for root in (r"C:\Program Files\Eclipse Adoptium", r"C:\Program Files\Java",
                     r"C:\Program Files\Microsoft"):
            cands = sorted(Path(root).glob("jdk-17*")) + sorted(Path(root).glob("jdk-11*")) \
                if Path(root).exists() else []
            if cands:
                os.environ["JAVA_HOME"] = str(cands[-1])
                break
    if not os.environ.get("HADOOP_HOME") and Path(r"C:\hadoop\bin\winutils.exe").exists():
        os.environ["HADOOP_HOME"] = r"C:\hadoop"
    for var in ("JAVA_HOME", "HADOOP_HOME"):
        if os.environ.get(var):
            b = str(Path(os.environ[var]) / "bin")
            if b.lower() not in os.environ.get("PATH", "").lower():
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")


_windows_env()

from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402

# --------------------------------------------------------------------------- #
# Sabitler / yollar
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Colab'da veri başka yerdeyse: os.environ["FRAUD_DATA_DIR"] = "/content/data"
DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", PROJECT_ROOT / "data"))
RAW_FILES = {"train": "fraudTrain.csv", "test": "fraudTest.csv"}
SCALED_DIR = DATA_DIR / "scaled"        # Öğrenci 1 çıktısı
FEATURES_DIR = DATA_DIR / "features"    # Öğrenci 3 çıktısı
BALANCED_DIR = DATA_DIR / "balanced"    # Öğrenci 2 çıktısı
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_SIZES = [100_000, 1_000_000, 5_000_000]
LABEL_COL = "is_fraud"
ID_COL = "orig_id"          # orijinal satır kimliği (ölçeklemede kopyalar aynı id'yi taşır)
SEED = 42

# Kişisel bilgiler: hiçbir zaman modele girmez, okuma sırasında atılır
PII_COLS = ["first", "last", "street", "trans_num"]

# fraudTest satırlarının orig_id'si fraudTrain ile çakışmasın diye eklenen ofset
TEST_ID_OFFSET = 10_000_000

TS_FORMAT = "yyyy-MM-dd HH:mm:ss"

RAW_SCHEMA = T.StructType([
    T.StructField("idx", T.LongType()),               # adsız index sütunu
    T.StructField("trans_date_trans_time", T.StringType()),
    T.StructField("cc_num", T.LongType()),
    T.StructField("merchant", T.StringType()),
    T.StructField("category", T.StringType()),
    T.StructField("amt", T.DoubleType()),
    T.StructField("first", T.StringType()),
    T.StructField("last", T.StringType()),
    T.StructField("gender", T.StringType()),
    T.StructField("street", T.StringType()),
    T.StructField("city", T.StringType()),
    T.StructField("state", T.StringType()),
    T.StructField("zip", T.IntegerType()),
    T.StructField("lat", T.DoubleType()),
    T.StructField("long", T.DoubleType()),
    T.StructField("city_pop", T.LongType()),
    T.StructField("job", T.StringType()),
    T.StructField("dob", T.StringType()),
    T.StructField("trans_num", T.StringType()),
    T.StructField("unix_time", T.LongType()),
    T.StructField("merch_lat", T.DoubleType()),
    T.StructField("merch_long", T.DoubleType()),
    T.StructField("is_fraud", T.IntegerType()),
])

# read_raw() + scale_dataset() sonrası şema (ölçeklenmiş CSV'yi okurken kullanılır)
SCALED_SCHEMA = T.StructType([
    T.StructField(ID_COL, T.LongType()),
    T.StructField("trans_date_trans_time", T.TimestampType()),
    T.StructField("cc_num", T.LongType()),
    T.StructField("merchant", T.StringType()),
    T.StructField("category", T.StringType()),
    T.StructField("amt", T.DoubleType()),
    T.StructField("gender", T.StringType()),
    T.StructField("city", T.StringType()),
    T.StructField("state", T.StringType()),
    T.StructField("zip", T.IntegerType()),
    T.StructField("lat", T.DoubleType()),
    T.StructField("long", T.DoubleType()),
    T.StructField("city_pop", T.LongType()),
    T.StructField("job", T.StringType()),
    T.StructField("dob", T.DateType()),
    T.StructField("unix_time", T.LongType()),
    T.StructField("merch_lat", T.DoubleType()),
    T.StructField("merch_long", T.DoubleType()),
    T.StructField("is_fraud", T.IntegerType()),
    T.StructField("is_synthetic", T.IntegerType()),
])

# Modele giren temel sayısal özellikler (+ category one-hot)
NUMERIC_FEATURES = ["amt", "city_pop", "age", "hour", "day_of_week", "distance_km", "gender_m"]

# Öğrenci 3'ün ürettiği müşteri (cc_num) / zaman bazlı davranış özellikleri
BEHAVIOR_FEATURES = ["log_amt", "is_night", "txn_count_24h", "card_avg_amt", "amt_dev_from_avg",
                     "amt_ratio_to_avg", "log_secs_since_prev"]
MODEL_FEATURES = NUMERIC_FEATURES + BEHAVIOR_FEATURES


# --------------------------------------------------------------------------- #
# Spark
# --------------------------------------------------------------------------- #
def get_spark(app_name: str = "fraud-analizi", driver_memory: str = "4g",
              shuffle_partitions: int | None = None) -> SparkSession:
    """Yerel (local[*]) SparkSession. driver_memory JVM başlamadan önce verilmeli."""
    cores = os.cpu_count() or 2
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions or max(8, cores * 2)))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# --------------------------------------------------------------------------- #
# Okuma
# --------------------------------------------------------------------------- #
def read_raw(spark: SparkSession, data_dir: Path | str = DATA_DIR) -> DataFrame:
    """fraudTrain + fraudTest'i birleştirir; index sütununu atar, PII sütunlarını çıkarır.

    orig_id: her orijinal satır için benzersiz kimlik (train: idx, test: idx + 10M).
    """
    data_dir = Path(data_dir)
    parts = []
    for name, fname in RAW_FILES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} bulunamadı. Kaggle verisini data/ altına koyun (README).")
        df = (spark.read.option("header", True).option("quote", '"').option("escape", '"')
              .schema(RAW_SCHEMA).csv(str(path)))
        offset = 0 if name == "train" else TEST_ID_OFFSET
        parts.append(df.withColumn(ID_COL, F.col("idx") + F.lit(offset)))
    df = parts[0].unionByName(parts[1])
    df = (df.drop("idx", *PII_COLS)
            .withColumn("trans_date_trans_time", F.to_timestamp("trans_date_trans_time", TS_FORMAT))
            .withColumn("dob", F.to_date("dob", "yyyy-MM-dd"))
            .withColumn("is_synthetic", F.lit(0)))
    return df.select([f.name for f in SCALED_SCHEMA.fields])


# --------------------------------------------------------------------------- #
# Özellik çıkarımı
# --------------------------------------------------------------------------- #
def add_basic_features(df: DataFrame) -> DataFrame:
    """age, hour, day_of_week, distance_km (haversine), gender_m sütunlarını ekler."""
    ts = F.col("trans_date_trans_time")
    lat1, lon1 = F.radians("lat"), F.radians("long")
    lat2, lon2 = F.radians("merch_lat"), F.radians("merch_long")
    a = (F.sin((lat2 - lat1) / 2) ** 2
         + F.cos(lat1) * F.cos(lat2) * F.sin((lon2 - lon1) / 2) ** 2)
    return (df
            .withColumn("age", F.floor(F.months_between(ts, F.col("dob")) / 12).cast("double"))
            .withColumn("hour", F.hour(ts).cast("double"))
            .withColumn("day_of_week", F.dayofweek(ts).cast("double"))   # 1=Pazar ... 7=Cumartesi
            .withColumn("distance_km", 2 * 6371.0 * F.asin(F.sqrt(a)))
            .withColumn("gender_m", (F.col("gender") == "M").cast("double")))


def build_feature_stages(numeric_cols: list[str] | None = None, scale: bool = True):
    """category -> StringIndexer + OneHotEncoder; sayısal + one-hot -> VectorAssembler -> StandardScaler.

    Dönen stage listesi bir Pipeline'a konur ve SADECE train setine fit edilir.
    Çıktı sütunu: "features".
    """
    from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler

    numeric_cols = numeric_cols or NUMERIC_FEATURES
    stages = [
        StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep"),
        OneHotEncoder(inputCols=["category_idx"], outputCols=["category_ohe"]),
        VectorAssembler(inputCols=numeric_cols + ["category_ohe"],
                        outputCol="raw_features" if scale else "features"),
    ]
    if scale:
        stages.append(StandardScaler(inputCol="raw_features", outputCol="features",
                                     withMean=True, withStd=True))
    return stages


FEATURE_TMP_COLS = ["category_idx", "category_ohe", "raw_features", "features"]


# --------------------------------------------------------------------------- #
# Ölçekleme
# --------------------------------------------------------------------------- #
def class_counts(df: DataFrame, label: str = LABEL_COL) -> tuple[int, int]:
    """(normal, fraud) sayıları."""
    rows = {r[label]: r["count"] for r in df.groupBy(label).count().collect()}
    return int(rows.get(0, 0)), int(rows.get(1, 0))


def _add_noise(df: DataFrame, seed: int) -> DataFrame:
    """Sentetik satırlara küçük gürültü: birebir kopya olmasın.

    amt ve city_pop ±%1 civarı, koordinatlar ~±100 m, zaman ±5 dk kayar.
    Etiket, kategori, müşteri (cc_num) değişmez.
    """
    shift = F.round(F.randn(seed + 5) * 300).cast("long")   # saniye
    return (df
            .withColumn("amt", F.round(F.greatest(F.lit(1.0),
                                                  F.col("amt") * (1 + F.randn(seed) * 0.01)), 2))
            .withColumn("city_pop", F.greatest(F.lit(1),
                                               F.round(F.col("city_pop") * (1 + F.randn(seed + 1) * 0.01))
                                               .cast("long")))
            .withColumn("lat", F.col("lat") + F.randn(seed + 2) * 0.001)
            .withColumn("long", F.col("long") + F.randn(seed + 3) * 0.001)
            .withColumn("merch_lat", F.col("merch_lat") + F.randn(seed + 4) * 0.001)
            .withColumn("merch_long", F.col("merch_long") + F.randn(seed + 6) * 0.001)
            .withColumn("_shift", shift)
            .withColumn("unix_time", F.col("unix_time") + F.col("_shift"))
            .withColumn("trans_date_trans_time",
                        (F.col("trans_date_trans_time").cast("long") + F.col("_shift")).cast("timestamp"))
            .drop("_shift"))


def scale_dataset(df: DataFrame, target_size: int, seed: int = SEED,
                  label: str = LABEL_COL) -> DataFrame:
    """Veriyi target_size satıra getirir; fraud oranı korunur.

    - target_size <= mevcut: her sınıftan oranı koruyarak iadesiz örnekleme.
    - target_size  > mevcut: orijinal satırların hepsi + eksik kadar iadeli örneklenmiş,
      gürültü eklenmiş sentetik satır (is_synthetic=1). Kopyalar orig_id'yi korur.

    df önceden cache'lenmiş olmalı (birden fazla kez taranır).
    """
    n_normal, n_fraud = class_counts(df, label)
    n_total = n_normal + n_fraud
    fraud_ratio = n_fraud / n_total
    t_fraud = int(round(target_size * fraud_ratio))
    t_normal = target_size - t_fraud

    out = []
    for cls, have, want in ((0, n_normal, t_normal), (1, n_fraud, t_fraud)):
        part = df.filter(F.col(label) == cls)
        if want <= have:
            frac = min(1.0, want / have * 1.03 + 1e-4)   # biraz fazla örnekle, sonra kes
            out.append(part.sample(False, frac, seed + cls).limit(want))
        else:
            extra = want - have
            frac = extra / have * 1.03 + 1e-4
            synth = (part.sample(True, frac, seed + 10 + cls).limit(extra)
                     .withColumn("is_synthetic", F.lit(1)))
            out.append(part.unionByName(_add_noise(synth, seed + 20 + cls)))
    return out[0].unionByName(out[1])


def group_split(df: DataFrame, train_frac: float = 0.8, seed: int = SEED,
                id_col: str = ID_COL) -> tuple[DataFrame, DataFrame]:
    """Train/test ayrımı. Atama orig_id'nin hash'ine göre yapılır:
    aynı orijinal satırın gürültülü kopyaları hep aynı tarafa düşer (sızıntı olmaz).
    Hash etiketten bağımsız olduğu için her sınıf ~%80/%20 bölünür (sınıf oranı korunur).
    """
    bucket = F.pmod(F.xxhash64(F.col(id_col), F.lit(seed)), F.lit(10_000)) / 10_000.0
    flagged = df.withColumn("_is_train", bucket < train_frac)
    train = flagged.filter("_is_train").drop("_is_train")
    test = flagged.filter(~F.col("_is_train")).drop("_is_train")
    return train, test


# --------------------------------------------------------------------------- #
# Yazma / okuma
# --------------------------------------------------------------------------- #
def scaled_path(size: int, fmt: str, base: Path | str = SCALED_DIR) -> Path:
    return Path(base) / f"fraud_{size}.{fmt}"


def write_scaled(df: DataFrame, size: int, fmt: str, base: Path | str = SCALED_DIR) -> Path:
    path = scaled_path(size, fmt, base)
    n_files = max(1, math.ceil(size / 1_000_000))
    writer = df.repartition(n_files).write.mode("overwrite")
    if fmt == "csv":
        writer.option("header", True).option("timestampFormat", TS_FORMAT).csv(str(path))
    elif fmt == "parquet":
        writer.parquet(str(path))
    else:
        raise ValueError(fmt)
    return path


def load_scaled(spark: SparkSession, size: int, fmt: str = "parquet",
                base: Path | str = SCALED_DIR) -> DataFrame:
    path = scaled_path(size, fmt, base)
    if fmt == "csv":
        return (spark.read.option("header", True).option("timestampFormat", TS_FORMAT)
                .schema(SCALED_SCHEMA).csv(str(path)))
    return spark.read.parquet(str(path))


def is_complete(path: Path | str) -> bool:
    """Spark çıktı klasörü eksiksiz mi? Spark _SUCCESS dosyasını yazmanın EN SONUNDA oluşturur;
    yoksa yazma yarıda kesilmiştir (klasördeki part dosyaları eksik/bozuk olabilir)."""
    return (Path(path) / "_SUCCESS").is_file()


def remove_incomplete(path: Path | str) -> bool:
    """Yarım kalmış Spark çıktı klasörünü siler. Silindiyse True."""
    import shutil
    path = Path(path)
    if path.exists() and not is_complete(path):
        shutil.rmtree(path)
        return True
    return False


def load_or_build(spark: SparkSession, size: int, fmt: str = "parquet",
                  raw: DataFrame | None = None, persist: bool = False, log=print) -> DataFrame:
    """data/scaled/ altında EKSİKSİZ (_SUCCESS'li) kopya varsa okur.

    Yoksa ham veriden üretir. persist=True ise üretilen veri data/scaled/ altına yazılıp
    oradan geri okunur: sonraki çalıştırmalar yeniden üretmez ve (örnekleme rastgele olduğu
    için önemli) kesintiden sonra kaldığı yerden devam eden deneyler AYNI veriyi görür.
    persist=False ise eski davranış: bellekte üretir, diske yazmaz.
    """
    path = scaled_path(size, fmt)
    if is_complete(path):
        log(f"[common] {path} hazır -> okunuyor")
        return load_scaled(spark, size, fmt)
    if remove_incomplete(path):
        log(f"[common] {path} yarım kalmış (_SUCCESS yok) -> silindi")
    if raw is None:
        raw = read_raw(spark).cache()
    if not persist:
        log(f"[common] {path} yok -> ham veriden {size:,} satır bellekte üretiliyor")
        return scale_dataset(raw, size)
    log(f"[common] {path} yok -> {size:,} satır üretilip diske yazılıyor")
    with Timer() as t:
        write_scaled(scale_dataset(raw, size), size, fmt)
    log(f"[common] {path} yazıldı ({t.sec:.1f} sn)")
    return load_scaled(spark, size, fmt)


def features_path(size: int, base: Path | str = FEATURES_DIR) -> Path:
    return Path(base) / f"fraud_{size}.parquet"


def load_features(spark: SparkSession, size: int, log=print) -> DataFrame:
    """Öğrenci 3'ün özellikli verisi (eksiksizse). Yoksa anlaşılır hata verir."""
    path = features_path(size)
    if not is_complete(path):
        raise FileNotFoundError(f"{path} yok/yarım. Öğrenci 3'ün özellik çıktısı bekleniyor "
                                f"(README > Arayüz) ya da Öğrenci 2'yi --input scaled ile çalıştırın.")
    log(f"[common] {path} okunuyor")
    return spark.read.parquet(str(path))


def size_label(n: int) -> str:
    """100000 -> 100K, 1000000 -> 1M"""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    return f"{n // 1000}K"


# --------------------------------------------------------------------------- #
# Log + kesintiye dayanıklı sonuç CSV'leri (tüm öğrenci scriptleri kullanır)
# --------------------------------------------------------------------------- #
def setup_logging(out: Path, name: str):
    """Zaman damgalı log: konsol + <out>/run.log (append)."""
    import logging
    for stream in (sys.stdout, sys.stderr):   # Windows'ta dosyaya yönlendirirken Türkçe karakterler
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%Y-%m-%d %H:%M:%S")
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(out / "run.log", mode="a", encoding="utf-8")):
        h.setFormatter(fmt)
        lg.addHandler(h)
    lg.propagate = False
    return lg


def append_row(row: dict, path: Path, cols: list[str]):
    """Tek satırı hemen diske ekler (flush + fsync). Dosya yoksa başlık yazılır.
    Önceki kesinti satırı yarım bıraktıysa (sonda \\n yok) önce satır sonu eklenir."""
    import csv
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    prefix = ""
    if not new:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                prefix = "\n"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    if new:
        w.writeheader()
    w.writerow(row)
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(prefix + buf.getvalue())
        f.flush()
        os.fsync(f.fileno())


def read_rows(path: Path, keys: list[str], required: list[str], cols: list[str] | None = None):
    """append_row ile yazılmış CSV'yi okur: yarım satırlar (required boş) atılır,
    aynı anahtar birden çok varsa sonuncusu tutulur. Anahtarlar string olarak döner."""
    import pandas as pd
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols or keys + required)
    df = pd.read_csv(path, on_bad_lines="skip", dtype={k: str for k in keys})
    df = df.dropna(subset=required)
    return df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def done_set(path: Path, keys: list[str], required: list[str]) -> set[tuple]:
    df = read_rows(path, keys, required)
    return {tuple(str(v) for v in r) for r in df[keys].itertuples(index=False)}


# --------------------------------------------------------------------------- #
# Ortak grafik stili (tüm öğrencilerin grafikleri aynı görünsün)
# --------------------------------------------------------------------------- #
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def style_ax(ax, title: str | None = None, xgrid: bool = False, ygrid: bool = True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def dir_size_bytes(path: Path | str) -> int:
    """Spark çıktı klasörünün diskteki boyutu (.crc ve _SUCCESS hariç)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*")
               if p.is_file() and not p.name.endswith(".crc") and not p.name.startswith("_"))


class Timer:
    """with Timer() as t: ...  ->  t.sec"""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.sec = time.perf_counter() - self._t0
        return False


def merge_csv(new_rows: list[dict], path: Path, keys: list[str]):
    """Sonuçları mevcut CSV ile birleştirir (aynı anahtarlı eski satırlar güncellenir).
    Böylece 100K/1M ve 5M ayrı çalıştırılsa da tek tabloda toplanır."""
    import pandas as pd
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        merged_keys = new[keys].astype(str).agg("|".join, axis=1)
        old_keys = old[keys].astype(str).agg("|".join, axis=1)
        new = pd.concat([old[~old_keys.isin(set(merged_keys))], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


# --------------------------------------------------------------------------- #
# CLI: ölçeklenmiş veri setlerini üret (CSV + Parquet)
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="100K/1M/5M veri setlerini CSV ve Parquet olarak üretir.")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--formats", nargs="+", default=["csv", "parquet"])
    ap.add_argument("--driver_memory", default="4g")
    args = ap.parse_args()

    spark = get_spark("common-scale", args.driver_memory)
    raw = read_raw(spark).cache()
    n0, n1 = class_counts(raw)
    print(f"Ham veri: {n0 + n1:,} satır, fraud={n1:,} (%{100 * n1 / (n0 + n1):.3f})")

    rows = []
    for size in args.sizes:
        df = scale_dataset(raw, size).cache()
        c0, c1 = class_counts(df)
        print(f"\n== {size:,}: normal={c0:,} fraud={c1:,} (%{100 * c1 / (c0 + c1):.3f})")
        for fmt in args.formats:
            with Timer() as t:
                path = write_scaled(df, size, fmt)
            mb = dir_size_bytes(path) / 1e6
            print(f"   {fmt:8s} yazıldı: {t.sec:6.1f} sn, {mb:8.1f} MB -> {path}")
            rows.append({"size": size, "format": fmt, "rows": c0 + c1, "fraud": c1,
                         "write_sec": round(t.sec, 2), "disk_mb": round(mb, 2)})
        df.unpersist()

    out = merge_csv(rows, RESULTS_DIR / "scaled_datasets.csv", ["size", "format"])
    print("\n", out.to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
