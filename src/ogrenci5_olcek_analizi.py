"""Öğrenci 5: ölçeklenebilirlik analizi (LR ve RF).

Girdiler salt okunur. Sonuçlar varsayılan olarak results/ogrenci5/ altında.
Colab için --out ile kalıcı Drive klasörü seçilebilir.
Örnek pilot: python src/ogrenci5_olcek_analizi.py --sizes 100000 --experiments scaling --models lr rf
Her kombinasyon: 1 ısınma, 3 ölçüm; CSV ham ölçümleri, grafikler medyanları içerir.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

import common as C
from ogrenci2_class_imbalance import balance, evaluate

LABEL = C.LABEL_COL
INPUT_COLS = [*C.MODEL_FEATURES, 'category', LABEL, C.ID_COL]
RATIOS = ['original', '1:50', '1:10', '1:1']
REPEATS = 3
KEYS = ['size', 'model', 'target_ratio', 'repeat']
MEASUREMENTS = ['read_sec', 'prepare_sec', 'balance_sec', 'train_sec', 'predict_sec', 'eval_sec',
                'throughput', 'rss_start_mb', 'rss_peak_mb', 'rss_increase_mb',
                'f1', 'auprc', 'precision', 'recall', 'accuracy']
COLS = KEYS + ['input', 'method', 'source_rows', 'train_rows', 'train_normal', 'train_fraud',
               'test_rows', 'test_fraud', 'actual_normal_per_fraud', 'threshold',
               'TP', 'FP', 'FN', 'TN', *MEASUREMENTS, 'cpu_count', 'cpu_model',
               'partitions', 'shuffle_partitions', 'driver_memory', 'spark_version',
               'python_version', 'ram_mb', 'runtime_id', 'memory_method', 'finished_at']
REQUIRED = [k for k in COLS if k not in KEYS]
# CSV'de her ölçümün kimliği: deney dosyası + boyut + model + oran + tekrar.
FILES = {'scaling': 'scaling_results.csv', 'imbalance': 'imbalance_scaling.csv'}


class MemoryMonitor:
    """Spark JVM RSS'ini 0.2 saniye arayla örnekler (Python RAM'i dahil değil)."""

    def __init__(self, spark):
        # Java yönetim arayüzündeki PID ile doğrudan Spark JVM'ini bul.
        pid = int(spark._jvm.java.lang.management.ManagementFactory.getRuntimeMXBean().getName().split('@')[0])
        self.process = psutil.Process(pid)
        self.stop_event = threading.Event()
        self.start_rss = self.peak = self.rss()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def rss(self):
        return self.process.memory_info().rss

    def poll(self):
        while not self.stop_event.wait(0.2):
            self.peak = max(self.peak, self.rss())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()
        if exc[0] is None:
            self.peak = max(self.peak, self.rss())

    def values(self):
        # MB: 10^6 bayt; başlangıçtan kalan bellek de ayrı gösterilir.
        return {'rss_start_mb': self.start_rss / 1e6, 'rss_peak_mb': self.peak / 1e6,
                'rss_increase_mb': (self.peak - self.start_rss) / 1e6}


def cpu_name():
    """Linux/Colab'da CPU modeli, diğer sistemlerde platform bilgisi."""
    path = Path('/proc/cpuinfo')
    if path.exists():
        for line in path.read_text(errors='replace').splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor() or platform.machine() or 'unknown'


def environment(spark):
    return {'cpu_count': os.cpu_count() or 1, 'cpu_model': cpu_name(),
            'partitions': spark.sparkContext.defaultParallelism,
            'shuffle_partitions': int(spark.conf.get('spark.sql.shuffle.partitions')),
            'driver_memory': spark.conf.get('spark.driver.memory'), 'spark_version': spark.version,
            'python_version': platform.python_version(), 'ram_mb': psutil.virtual_memory().total / 1e6,
            'runtime_id': socket.gethostname(), 'memory_method': 'JVM_RSS_psutil_0.2s'}


def classifier(name):
    """Öğrenci 4'ün varsayılan parametreleri; bütün boyutlarda aynı."""
    if name == 'lr':
        return LogisticRegression(labelCol=LABEL, featuresCol='features', maxIter=100,
                                  regParam=0.0, threshold=0.5)
    if name == 'rf':
        # İki sınıfta eşit ağırlıklı varsayılan karar: yaklaşık 0.5 eşiği.
        return RandomForestClassifier(labelCol=LABEL, featuresCol='features', numTrees=100,
                                      maxDepth=10, seed=C.SEED, featureSubsetStrategy='sqrt')
    raise ValueError(name)


def verify_metadata():
    path = C.BALANCED_DIR / 'SECILEN_YONTEM.json'
    info = json.loads(path.read_text(encoding='utf-8'))
    if (info.get('input'), info.get('method'), info.get('target_ratio'), info.get('weight_col')) != ('features', 'undersample', '1:10', None):
        raise ValueError('balanced verisi beklenen features / undersample / 1:10 sürümü değil.')
    if info.get('feature_cols') != C.MODEL_FEATURES:
        raise ValueError('JSON özellik listesi common.MODEL_FEATURES ile uyuşmuyor.')
    return info


def run_measurement(spark, experiment, size, name, ratio, repeat, env, metadata):
    """Bir tam ölçüm: gerçek okuma, hazırlama, eğitim, tahmin ve metrikler."""
    cached = []

    def keep(df):
        # Lazy evaluation: count ile önbellek gerçekten doldurulur.
        df = df.cache()
        cached.append(df)
        df.count()
        return df

    def read(path):
        if not C.is_complete(path):
            raise FileNotFoundError(f'{path}: _SUCCESS yok; veri kullanılmadı.')
        # Yalnızca deneyde gereken sütunlar okunur; diğer sütunlar değiştirilmez.
        return keep(spark.read.parquet(str(path)).select(*INPUT_COLS))

    def prepared(df):
        return keep(df.repartition(env['partitions']))

    try:
        with MemoryMonitor(spark) as memory:
            t = time.perf_counter()
            if experiment == 'scaling':
                raw_train = read(C.BALANCED_DIR / f'train_n{size}.parquet')
                raw_test = read(C.BALANCED_DIR / f'test_n{size}.parquet')
            else:
                raw = read(C.features_path(size))
            read_sec = time.perf_counter() - t

            t = time.perf_counter()
            balance_sec = 0.0
            if experiment == 'scaling':
                train, test = prepared(raw_train), prepared(raw_test)
                raw_train.unpersist(blocking=True)
                raw_test.unpersist(blocking=True)
            else:
                # orig_id hash bölmesi: kopyalar orijinalleriyle aynı tarafta.
                base_train, base_test = C.group_split(raw)
                n0, n1 = C.class_counts(base_train)
                if n0 == 0 or n1 == 0:
                    raise ValueError('Eğitimde iki sınıf da bulunmalı.')
                bt = time.perf_counter()
                if ratio == 'original':
                    train = prepared(base_train)
                else:
                    # Öğrenci 2'nin örnekleme fonksiyonu; hedef oran yaklaşık gerçekleşir.
                    sampled, _ = balance(base_train, 'undersample', int(ratio.split(':')[1]), n0, n1, C.SEED)
                    train = prepared(sampled)
                    balance_sec = time.perf_counter() - bt
                test = prepared(base_test)
                raw.unpersist(blocking=True)
            tr0, tr1 = C.class_counts(train)
            te0, te1 = C.class_counts(test)
            if min(tr0, tr1, te0, te1) <= 0:
                raise ValueError('Eğitim ve testte her iki sınıf da bulunmalı.')
            if experiment == 'scaling':
                expected = metadata['sizes'][str(size)]
                if (tr0, tr1, te0 + te1, te1) != tuple(expected[k] for k in ('train_normal', 'train_fraud', 'test_rows', 'test_fraud')):
                    raise ValueError('Parquet sayıları SECILEN_YONTEM.json ile uyuşmuyor.')
            elif n0 + n1 + te0 + te1 != size:
                raise ValueError('Features satır sayısı beklenen büyüklükte değil.')
            prepare_sec = time.perf_counter() - t

            # Pipeline ve sınıflandırıcı sadece dengelenmiş train'e fit edilir.
            t = time.perf_counter()
            pipeline = Pipeline(stages=[*C.build_feature_stages(C.MODEL_FEATURES), classifier(name)])
            fitted = pipeline.fit(train)
            train_sec = time.perf_counter() - t

            # Cache edilen prediction ve rawPrediction gerçekten hesaplanır; sadece transform süresi değil.
            t = time.perf_counter()
            pred = keep(fitted.transform(test).select(LABEL, 'prediction', 'rawPrediction'))
            predict_sec = time.perf_counter() - t

            # Öğrenci 2'nin ortak değerlendirme yöntemi: F1 ve Spark areaUnderPR.
            t = time.perf_counter()
            metrics = evaluate(pred)
            eval_sec = time.perf_counter() - t
            if sum(metrics[k] for k in ('TP', 'FP', 'FN', 'TN')) != te0 + te1:
                raise ValueError('Tahmin satır sayısı test ile uyuşmuyor.')

        row = {'size': size, 'model': name, 'target_ratio': ratio, 'repeat': repeat,
               'input': 'balanced' if experiment == 'scaling' else 'features',
               'method': 'none' if ratio == 'original' else 'undersample',
               'source_rows': size, 'train_rows': tr0 + tr1, 'train_normal': tr0, 'train_fraud': tr1,
               'test_rows': te0 + te1, 'test_fraud': te1, 'actual_normal_per_fraud': tr0 / tr1,
               'threshold': 0.5, 'read_sec': read_sec, 'prepare_sec': prepare_sec,
               'balance_sec': balance_sec, 'train_sec': train_sec, 'predict_sec': predict_sec,
               'eval_sec': eval_sec, 'throughput': (te0 + te1) / predict_sec,
               **metrics, **memory.values(), **env, 'finished_at': C.now_iso()}
        return row
    finally:
        # Önbellek başka bir ölçüme taşınmaz (JVM'in kendi ayırdığı RAM kalabilir).
        for df in reversed(cached):
            df.unpersist(blocking=True)
        spark.catalog.clearCache()


def csv_path(out, experiment):
    return out / FILES[experiment]


def read_results(out, experiment):
    # Ortak yardımcı: yarım CSV satırları atılır, tekrarlanan anahtarın son satırı kalır.
    return C.read_rows(csv_path(out, experiment), KEYS, REQUIRED, COLS)


def selected_cases(args, experiment):
    ratios = ['1:10'] if experiment == 'scaling' else args.ratios
    return [(n, model, ratio) for n in args.sizes for model in args.models for ratio in ratios]


def clear_selected(out, experiment, cases):
    """--force: yalnızca seçilen kombinasyonların eski ölçümlerini kaldırır."""
    path = csv_path(out, experiment)
    if not path.exists():
        return
    df = read_results(out, experiment)
    selected = {(str(n), m, r) for n, m, r in cases}
    keep = [tuple(row) not in selected for row in df[['size', 'model', 'target_ratio']].itertuples(index=False, name=None)]
    tmp = path.with_suffix('.tmp')
    df.loc[keep, COLS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def aggregate(out, experiment):
    """Yalnızca üç geçerli ölçümü bulunan kombinasyonların medyanı."""
    df = read_results(out, experiment)
    if df.empty:
        return pd.DataFrame()
    df = df[df['repeat'].isin([str(i) for i in range(1, REPEATS + 1)])].copy()
    df['size'] = pd.to_numeric(df['size'])
    numeric = MEASUREMENTS + ['train_rows', 'test_rows', 'actual_normal_per_fraud', 'train_fraud', 'test_fraud']
    for k in numeric:
        df[k] = pd.to_numeric(df[k], errors='coerce')
    group_cols = ['size', 'model', 'target_ratio']
    rows = []
    for keys, group in df.groupby(group_cols):
        if len(group) != REPEATS:
            continue
        # Donanım/yazılım değişmişse aynı kombinasyonun medyanını karıştırma.
        settings = ['cpu_count', 'cpu_model', 'partitions', 'shuffle_partitions', 'driver_memory',
                    'spark_version', 'python_version', 'ram_mb', 'memory_method']
        if any(group[k].nunique() != 1 for k in settings):
            print(f'UYARI: {experiment} {keys}: ortam değişmiş; --force ile yeniden ölçün.')
            continue
        rows.append({**dict(zip(group_cols, keys)), **group[numeric].median().to_dict(),
                     'runtime_count': group['runtime_id'].nunique(),
                     **{k: group[k].iloc[0] for k in settings}})
    return pd.DataFrame(rows)


def plot_results(out):
    """CSV medyanlarından grafikler; sayılar elle girilmez."""
    plt = C.plt_setup()
    scaling = aggregate(out, 'scaling')
    imbalance = aggregate(out, 'imbalance')
    if not scaling.empty:
        specs = [
            ('time_vs_size.png', [('read_sec', 'Okuma (sn)'), ('prepare_sec', 'Hazırlama (sn)'),
                                  ('train_sec', 'Eğitim (sn)'), ('predict_sec', 'Tahmin (sn)')], True),
            ('memory_vs_size.png', [('rss_peak_mb', 'JVM tepe RSS (MB)'), ('rss_increase_mb', 'JVM RSS artışı (MB)')], False),
            ('throughput.png', [('throughput', 'Tahmin throughput (satır/sn)')], False),
            ('performance_vs_size.png', [('f1', 'F1'), ('auprc', 'AUPRC')], False),
        ]
        for filename, metrics, loglog in specs:
            fig, axs = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4), squeeze=False)
            for ax, (metric, title) in zip(axs[0], metrics):
                for i, (name, group) in enumerate(scaling.groupby('model')):
                    group = group.sort_values('size')
                    ax.plot(group['size'], group[metric], 'o-', label=name.upper(), color=C.SERIES[i])
                if loglog:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                else:
                    ax.set_xscale('log')
                ax.set_xticks(C.DEFAULT_SIZES, [C.size_label(n) for n in C.DEFAULT_SIZES])
                ax.set_xlabel('Başlangıç veri büyüklüğü')
                if metric in ('f1', 'auprc'):
                    ax.set_ylim(0, 1)
                C.style_ax(ax, title)
                ax.legend()
            C.save_fig(fig, out / filename)
    if not imbalance.empty:
        for filename, metrics in [('imbalance_vs_performance.png', [('f1', 'F1'), ('auprc', 'AUPRC')]),
                                  ('imbalance_vs_time.png', [('train_sec', 'Eğitim (sn)'), ('predict_sec', 'Tahmin (sn)')]),
                                  ('imbalance_vs_resources.png', [('rss_peak_mb', 'JVM RSS (MB)'), ('throughput', 'Satır/sn')])]:
            fig, axs = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4), squeeze=False)
            for ax, (metric, title) in zip(axs[0], metrics):
                for i, ((size, name), group) in enumerate(imbalance.groupby(['size', 'model'])):
                    group = group.assign(order=group['target_ratio'].map({r: j for j, r in enumerate(RATIOS)})).sort_values('order')
                    ax.plot(group['order'], group[metric], 'o-', color=C.SERIES[i], label=f'{C.size_label(size)} {name.upper()}')
                ax.set_xticks(range(len(RATIOS)), ['Orijinal', *RATIOS[1:]])
                ax.set_xlabel('Hedef fraud:normal oranı')
                if metric in ('f1', 'auprc'):
                    ax.set_ylim(0, 1)
                C.style_ax(ax, title)
                ax.legend(fontsize=7)
            C.save_fig(fig, out / filename)
    return scaling, imbalance


def findings(out, scaling, imbalance):
    """Tamamlanmış üçlü ölçümlerden kısa, veri odaklı bulgular."""
    lines = ['# Öğrenci 5 — Ölçeklenebilirlik bulguları', '',
             'Yalnızca üç geçerli ölçümü tamamlanan kombinasyonlar gösterilir; süreler ve metrikler medyandır.', '',
             '## Veri büyüklüğü', '']
    if scaling.empty:
        lines.append('Henüz tamamlanmış boyut deneyi yok.')
    else:
        for name, group in scaling.groupby('model'):
            group = group.sort_values('size')
            lines += [f'### {name.upper()}', '', '| Boyut | Train | Test | Eğitim (sn) | Tahmin (sn) | RSS tepe (MB) | Satır/sn | F1 | AUPRC |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in group.itertuples():
                lines.append(f'| {C.size_label(r.size)} | {r.train_rows:.0f} | {r.test_rows:.0f} | {r.train_sec:.3f} | {r.predict_sec:.3f} | {r.rss_peak_mb:.1f} | {r.throughput:.1f} | {r.f1:.4f} | {r.auprc:.4f} |')
            a, b = group.iloc[0], group.iloc[-1]
            if len(group) > 1:
                if all(a[k] == b[k] for k in ['cpu_count', 'cpu_model', 'partitions', 'shuffle_partitions', 'driver_memory', 'spark_version', 'python_version', 'ram_mb']):
                    lines += ['', f"{C.size_label(int(a['size']))} → {C.size_label(int(b['size']))}: başlangıç verisi {b['size']/a['size']:.2f} kat, "
                              f"eğitim süresi {b['train_sec']/a['train_sec']:.2f} kat, tahmin süresi {b['predict_sec']/a['predict_sec']:.2f} kat, "
                              f"tepe RSS {b['rss_peak_mb']/a['rss_peak_mb']:.2f} kat; F1 değişimi {b['f1']-a['f1']:+.4f}, AUPRC değişimi {b['auprc']-a['auprc']:+.4f}."]
                else:
                    lines += ['', 'Uç boyutların donanım/yazılım ayarları farklı; süre ve bellek katları karşılaştırılmadı.']
            lines.append('')
    lines += ['## Dengesizlik oranı', '']
    if imbalance.empty:
        lines.append('Henüz tamamlanmış oran deneyi yok.')
    else:
        for (n, name), group in imbalance.groupby(['size', 'model']):
            best = group.loc[group['f1'].idxmax()]
            lines.append(f"- {C.size_label(n)} {name.upper()}: tamamlanan oranlar arasında en yüksek F1 {best['target_ratio']} ({best['f1']:.4f}); AUPRC {best['auprc']:.4f}." )
    lines += ['', '## Öğrenci 2 ile bağlam karşılaştırması', '']
    old_path = C.RESULTS_DIR / 'ogrenci2' / 'imbalance_results.csv'
    if old_path.exists():
        old = pd.read_csv(old_path)
        old = old[old['input'] == 'features']
        for n, g in old.groupby('size'):
            r = g.loc[g['f1'].idxmax()]
            lines.append(f"- {C.size_label(int(n))}: Öğrenci 2 en yüksek F1 = {r['f1']:.4f} ({r['method']}, {r['target_ratio']}).")
    lines += ['', '## Sınırlamalar', '',
              '- Başlangıç büyüklüğü, gerçek eğitim satırı değildir; dengelenmiş train daha küçüktür.',
              '- Kaggle kaynağı simülasyondur; 5M büyüklüğünde ek sentetik kopyalar bağımsız yeni gözlemler değildir. Boyutlar arasında davranış özellikleri ve test kümeleri de değişir.',
              '- Oran deneyinde undersampling eğitim satırı sayısını da değiştirir; test her boyut içinde sabittir. Rastgele örneklemede hedef oran yaklaşık gerçekleşir.',
              '- Okuma süresi, seçilen sütunların ilk cache+count ile okunmasını; hazırlama süresi split, örnekleme, repartition ve cache+count işlemlerini içerir. Eğitim özellik pipeline fit ve model fit; tahmin pipeline transform ve tahmin cache+count süresidir. F1/AUPRC hesabı ayrı ölçülür.',
              '- RSS 0,2 sn örneklemeli JVM fiziksel belleğidir; tüm sistem belleği veya gerçek mutlak tepe garantisi değildir. JVM önceki ayırdığı belleği tutabilir.',
              '- Colab oturumu ve disk önbelleği süreyi etkileyebilir. Farklı ortamların süreleri doğrudan eşdeğer sayılmaz; Öğrenci 2 süreleri bu nedenle kullanılmaz.',
              '- Modellerin ayarları ve eşik sabittir; amaç en iyi modeli seçmek değildir.']
    # Sentetik payı mevcut kayıtlı Öğrenci 1 CSV'sinde varsa gerçek değerinden alınır.
    src = C.RESULTS_DIR / 'ogrenci1' / 'write_stats.csv'
    if src.exists():
        original = pd.read_csv(src)
        if {'size', 'synthetic_rows'}.issubset(original.columns):
            vals = original.loc[original['size'] == max(C.DEFAULT_SIZES), 'synthetic_rows']
            if not vals.empty:
                lines.append(f"- Öğrenci 1 CSV'sine göre en büyük boyutta ek sentetik payı: %{100 * vals.iloc[0] / max(C.DEFAULT_SIZES):.2f}.")
    (out / 'bulgular.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sizes', nargs='+', type=int, choices=C.DEFAULT_SIZES, default=C.DEFAULT_SIZES)
    ap.add_argument('--models', nargs='+', choices=['lr', 'rf'], default=['lr', 'rf'])
    ap.add_argument('--experiments', nargs='+', choices=list(FILES), default=list(FILES))
    ap.add_argument('--ratios', nargs='+', choices=RATIOS, default=RATIOS)
    ap.add_argument('--out', type=Path, default=C.RESULTS_DIR / 'ogrenci5')
    ap.add_argument('--driver_memory', default='8g')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--plot_only', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    log = C.setup_logging(out, 'ogrenci5')
    if args.plot_only:
        findings(out, *plot_results(out))
        return
    metadata = verify_metadata() if 'scaling' in args.experiments else None
    # Tamamlanan tekrarlar tek tek atlanır. Force sadece seçilen kombinasyonlara uygulanır.
    todo = []
    for experiment in args.experiments:
        cases = selected_cases(args, experiment)
        if args.force:
            clear_selected(out, experiment, cases)
        done = C.done_set(csv_path(out, experiment), KEYS, REQUIRED)
        for n, name, ratio in cases:
            pending = [rep for rep in range(1, REPEATS + 1) if (str(n), name, ratio, str(rep)) not in done]
            if pending:
                todo.append((experiment, n, name, ratio, pending))
    if todo:
        spark = C.get_spark('ogrenci5-olcek', driver_memory=args.driver_memory)
        try:
            env = environment(spark)
            log.info('Ortam: %s', env)
            for experiment, n, name, ratio, pending in todo:
                tag = f'{experiment} {C.size_label(n)} {name.upper()} {ratio}'
                log.info('%s | Isınma başladı (CSV medyanına dahil değil)', tag)
                run_measurement(spark, experiment, n, name, ratio, 0, env, metadata)
                for rep in pending:
                    log.info('%s | Ölçüm %s/%s başladı', tag, rep, REPEATS)
                    row = run_measurement(spark, experiment, n, name, ratio, rep, env, metadata)
                    C.append_row(row, csv_path(out, experiment), COLS)
                    log.info('%s | Ölçüm %s kaydedildi | eğitim=%.3fs tahmin=%.3fs F1=%.4f AUPRC=%.4f RSS=%.1f MB',
                             tag, rep, row['train_sec'], row['predict_sec'], row['f1'], row['auprc'], row['rss_peak_mb'])
        finally:
            spark.stop()
    else:
        log.info('Seçilen tüm ölçümler mevcut; deney atlandı.')
    C.safe_plot(lambda: findings(out, *plot_results(out)), log=log.warning)
    log.info('Çıktılar: %s', out)


if __name__ == '__main__':
    main()
