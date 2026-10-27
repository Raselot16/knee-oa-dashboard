"""Data summaries, model inference and explicitly separate symptom guidance."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd

FEATURES = ['age', 'bmi', 'smoking', 'osteoporosis_baseline']
TARGET = 'koa_followup'
MODEL_NAMES = {'LogisticRegression': 'Logistic Regression', 'RandomForest': 'Random Forest'}
LABEL_NAMES = ['Nhãn 0 · at risk', 'Nhãn 1 · diagnosed', 'Chưa có nhãn']
RED_FLAGS = [
    'Đau gối rất nhiều',
    'Không cử động được gối hoặc không chịu được lực lên chân',
    'Gối sưng nhiều hoặc biến dạng',
    'Gối bị kẹt, khuỵu hoặc kêu kèm đau',
    'Sốt hoặc rét run kèm gối đỏ hay nóng',
]
SOURCES = {
    'Chăm sóc khớp · NIAMS': 'https://www.niams.nih.gov/health-topics/osteoarthritis/diagnosis-treatment-and-steps-to-take',
    'Đau gối và dấu hiệu cần khám sớm · NHS': 'https://www.nhs.uk/symptoms/knee-pain/',
    'Loãng xương · NIAMS': 'https://www.niams.nih.gov/health-topics/osteoporosis/diagnosis-treatment-and-steps-to-take',
    'Tài liệu dữ liệu MatchThem': 'https://search.r-project.org/CRAN/refmans/MatchThem/html/osteoarthritis.html',
    'Nghiên cứu OAI · NIAMS': 'https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_model(base):
    base = Path(base)
    report_path = base / 'results/model_comparison/comparison.json'
    if report_path.is_file():
        report = read_json(report_path)
        name = report.get('selected_model_by_cv')
        if name not in MODEL_NAMES:
            raise ValueError('Tên mô hình được chọn trong comparison.json không hợp lệ.')
        model_path = report_path.parent / (name + '.joblib')
        kind = 'comparison'
    else:
        report_path = base / 'results/logistic_baseline/metrics.json'
        report = read_json(report_path)
        name, kind = 'LogisticRegression', 'baseline'
        model_path = report_path.parent / 'logistic_baseline.joblib'
    if not model_path.is_file():
        raise FileNotFoundError('Chưa có mô hình đã huấn luyện. Hãy chạy train_baseline.py và compare_models.py.')
    return model_path, name, report_path, report, kind


def dataset_summary(frame, title):
    frame = frame.copy()
    for c in FEATURES + [TARGET]:
        frame[c] = pd.to_numeric(frame[c], errors='raise')
    n = len(frame)
    if not n:
        raise ValueError('Bảng dữ liệu rỗng.')
    counts = [int(frame[TARGET].eq(0).sum()), int(frame[TARGET].eq(1).sum()), int(frame[TARGET].isna().sum())]
    if sum(counts) != n:
        raise ValueError('Nhãn trong dữ liệu phải là 0/1 hoặc để trống.')
    age_edges = np.arange(60, 85, 5, dtype=float)
    bmi_edges = np.arange(15, 55, 5, dtype=float)
    age_hist, _ = np.histogram(frame.age.dropna(), bins=age_edges)
    bmi_hist, _ = np.histogram(frame.bmi.dropna(), bins=bmi_edges)
    pairs = frame[['age', 'bmi']].dropna()
    heat, _, _ = np.histogram2d(pairs.age, pairs.bmi, bins=[age_edges, bmi_edges])
    groups = []
    bins = pd.cut(frame.age, bins=age_edges, right=False)
    for interval in bins.cat.categories:
        g = frame.loc[bins.eq(interval) & frame[TARGET].notna()]
        groups.append({'group': f'{int(interval.left)}–{int(interval.right)-1}',
                       'labeled_n': len(g), 'label_1_n': int(g[TARGET].eq(1).sum())})
    return {
        'title': title, 'n': n, 'label_counts': counts,
        'age_range': [float(frame.age.min()), float(frame.age.max())],
        'bmi_range': [float(frame.bmi.min()), float(frame.bmi.max())],
        'age_median': float(frame.age.median()), 'bmi_median': float(frame.bmi.median()),
        'missing': {str(c): int(frame[c].isna().sum()) for c in frame.columns if c != 'record_key'},
        'age_edges': age_edges.tolist(), 'age_hist': age_hist.astype(int).tolist(),
        'bmi_edges': bmi_edges.tolist(), 'bmi_hist': bmi_hist.astype(int).tolist(),
        'heatmap': heat.astype(int).tolist(), 'paired_n': len(pairs), 'age_groups': groups,
    }


def build_summary(base, model, model_path, report_path, report):
    base = Path(base)
    csv_path = base / 'knee_oa_labeled_2392.csv'
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if any(c not in df for c in FEATURES + [TARGET, 'record_key']):
        raise ValueError('CSV thiếu cột cần thiết.')
    if df.record_key.isna().any() or df.record_key.duplicated().any():
        raise ValueError('record_key phải đầy đủ và duy nhất trong CSV có nhãn.')
    if df[TARGET].isna().any():
        raise ValueError('CSV dùng cho mô hình phải có nhãn đầy đủ.')
    if report.get('source_sha256') != sha(csv_path):
        raise ValueError('CSV đã khác lần huấn luyện. Hãy dùng đúng dữ liệu hoặc huấn luyện và so sánh lại.')
    ds = {'labeled': dataset_summary(df, 'Hồ sơ có nhãn · dữ liệu mô hình')}
    all_path = base / 'matchthem_osteoarthritis_all_2585.csv'
    if all_path.is_file():
        full = pd.read_csv(all_path, encoding='utf-8-sig').rename(columns={
            'AGE': 'age', 'SEX': 'sex_code_unverified', 'BMI': 'bmi', 'RAC': 'race_code',
            'SMK': 'smoking', 'OSP': 'osteoporosis_baseline', 'KOA': TARGET})
        # Verify that this is an extension of the same labeled data, preserving duplicate multiplicities.
        cols = FEATURES + [TARGET]
        left = full.loc[full[TARGET].notna(), cols].astype(float).fillna(-9999).value_counts(dropna=False).sort_index()
        right = df[cols].astype(float).fillna(-9999).value_counts(dropna=False).sort_index()
        if not left.equals(right):
            raise ValueError('Bảng ban đầu và bảng có nhãn không khớp. Không trộn hai nguồn dữ liệu.')
        ds['all'] = dataset_summary(full, 'Toàn bộ hồ sơ ban đầu')
    train, train_verified = df, False
    split_path = base / 'results/logistic_baseline/split.csv'
    if split_path.is_file():
        split = pd.read_csv(split_path, encoding='utf-8-sig')
        if split.record_key.isna().any() or split.record_key.duplicated().any() or set(split.record_key) != set(df.record_key):
            raise ValueError('Danh sách train/test không khớp dữ liệu.')
        if set(split['split']) != {'train', 'test'}:
            raise ValueError('Danh sách train/test không hợp lệ.')
        train = df.loc[df.record_key.isin(split.loc[split['split'].eq('train'), 'record_key'])]
        if len(train) != report.get('train_rows', report.get('split', {}).get('train_rows')):
            raise ValueError('Số hồ sơ train không khớp báo cáo.')
        train_verified = True
    limits = {c: [float(train[c].min()), float(train[c].max())] for c in ['age', 'bmi']}
    score_reference = None
    if train_verified:
        scores = np.asarray(model.predict_proba(train[FEATURES])[:, list(model.classes_).index(1)], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('Điểm tham chiếu của mô hình không hợp lệ.')
        q = np.quantile(scores, [1/3, 2/3])
        hist, edges = np.histogram(scores, bins=np.linspace(0, 1, 26))
        score_reference = {'n': len(scores), 'tertiles': q.tolist(), 'hist': hist.astype(int).tolist(),
                           'edges': edges.tolist(), 'source': 'model predictions on saved training rows',
                           'clinical_thresholds': False}
    return {
        'schema_version': 2, 'source_sha256': sha(csv_path), 'model_sha256': sha(model_path),
        'report_sha256': sha(report_path), 'datasets': ds, 'input_ranges': limits,
        'ranges_from_training': train_verified, 'score_reference': score_reference,
        'created_date': '2026-10-27', 'sources': SOURCES,
    }


def load_context(base):
    base = Path(base)
    model_path, name, report_path, report, kind = resolve_model(base)
    bundle = joblib.load(model_path)
    if bundle.get('features') != FEATURES or bundle.get('target') != TARGET:
        raise ValueError('Mô hình không dùng đúng bốn đặc trưng và nhãn của giao diện.')
    model = bundle['pipeline']
    if set(model.classes_) != {0, 1}:
        raise ValueError('Mô hình phải có hai nhãn 0 và 1.')
    if (base/'knee_oa_labeled_2392.csv').is_file():
        summary = build_summary(base, model, model_path, report_path, report)
    elif (base/'dashboard_data.json').is_file():
        summary = read_json(base/'dashboard_data.json')
        if summary.get('schema_version') != 2 or summary.get('model_sha256') != sha(model_path) or summary.get('report_sha256') != sha(report_path):
            raise ValueError('Thống kê dashboard không khớp mô hình/báo cáo. Hãy tạo lại bằng prepare_cloud.py.')
        if summary.get('source_sha256') != report.get('source_sha256'):
            raise ValueError('Nguồn của dashboard không khớp báo cáo.')
    else:
        raise FileNotFoundError('Thiếu CSV hoặc dashboard_data.json. Đặt giao diện vào đúng thư mục dự án.')
    return {'model': model, 'model_name': name, 'model_path': model_path, 'report_path': report_path,
            'report': report, 'report_kind': kind, 'summary': summary}


def relative_group(score, reference):
    if not reference:
        return None
    low, high = reference['tertiles']
    if not np.isfinite([low, high]).all() or high <= low:
        return None
    key = 'low' if score < low else 'middle' if score < high else 'high'
    return {'key': key, 'name': {'low': 'Điểm tương đối thấp', 'middle': 'Điểm tương đối trung gian',
                                 'high': 'Điểm tương đối cao'}[key], 'cutoffs': [low, high]}


def care_direction(flags, checked, persistent):
    if any(flag not in RED_FLAGS for flag in flags):
        raise ValueError('Dấu hiệu không hợp lệ.')
    if flags:
        return {'level': 'urgent', 'title': 'Cần được đánh giá y tế sớm',
                'message': 'Bạn khai có dấu hiệu cần được khám khẩn. Hãy liên hệ cơ sở y tế ngay để được hướng dẫn; đến cơ sở cấp cứu nếu tình trạng nặng. Không chờ kết quả mô hình.',
                'basis': flags, 'source': SOURCES['Đau gối và dấu hiệu cần khám sớm · NHS']}
    if not checked or persistent is None:
        return {'level': 'unknown', 'title': 'Chưa đủ thông tin về triệu chứng',
                'message': 'Bạn có thể xem thông tin chăm sóc chung. Cần trả lời phần triệu chứng để nhận hướng dẫn về thời điểm đi khám.',
                'basis': [], 'source': SOURCES['Đau gối và dấu hiệu cần khám sớm · NHS']}
    if persistent:
        return {'level': 'appointment', 'title': 'Nên đặt lịch khám',
                'message': 'Đau gối không cải thiện sau vài tuần cần được nhân viên y tế đánh giá.',
                'basis': ['Bạn khai đau gối không cải thiện sau vài tuần.'], 'source': SOURCES['Đau gối và dấu hiệu cần khám sớm · NHS']}
    return {'level': 'selfcare', 'title': 'Theo dõi và chăm sóc thường ngày',
            'message': 'Câu trả lời hiện chưa ghi nhận dấu hiệu cần khám khẩn. Điều này không xác nhận bạn khỏe mạnh hoặc loại trừ bệnh.',
            'basis': ['Bạn xác nhận đã kiểm tra dấu hiệu và không khai đau kéo dài.'], 'source': SOURCES['Đau gối và dấu hiệu cần khám sớm · NHS']}


def assess(context, inputs, flags=(), checked=False, persistent=None):
    care = care_direction(flags, checked, persistent)
    issues = []
    for c in FEATURES:
        if inputs.get(c) is None:
            issues.append('Chưa nhập ' + {'age':'tuổi','bmi':'BMI','smoking':'hút thuốc','osteoporosis_baseline':'tình trạng loãng xương đã biết'}[c])
    if not issues:
        for c, label in [('age', 'Tuổi'), ('bmi', 'BMI')]:
            lo, hi = context['summary']['input_ranges'][c]
            if not np.isfinite(inputs[c]) or not lo <= inputs[c] <= hi:
                issues.append(f'{label} ngoài phạm vi dữ liệu tham chiếu ({lo:g}–{hi:g}); chưa hỗ trợ dự đoán hồ sơ này.')
        if any(inputs[c] not in [0, 1] for c in ['smoking','osteoporosis_baseline']):
            issues.append('Thông tin nhị phân phải có mã 0 hoặc 1.')
    prediction = None
    if not issues:
        x = pd.DataFrame([inputs], columns=FEATURES)
        model = context['model']
        p = float(model.predict_proba(x)[0, list(model.classes_).index(1)])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Điểm mô hình không hợp lệ.')
        prediction = {'label': int(model.predict(x)[0]), 'score': p,
                      'relative_group': relative_group(p, context['summary'].get('score_reference'))}
    return {'inputs': inputs, 'prediction': prediction, 'issues': issues, 'care': care,
            'symptoms': {'flags': list(flags), 'checked': bool(checked), 'persistent': persistent},
            'model_name': context['model_name'], 'model_sha256': context['summary']['model_sha256'],
            'scope': 'Experimental KOA label; relative score groups are not validated clinical risk categories.'}
