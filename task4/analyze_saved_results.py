"""Recompute descriptive report metrics from saved validation predictions.

Run from the task4 directory. No model training or external downloads.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

PROJECT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/report_analysis.json')
args = parser.parse_args()

def read_json(name):
    return json.loads((PROJECT / name).read_text(encoding='utf-8'))

def to512(mask):
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((512, 512), Image.Resampling.BILINEAR)) > 0

def counts(pred, target):
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    tn = int(np.count_nonzero(~pred & ~target))
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'dice': (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7),
            'iou': tp / (tp + fp + fn), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn)}

selection = read_json('outputs/improved/ensemble_selection.json')
base_summary = read_json('outputs/validation/summary.json')
new_summary = read_json('outputs/improved/validation_context/summary.json')
val_ids = selection['validation_ids']
images, labels, stats = {}, {}, []
for i in map(str, range(1, 21)):
    image = np.array(Image.open(PROJECT / f'segmentation/train/image/{i}.jpg').convert('RGB'))
    label = np.array(Image.open(PROJECT / f'segmentation/train/label/{i}.jpg').convert('L')) > 128
    images[i], labels[i] = image, label
    field = image.max(axis=2) > 20
    pixels = image[field].astype(np.float32) / 255
    stats.append({'id': i, 'split': '验证' if i in val_ids else '训练',
                  'width': image.shape[1], 'height': image.shape[0],
                  'vessel_pixels': int(label.sum()), 'vessel_fraction': float(label.mean()),
                  'vessel_fraction_in_field': float(label[field].mean()),
                  'field_fraction': float(field.mean()), 'green_mean': float(pixels[:, 1].mean()),
                  'rgb_mean': pixels.mean(axis=0).tolist(), 'rgb_std': pixels.std(axis=0).tolist()})

runs = ['outputs/validation', 'outputs/improved/baseline_d4', 'outputs/improved/validation_context']
caches = [np.load(PROJECT / run / 'validation_predictions.npz') for run in runs]
for cache in caches:
    assert cache['ids'].tolist() == val_ids
probs = [c['probabilities'] for c in caches]
probs.append((probs[1] + probs[2]) / 2)
per_strategy, threshold_curves = [], []
for probabilities, candidate in zip(probs, selection['comparisons']):
    per_native, per_converted = [], []
    t = candidate['threshold']
    for i, p in zip(val_ids, probabilities):
        gt = labels[i]
        native = counts(p > t, gt)
        native['id'] = i
        distance = distance_transform_edt(gt)
        edge = gt & (distance <= 2)
        interior = gt & (distance > 2)
        native['boundary_band_pixels'] = int(edge.sum())
        native['interior_pixels'] = int(interior.sum())
        native['boundary_band_recall'] = float((p > t)[edge].mean())
        native['interior_recall'] = float((p > t)[interior].mean())
        converted = counts(to512(p > t), to512(gt))
        converted['id'] = i
        per_native.append(native)
        per_converted.append(converted)
    mean_dice = float(np.mean([c['dice'] for c in per_converted]))
    assert abs(mean_dice - candidate['mean_dice']) < 1e-12
    assert abs(float(np.mean([c['dice'] for c in per_native])) - candidate['native_mean_dice']) < 1e-12
    pooled = {}
    for label, values in [('native', per_native), ('converted', per_converted)]:
        tp, fp, fn, tn = [sum(c[k] for c in values) for k in ['tp', 'fp', 'fn', 'tn']]
        pooled[label] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                         'dice': 2 * tp / (2 * tp + fp + fn),
                         'precision': tp / (tp + fp), 'recall': tp / (tp + fn),
                         'mean_iou': float(np.mean([c['iou'] for c in values]))}
    curve = []
    for threshold in np.arange(.2, .751, .025):
        values = [counts(to512(p > threshold), to512(labels[i]))['dice'] for i, p in zip(val_ids, probabilities)]
        curve.append({'threshold': float(threshold), 'dice': float(np.mean(values))})
    assert abs(max(p['dice'] for p in curve) - mean_dice) < 1e-12
    per_strategy.append({'strategy': candidate['strategy'], 'threshold': t,
                         'per_native': per_native, 'per_converted': per_converted, 'pooled': pooled})
    threshold_curves.append({'strategy': candidate['strategy'], 'points': curve})

histories = []
for run in ['outputs/validation', 'outputs/improved/validation_context']:
    with (PROJECT / run / 'history.csv').open(encoding='utf-8') as handle:
        histories.append([{k: (float(v) if v else None) for k, v in row.items()} for row in csv.DictReader(handle)])
costs = []
for summary, history, final_path in zip([base_summary, new_summary], histories,
                                       ['outputs/final/summary.json', 'outputs/improved/final_context/summary.json']):
    config = summary['config']
    selected = summary['best_epoch']
    sampled = selected * config['steps_per_epoch'] * config['batch_size']
    costs.append({'parameters': config['parameters'], 'selected_epoch': selected,
                  'training_epoch_budget': config['epochs'], 'batch_size': config['batch_size'],
                  'patch_size': config['patch_size'], 'gradient_updates': selected * config['steps_per_epoch'],
                  'sampled_patches': sampled, 'sampled_pixels': sampled * config['patch_size'] ** 2,
                  'development_seconds': history[-1]['elapsed_seconds'],
                  'full_training_seconds': read_json(final_path)['elapsed_seconds'],
                  'initial_loss': history[0]['train_loss'], 'last_loss': history[-1]['train_loss'],
                  'selected_loss': history[selected - 1]['train_loss']})

threshold_details = []
for curve in threshold_curves:
    best = max(curve['points'], key=lambda x: x['dice'])
    stable = [p['threshold'] for p in curve['points'] if p['dice'] >= best['dice'] - .001]
    threshold_details.append({'strategy': curve['strategy'], 'best': best,
                              'within_0_001_range': [min(stable), max(stable)],
                              'at_0_475': min(curve['points'], key=lambda x: abs(x['threshold'] - .475))})

conversion_stats = []
for i in val_ids:
    gt = labels[i]
    nearest = np.array(Image.fromarray(gt.astype('uint8') * 255).resize((512, 512), Image.Resampling.NEAREST)) > 0
    converted = to512(gt)
    conversion_stats.append({'id': i, 'native_vessel_pixels': int(gt.sum()),
                             'native_fraction': float(gt.mean()), 'nearest_512_pixels': int(nearest.sum()),
                             'bilinear_nonzero_512_pixels': int(converted.sum()),
                             'extra_vs_nearest': int(converted.sum() - nearest.sum()),
                             'converted_fraction': float(converted.mean())})

analysis = {'student': '李星翰', 'student_id': '20245624', 'validation_ids': val_ids,
             'data_stats': stats, 'strategies': per_strategy, 'threshold_curves': threshold_curves,
             'threshold_details': threshold_details, 'costs': costs,
             'conversion_stats': conversion_stats,
             'mean_foreground_fraction': float(np.mean([s['vessel_fraction'] for s in stats])),
             'foreground_fraction_range': [min(s['vessel_fraction'] for s in stats), max(s['vessel_fraction'] for s in stats)],
             'additional_analysis_scope': '既有验证概率的描述性分析；没有新增训练或独立测试。'}

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved', args.output)
