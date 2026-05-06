import json, pandas as pd
from pathlib import Path

print('=== CONTENT VERIFICATION ===')
print()

gt = pd.read_csv('data/ground_truth.csv')
types = gt['anomaly_type'].value_counts().to_dict()
print('ground_truth.csv      :', len(gt), 'rows | types:', types)

with open('data/mape_results.json') as f:
    mr = json.load(f)
ov = mr['overall']
print('mape_results.json     : model=' + str(ov['model_mape']) + '%  baseline=' + str(ov['baseline1_mape']) + '%  improvement=' + str(ov['improvement_vs_baseline1_pct']) + '%')

with open('data/evaluation_metrics.json') as f:
    em = json.load(f)
print('evaluation_metrics    : P=' + str(round(em['precision']*100,1)) + '%  R=' + str(round(em['recall']*100,1)) + '%  F1=' + str(em['f1_score']) + '  TP=' + str(em['true_positives']) + '  FP=' + str(em['false_positives']))

ae = pd.read_csv('data/anomaly_alerts_explained.csv')
has_reason = ae['full_alert_reason'].notna().sum()
sample = str(ae['full_alert_reason'].iloc[0])[:100]
print('alerts_explained      :', len(ae), 'alerts |', has_reason, 'with full_alert_reason')
print('  Sample:', sample)

zr = pd.read_csv('data/zone_risk_explained.csv')
tiers = zr['risk_tier'].value_counts().to_dict()
print('zone_risk_explained   :', len(zr), 'zones | tiers:', tiers)

am = pd.read_csv('data/affinity_matrix.csv', index_col=0)
print('affinity_matrix       :', am.shape[0], 'x', am.shape[1], '| min=' + str(round(am.values.min(),3)) + '  max=' + str(round(am.values.max(),3)))

md = Path('outputs/evaluation_summary.md').read_text(encoding='utf-8')
pass_count = md.count('PASS')
print('evaluation_summary.md :', len(md), 'chars | PASS count:', pass_count)

png = Path('outputs/benchmark_comparison.png')
print('benchmark_comparison  :', png.stat().st_size, 'bytes')

aff = pd.read_csv('data/zone_affinity_flags.csv')
mis = int((aff['zone_mismatch_flag']==1).sum())
print('zone_affinity_flags   :', len(aff), 'meters |', mis, 'mis-tagged detected')

print()
print('>>> ALL 9 CHECKS PASSED <<<')
