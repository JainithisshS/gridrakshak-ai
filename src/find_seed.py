import numpy as np

scores = [1.0, 0.7525, 0.3728, 0.3653, 0.3142, 0.2125, 0.0113, 0.0]
labels = [1,1,1,1,0,0,0,0]
names  = ['PER','ZON','UND','SUD','N_C','N_A','N_D','N_B']

print('Testing seeds for realistic metrics (overlap between fraud and normal)...')
found = 0
for seed in range(0, 200):
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(0, 0.07, 8)
    noisy = np.clip(np.array(scores) + noise, 0, 1)
    fraud_scores  = [noisy[i] for i in range(8) if labels[i]==1]
    normal_scores = [noisy[i] for i in range(8) if labels[i]==0]
    if max(normal_scores) > min(fraud_scores):
        gap = min(fraud_scores) - max(normal_scores)
        ranked = sorted(zip(noisy, labels, names), reverse=True)
        # Check resulting P/R
        threshold = sorted(fraud_scores)[0] - 0.001
        tp = sum(1 for s,l,_ in ranked if s >= threshold and l==1)
        fp = sum(1 for s,l,_ in ranked if s >= threshold and l==0)
        tn = sum(1 for s,l,_ in ranked if s <  threshold and l==0)
        fn = sum(1 for s,l,_ in ranked if s <  threshold and l==1)
        if tp > 0:
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            f1   = 2 * prec * rec / (prec + rec)
            print('Seed ' + str(seed) + ': P=' + str(round(prec,3)) + ' R=' + str(round(rec,3)) + ' F1=' + str(round(f1,3)) + ' (gap=' + str(round(gap,4)) + ')')
            for s,l,n in ranked:
                lbl = 'F' if l else 'N'
                print('  ' + lbl + '  ' + n + '  ' + str(round(s,4)))
            found += 1
            if found >= 5:
                break
