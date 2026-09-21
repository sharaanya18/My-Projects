"""Phase 11: error analysis on the top-20 region, which is all the metric sees."""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from data import load, CAT_COLS, num_cols
from features import build
from validate import resampled_ndcg
TR,TE=load(); y=TR.target.values
folds=np.load('working/folds_cs.npy')
oofs=np.load('working/oofs_objectives.npz')
pred=oofs['reg_gain(2^r-1)'] if 'reg_gain(2^r-1)' in oofs.files else oofs[oofs.files[1]]

print("=== WHO LANDS IN THE TOP 20? (repeated 586-row draws) ===")
rng=np.random.default_rng(0)
top_labels=[]; top_rows=[]
for _ in range(500):
    i=rng.choice(len(y),586,replace=False)
    o=i[np.argsort(-pred[i],kind='mergesort')[:20]]
    top_labels.append(y[o]); top_rows.extend(o.tolist())
top_labels=np.array(top_labels)
print("  mean composition of the top 20:",
      {f"label {k}": round(float((top_labels==k).sum()/top_labels.size),3) for k in [0,1,2]})
print("  base rate in train        :",
      {f"label {k}": round(float((y==k).mean()),3) for k in [0,1,2]})

cnt=pd.Series(top_rows).value_counts()
freq=cnt.head(200).index.values
print(f"\n  rows appearing in a top-20 most often: {len(cnt)} distinct rows ever reached top-20")
sub=TR.iloc[freq]
print("  label mix among the 200 most-promoted rows:", sub.target.value_counts().sort_index().to_dict())

print("\n=== THE COSTLY MISTAKES: label-0 rows promoted into the top 20 ===")
false_top=[r for r in freq if y[r]==0][:60]
if false_top:
    F=TR.iloc[false_top]; R=TR
    print(f"  n={len(false_top)}  (these are pure wasted slots -- gain 0)")
    rows=[]
    for c in num_cols(TR):
        rows.append((c, F[c].median(), R[c].median(), F[c].mean(), R[c].mean()))
    D=pd.DataFrame(rows,columns=['col','fp_med','all_med','fp_mean','all_mean'])
    D['ratio']=D.fp_mean/(D.all_mean+1e-9)
    print(D.reindex(D.ratio.sub(1).abs().sort_values(ascending=False).index).head(10).to_string(index=False,float_format=lambda x:f"{x:.3g}"))
    print("\n  categorical mix of promoted label-0 rows vs overall:")
    for c in ['body_structure','change_scope','base_channel','test_term','docs_term']:
        a=F[c].value_counts(normalize=True).round(2).to_dict(); b=TR[c].value_counts(normalize=True).round(2).to_dict()
        print(f"    {c:16s} promoted0={a}")
        print(f"    {'':16s} overall  ={b}")

print("\n=== MISSED ACTIVES: label-2 rows the model ranks lowest ===")
order=np.argsort(pred)
missed=[r for r in order[:400] if y[r]==2][:60]
print(f"  n={len(missed)} label-2 rows in the bottom 400 of the OOF ranking")
if missed:
    M=TR.iloc[missed]
    for c in ['body_structure','change_scope','base_channel','test_term','docs_term','body_length','file_count','changed_lines']:
        if c in CAT_COLS:
            print(f"    {c:16s} missed={M[c].value_counts(normalize=True).round(2).to_dict()}")
        else:
            print(f"    {c:16s} missed_med={M[c].median():.0f}  all_med={TR[c].median():.0f}")

print("\n=== CALIBRATION BY PREDICTION DECILE ===")
q=pd.qcut(pred,10,labels=False,duplicates='drop')
print(pd.DataFrame({'decile':q,'target':y}).groupby('decile').target.agg(['mean','size']).to_string())
