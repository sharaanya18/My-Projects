"""Phase 7: baseline ladder. Honest (group-disjoint + shift) evaluation only."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, '../nlp_knowledge/code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from data import load, num_cols, CAT_COLS
from features import build, cat_dtypes
from validate import (template_clusters, adversarial_p_test, evaluate,
                      resampled_ndcg, oof_grouped, TEST_N)
from metric import ndcg_at_k
from experiment_log import ExperimentLog

TR, TE, RAWTXT, RAWTXT_TE = load(with_text=True); y = TR.target.values
X, XT = build(TR, TE)
groups = template_clusters(TR, 100)
p_test = adversarial_p_test(X, XT)
np.save('working/groups.npy', groups); np.save('working/p_test.npy', p_test)
log = ExperimentLog('working/EXPERIMENT_LOG.jsonl', artifact_dir='working/artifacts')

print(f"X={X.shape}  groups={len(set(groups))}  p_test mean={p_test.mean():.3f}")
print("\n=== FLOORS ===")
rng = np.random.default_rng(0)
rm, rs = resampled_ndcg(y, rng.random(len(y)))
print(f"  {'R00 random':38s} GRP {rm:.4f}+/-{rs:.4f}")
log.record(exp_id='R00', model='random', features='-', cv='resample586',
           metric='ndcg@20', score=rm, score_std=rs, decision='floor')
cm, cs = resampled_ndcg(y, np.zeros(len(y)))
print(f"  {'R01 constant':38s} GRP {cm:.4f}+/-{cs:.4f}  (all ties -> stable sort = id order)")
log.record(exp_id='R01', model='constant', features='-', cv='resample586',
           metric='ndcg@20', score=cm, score_std=cs, decision='floor')

print("\n=== RUNG 1: single causal features (LightGBM, 1 feature) ===")
for c in ['test_file_count', 'docs_file_count', 'body_code_block_count',
          'changed_lines', 'title_digit_count']:
    m = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=7,
                          min_child_samples=40, verbose=-1, random_state=0)
    evaluate(m, X[[c]], y, groups, p_test, name=f'1feat: {c}', draws=300)

print("\n=== RUNG 2: TF-IDF on raw profile_text (is the 'text' view worth anything?) ===")
class TextRanker:
    def __init__(self, analyzer='word', ngram=(1,1), C=1.0):
        self.v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, sublinear_tf=True,
                                 token_pattern=r'\S+', min_df=2)
        self.C = C
    def fit(self, idx, yy):
        Xt = self.v.fit_transform(RAWTXT[idx])
        self.m = LogisticRegression(max_iter=2000, C=self.C).fit(Xt, yy)
        self.classes_ = self.m.classes_
        return self
    def predict_proba(self, idx):
        return self.m.predict_proba(self.v.transform(RAWTXT[idx]))
idxdf = pd.DataFrame({'i': np.arange(len(y))})
for name, kw in [('tfidf word 1g', dict()), ('tfidf word 1-2g', dict(ngram=(1,2)))]:
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(5).split(idxdf, y, groups=groups):
        mm = TextRanker(**kw).fit(tr, y[tr])
        P = mm.predict_proba(va)
        oof[va] = P @ np.array([2.0**float(c)-1 for c in mm.classes_])
    gm, gs = resampled_ndcg(y, oof, draws=300)
    print(f"  {name:38s} GRP {gm:.4f}+/-{gs:.4f} (sp {spearmanr(oof,y).statistic:+.3f})")
    log.record(exp_id=f'T_{name}', model='tfidf+logreg', features=name, cv='GroupKFold(100clust)',
               metric='ndcg@20', score=gm, score_std=gs, decision='see report', oof=oof)

print("\n=== RUNG 3-4: LightGBM regression, leaky vs honest ===")
from sklearn.model_selection import StratifiedKFold, cross_val_predict
mreg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15,
                         min_child_samples=20, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, verbose=-1, random_state=0)
oof_leaky = cross_val_predict(mreg, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
lm, ls = resampled_ndcg(y, oof_leaky, draws=400)
print(f"  {'LGB reg, random KFold (LEAKY)':38s} GRP {lm:.4f}+/-{ls:.4f} (sp {spearmanr(oof_leaky,y).statistic:+.3f})")
log.record(exp_id='E03_leaky', model='LGBMRegressor', features='raw+derived+cat',
           cv='StratifiedKFold(5) LEAKY', metric='ndcg@20', score=lm, score_std=ls,
           decision='REJECT as evidence -- project leakage')
r = evaluate(mreg, X, y, groups, p_test, name='LGB reg, honest', draws=400)
log.record(exp_id='E04_honest', model='LGBMRegressor', features='raw+derived+cat',
           cv='GroupKFold(100clust)+shift', metric='ndcg@20', score=r['grp_ndcg'],
           score_std=r['grp_std'], diff_from='E03_leaky',
           error_notes=f"shift view {r['shift_ndcg']:.4f}+/-{r['shift_std']:.4f}",
           decision='HONEST REFERENCE', oof=r['oof'])
np.save('working/oof_e04.npy', r['oof'])
log.write_markdown('working/EXPERIMENT_LOG.md')
print("\nlogged ->", 'working/EXPERIMENT_LOG.jsonl')
