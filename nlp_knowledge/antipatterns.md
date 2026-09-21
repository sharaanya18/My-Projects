# Anti-Patterns

Every item here was **found in the code the PDF recommends** (`[VERIFIED]` =
I read the file this session). They are recorded because the PDF presents these
repositories as models to follow, and several of them contain patterns that will
cost you a leaderboard position.

---

## 1. Reporting a single 80/20 split as a model comparison

**Where:** `Snigdho8869/Multiclass Text Classification` `[VERIFIED]`.

13 models compared on one `train_test_split(test_size=0.2, random_state=42)` of
2,225 BBC documents — ~445 test rows, so one document ≈ 0.22 accuracy points.
The reported gap between LinearSVC (96.40) and MultinomialNB (96.18) is **one
document**.

**Why it's wrong:** the measurement cannot resolve the differences being
reported, and picking the max over 13 models evaluated on the same split biases
the winner upward.

**Do instead:** `StratifiedKFold`, fixed folds reused across experiments, report
mean ± per-fold std, and treat any gap smaller than the std as no gap.

---

## 2. Evaluating on the split used for early stopping

**Where:** same repo `[VERIFIED]` — `learner.validate(val_data=test_data)` where
`test_data` was passed as `val_data` during `fit_onecycle`. The 98.20% BERT and
97.75% XLNet numbers come from this.

**Why it's wrong:** the split guided training; the number is optimistic by
construction and not comparable to the classical models' numbers.

**Do instead:** three-way split, or nested CV. Early-stop on an inner split;
report on an outer one.

---

## 3. Grid-searching on train, then reporting CV on the same train

**Where:** same repo `[VERIFIED]` — `GridSearchCV(cv=5).fit(x_train_tfidf)` and
then `cross_val_score(best_model, x_train_tfidf, y_train, cv=10)`.

**Why it's wrong:** the hyperparameters were selected using those very rows. The
CV score is post-selection and optimistic.

**Do instead:** nested CV, or select on an inner fold and report on the outer.
At minimum, state that the number is post-selection.

---

## 4. Ensembling correlated models with hand-picked weights

**Where:** same repo `[VERIFIED]`:
`VotingClassifier([...], voting='soft', weights=[1,2,3,4])`.

Four models on the **same** TF-IDF matrix, arbitrary weights, result **96.40%**
— exactly the best single member's score. Zero gain.

**Why it's wrong:** ensembling requires **complementary errors**, and weights
require fitting. Neither condition was checked.

**Do instead:** measure OOF prediction correlation first; fit weights on OOF;
compare the blend against its best single member on the same folds.
See `techniques/ensembling.md`.

---

## 5. Silently swapping the model inside the ensemble

**Where:** same repo `[VERIFIED]`. The headline LinearSVC score uses
`LinearSVC()`. The ensemble uses `SVC(probability=True)` — a **kernel** SVM,
O(n²), a different model — because `voting='soft'` requires `predict_proba`,
which `LinearSVC` doesn't have.

**Why it's wrong:** the ensemble is not "the reported models combined."

**Do instead:** wrap `LinearSVC` in `CalibratedClassifierCV` if you need
probabilities, and say so.

---

## 6. `argmax` over a single-logit output

**Where:** `TechNBusiness/week8_text_classification.ipynb` `[VERIFIED]`.

```python
model.add(tf.keras.layers.Dense(1))         # single logit
...
results_pred = model.predict(test_examples)  # shape (n, 1)
classes_x = np.argmax(results_pred, axis=1)  # ALWAYS 0
```

**Why it's wrong:** `argmax` over a length-1 axis is always 0. The prediction
path silently returns all-negative.

**Do instead:** for a single logit, `(sigmoid(logit) > threshold)`. And
**always assert on the shape and class balance of your prediction vector before
submitting** — an all-one-class submission is the classic silent disaster.

---

## 7. Training 40 epochs with no early stopping

**Where:** same notebook `[VERIFIED]` — `epochs=40`, no `EarlyStopping`, no
checkpoint restore.

**Why it's wrong:** you keep whatever epoch 40 produced, not the best epoch.

**Do instead:** early stopping on the eval metric, restore the best checkpoint.

---

## 8. Retrieval framed as classification over document IDs

**Where:** `justin-aj/ml-retrieval` `[VERIFIED]` — `RandomForestClassifier`
trained on 10 documents with `labels = list(range(len(docs)))`: **one training
example per class**, nothing held out, top probability 0.150 over 10 classes
(uniform prior = 0.100).

**Why it's wrong:** the label space is the corpus, so it cannot generalize to
new documents; and with one example per class there is nothing to learn or
validate. The repo's headline ("both methods agree on the top result") rests on
**one query**.

**Do instead:** score query-document **pairs**. Evaluate on a labeled query set
with Recall@k / MRR / nDCG. If a Shipd task tempts you into per-document
classes, treat that as a signal you've mis-framed it.

---

## 9. Using a deprecated embedding model

**Where:** `SimonLaub/NLP_JobTrend` `[VERIFIED]` —
`SentenceTransformer('bert-base-nli-mean-tokens')`, a model SBERT itself
deprecated for producing low-quality embeddings.

**Do instead:** any current MiniLM / mpnet / BGE / E5 model. Check the model
card before adopting a model name you found in a tutorial.

---

## 10. Wrong-language tokenizer

**Where:** same repo `[VERIFIED]` — `jieba.lcut()` (a **Chinese** word
segmenter) applied to Danish and English text.

**Do instead:** match the tokenizer to the language. And run a language-ID check
over your fields as a standing data-audit step.

---

## 11. Monolingual encoder on multilingual text

**Where:** same repo `[VERIFIED]`. The author's own note is the lesson:
*"In danish this is not as good as the model is not trained on danish texts."*

**Why it's dangerous:** it fails **silently**. Embeddings still come out,
similarities still rank, the scores just mean much less.

**Do instead:** detect the language of every text field; use a multilingual
encoder when any non-target language is present.

---

## 12. Installing a dead framework

**Where:** `allenai/allennlp` `[VERIFIED]` — maintenance mode since 2022, last
commit 2022-11-21, cited by the PDF for "Retrieval-based QA" (which lived in a
different repo anyway).

**Do instead:** check the last commit date and any maintenance banner **before**
adopting a dependency. Use `transformers` + `sentence-transformers` + `datasets`.

---

## 13. Training a transformer from scratch on competition data

**Where:** `graykode/nlp-tutorial` `[VERIFIED]` — excellent <100-line
Transformer/BERT implementations, on toy corpora.

**Why it's wrong for competitions:** without pretraining, these lose badly to
TF-IDF at far greater cost. Measured in the same PDF's own sources: on BBC,
from-scratch Keras LSTM/GRU/CNN scored **91–95%** versus TF-IDF + LinearSVC's
**96.4%** `[VERIFIED]`.

**Do instead:** read them to understand attention. Use pretrained weights to
compete.

---

## 14. Internally inconsistent environment instructions

**Where:** `HKUNLP/instructor-embedding` `[VERIFIED]` — README says
`conda env create -n instructor python=3.7`, while `requirements.txt` demands
`torch>=2.0`, `sentence-transformers>=3.0.1,<4.0`, `numpy<=1.26.4`,
`pyarrow>=17,<18`.

**Why it matters:** Python 3.7 cannot satisfy those pins, and the
`sentence-transformers<4.0` pin conflicts with the v6 API documented elsewhere
in this KB.

**Do instead:** resolve dependency conflicts explicitly and pin your own
environment. Never trust a README's environment section over its
`requirements.txt` — check both, and test the import.

---

## Cross-cutting: the meta-anti-pattern

Nine of the fourteen above are **evaluation** errors, not modeling errors.
That ratio is the real lesson of reading this resource list: the cited work is
generally competent at fitting models and generally careless about measuring
them. Competitions are won by the measurement.
