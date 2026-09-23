# Sichuan Wavefront Witness Reconstruction — Top-5 Leaderboard Solution Notes

Companion to `notes.md` (Terraform Resource Complement Ranking). This is a
different competition: given 4 "witness" seismic stations' 256-sample
waveforms plus the lat/lon geometry of the witnesses and a "receiver"
station, reconstruct the 256-sample waveform that *would have been* recorded
at the receiver — i.e. spatial interpolation/extrapolation of a seismic
wavefield from nearby observations to an unobserved location, within the
same earthquake event.

## 0. The scoring function (shapes everything)

All five solutions reimplement the same metric, so it's worth stating once:

```
w = clip(1 - sum((pred-target)^2)/sum(target^2), 0, 1)   # signed reconstruction fit
a = clip(1 - sum((|pred|-|target|)^2)/sum(target^2), 0, 1) # magnitude/envelope fit
z = clip((sqrt(w*a) - floor) / (1 - floor), 0, 1)          # floor ~ 0.04–0.45/sqrt(n-1)
score = expm1(1.2*z) / expm1(1.2)
```

It's a **geometric mean of a signed-error term and a magnitude(envelope)-only
term**, then floor-clipped and convex-reshaped. Getting the *phase/shape*
right and getting the *amplitude* right are both necessary — you can't
compensate for one with the other, since it's a product inside the sqrt, not
a sum. Every solution's loss function and ensembling logic is visibly
shaped around this.

---

## 1. Shared structure across all 5 solutions

1. **Event/leakage grouping via byte-exact witness matching.** Rows aren't
   independent — the *same* witness recording (same 256 float32 samples)
   appears verbatim across multiple rows belonging to the same underlying
   earthquake event. Every solution recovers this event grouping by hashing
   each witness waveform (`blake2b`/`sha1`/`sha256` of `.tobytes()`) and
   union-find-clustering rows that share a hash, then splits
   train/validation **by event**, never by row — the direct analogue of
   problem 1's "module" leakage control.
2. **"Any station can be the target" symmetry.** Physically, the receiver
   and the 4 witnesses are the same kind of instrument recording the same
   event — which one is "withheld" is arbitrary. Four of the five solutions
   explicitly exploit this as free data augmentation: reassign a different
   pool member as the pseudo-target and treat the rest (or a nearest-by-
   distance subset) as pseudo-witnesses. This alone often multiplies the
   effective training set severalfold. (Rank 1 instead exploits it via
   leave-one-out statistics rather than resampled training rows.)
3. **Frequency-domain modeling dominates.** Nearly every model works
   primarily on `rfft` spectra rather than raw time-domain samples — wave
   attenuation/propagation and station-to-station transfer are naturally
   per-frequency-bin linear (or near-linear) operations, so several
   solutions solve small closed-form linear systems per frequency bin
   (classic **kriging** / **Wiener filtering** from geophysics/signal
   processing) rather than relying purely on learned convolutions.
4. **Metric-aware loss shaping.** Every training loss adds an explicit
   `MSE(|pred|, |target|)` term on top of plain MSE — directly targeting the
   scoring function's separate magnitude ("a") component rather than
   hoping plain reconstruction loss covers it implicitly.
5. **Odd-symmetry / sign robustness.** Since flipping the sign of all
   witness inputs should flip the sign of the true output (same physical
   event, opposite polarity convention), this is treated as a *known* prior:
   either hard-coded into the architecture (`f(x) = (raw(x) - raw(-x))/2`,
   ranks 1 and 3) or trained in via random polarity-flip augmentation
   (ranks 3, 5).
6. **Physics baseline + neural correction, calibrated against the true
   metric.** The stronger solutions (ranks 1, 2, 4) blend a closed-form
   statistical/physical predictor (kriging or Wiener filter, fit from
   training-event covariance/cross-spectra) with a neural residual
   corrector, then grid-search or numerically optimize (`minimize_scalar`)
   the blend weights and a global output-scale factor **directly against
   the competition score** on a held-out validation split — not against a
   proxy loss.
7. **Determinism engineering** (fixed seeds, deterministic cuDNN, disabled
   TF32, pinned thread counts) is universal, and rank 1 goes as far as
   asserting that re-running inference on the same input twice gives
   bit-identical output before writing the submission.

---

## 2. Solution-by-solution breakdown

### Rank 1 — closed-form frequency-domain kriging + two neural residual correctors
- `SpatialSpectrum`: builds a **per-frequency-bin spatial covariance matrix**
  across all unique station *locations* seen in training (not per-station-ID,
  per physical location), estimated from `rfft` spectra of training events,
  power-normalized (`power` exponent) and frequency-smoothed (`smooth`
  bandwidth via `gaussian_filter1d`). For each row, the nearest training
  location to the actual witness/receiver locations is substituted, so the
  model generalizes to previously-unseen exact coordinates.
- `.predict()`: solves a small **4×4 kriging (MMSE) linear system per
  frequency bin** (`cov[witnesses,witnesses]^-1 @ cov[witnesses,target]`,
  ridge-regularized by the diagonal) to get optimal linear combination
  weights of the 4 witness spectra, inverse-FFTs to get the physics-based
  prediction.
- `spatial_inputs()` computes a **leave-one-out** version of this predictor
  for training rows (subtracting the row's own event contribution from the
  covariance before solving) so training inputs to the neural stage never
  see their own answer baked into the statistics — leakage-safe stacking.
- Two neural residual correctors on top of the per-witness kriging outputs
  (`SymmetricCorrector`: dilated 1D convs, corrects the summed prediction;
  `PartsCorrector`/`UNetParts`: correct the per-witness kriging
  *contributions* individually before summing, `UNetParts` being a full
  encoder-decoder with skip connections). Both use the antisymmetric
  `(raw(x)-raw(-x))/2` trick and an EMA-averaged copy of weights for
  inference stability.
- Final blend: grid search over UNet-vs-Parts weight and a fallback
  "base-kriging" weight, with the combination scale calibrated by directly
  maximizing the real scoring function via `scipy.optimize.minimize_scalar`.
- Most rigorous leakage discipline and reproducibility verification of the
  five (explicit repeat-inference equality assertion).

### Rank 2 — FFT-domain PhaseNet + role-swap augmentation + dedicated magnitude network + statistical kernel stacking
- `PhaseNet`/`SitePhaseNet`: a neural net that computes **per-frequency
  complex mixing weights** for the 4 witness spectra (conditioned on
  pairwise cross-spectral coherence features and a learned, nearest-3-site
  weighted station embedding) — effectively a *learned* Wiener/kriging
  filter instead of a closed-form one.
- `receiver_augmentation`: randomly re-labels a different station as the
  target and re-sorts the remaining stations by distance to it, creating
  synthetic supervised rows from the same underlying event data — the
  "any station can be the target" trick implemented as row-level resampling.
  `refine()` continues training with this augmentation after an initial fit
  on real rows (curriculum: fit real, then refine with synthetic).
- Closed-form **Wiener-style kernel model** (`fit_balanced_kernel` /
  `predict_kernel`) as a second baseline, with `positive_kernel` **repairing**
  the estimated correlation matrix to be positive-semidefinite (eigen-clip
  negative/small eigenvalues) before inversion — a numerically-robust
  variant of rank 1's kriging.
- `oof_kernel`: an **out-of-fold** (4-fold, event-grouped) version of the
  kernel predictor computed on the training set itself, fed as an extra
  input channel into a second-stage stacked model (`fit_stack`) — leakage-
  safe model stacking.
- **`MagnitudeWaveNet`**: a wholly separate network trained only to predict
  `|target|` (envelope), decoupled from sign/shape prediction — a direct,
  explicit split matching the scoring metric's two independent terms.
  `magnitude_adjust` then blends this magnitude estimate (plus a smoothed
  witness-envelope average) into the final signed prediction while
  preserving its predicted sign.
- Most deliberately decomposed pipeline: separate models for phase/shape,
  magnitude, and a statistical prior, fused with fixed calibrated weights.

### Rank 3 — spectral-gain CNN + axial spatiotemporal transformer ensemble, pool-based augmentation
- Recovers full **event station pools** (every unique witness recording
  within an event, via hash-based union-find) and explicitly enumerates
  **every pool member as a potential pseudo-target**, paired with a
  nearest-by-distance selection of the rest of the pool as witnesses — the
  most exhaustive version of the "any station can be the target"
  augmentation among the five solutions.
- `SpectralWavefrontNet`: convolves each witness spectrum (first 48
  frequency bins) with station-embedding- and geometry-conditioned FiLM
  layers, plus **cross-correlation lag features** (sub-sample parabolic
  interpolation of the cross-spectrum peak between witness pairs) as
  additional input, to predict per-witness complex spectral "gain" filters
  that are summed and inverse-FFT'd, then refined by a time-domain residual
  conv stage.
- `AxialWavefrontTransformer`: treats the input as a (witness × time) grid
  and applies **axial attention** — attention across time within each
  witness channel (rotary positional embeddings) alternating with attention
  *across* the 4 witness channels — with adaptive FiLM normalization
  conditioned on array/receiver geometry. The most architecturally novel
  model of the five (a small custom transformer purpose-built for 4-witness
  slates), predicting via a final learned spectral-gain head.
- Stochastic resampling of which witnesses (by distance rank) are used per
  training step, sign-flip and small time-shift augmentation, and weight-EMA
  averaging for the transformer.
- Final prediction: fixed 0.75/0.25 weighted average of 2-seed spectral CNN
  and 2-seed axial transformer — no closed-form statistical baseline
  component, purely neural. Ranked 3rd despite being the most
  architecturally sophisticated — same "complexity doesn't automatically
  win" pattern seen in problem 1.

### Rank 4 — attention-pooled residual encoder/decoder + closed-form Wiener filter, simplest neural architecture
- `StationNet`: encodes the 4 witness waveforms independently via a small
  dilated residual CNN (FiLM-conditioned on station embeddings), computes a
  **softmax attention weight per witness** from geometry (relative offset,
  distance, target-station embedding) and a "Gram matrix" of pairwise
  normalized-waveform cross-correlation (a lightweight learned similarity
  kernel among witnesses), uses these weights both to pool the encoded
  hidden features *and* to form a weighted-average "anchor" waveform that
  the decoder's output is added onto as a residual.
- Bakes in the "any station can be target" trick as an **auxiliary loss
  term** rather than a separate augmented dataset: each training step also
  runs the same network with one random witness relabeled as the pseudo-
  target and adds `0.25 * MSE(prediction, that witness's real waveform)` to
  the loss — cheaper to implement than ranks 2/3's row-resampling, same
  self-supervised effect.
- Closed-form **Wiener filter** baseline (`wiener_statistics` /
  `wiener_parameters` / `wiener_predict`): per-station-pair cross/auto power
  spectral density estimated across training events (with NaN-masking for
  missing station-event pairs), ridge-regularized transfer function, and a
  **coherence-weighted combination** across witnesses (each witness weighted
  by its estimated reliability, `coherence^exponent`) — simpler than ranks
  1/2's full covariance-matrix inversion (treats witness contributions
  independently rather than jointly).
- `amplitude_envelope`: smoothed median-envelope magnitude estimator blended
  into the final prediction the same way as rank 2's magnitude network, but
  via a cheap statistical estimate rather than a trained network.
- All blend weights, Wiener ridge/exponent, envelope width/weight, and
  global scale are grid-searched **directly against the real scoring
  function** on held-out validation — same calibration discipline as ranks
  1 and 2.
- Simplest neural component of the top 4 (no transformer, no explicit
  spatial-covariance kriging network) — writes an immediate trivial
  "mean-of-witnesses" fallback submission before any training even starts,
  as a defensive measure against crashes/timeouts.

### Rank 5 — large seed/architecture-diverse ensemble of one FiLM-conditioned WaveNet, heavy combinatorial augmentation, no statistical baseline
- `WaveNet`: dilated-conv encoder applied per-witness (batched as `B*4`),
  with a cheap **cross-witness mixing** mechanism (mean-pool hidden state
  across the 4 witnesses, injected back via a 1×1 conv at every encoder
  block — far lighter than rank 3's full axial attention), geometry-only
  attention pooling across witnesses, and a decoder ("trunk") producing the
  output.
- Combines a **fixed near-identity linear FIR filter** per witness
  (initialized to ≈0.195, i.e. close to a simple 4-way average) as a robust
  linear baseline term, with a **geometry-conditioned adaptive FIR filter**
  (`tapnet` predicts convolution taps from geometry, applied via
  `unfold`+`einsum`) added as a residual — a learned, end-to-end analogue of
  rank 1's "closed-form kriging + neural correction" idea, but with both
  pieces trained jointly by gradient descent instead of one being
  closed-form.
- `event_pools` / `extra_cases`: the same event-recovery-via-hashing +
  "any pool member can be target" trick as ranks 2/3, but expanded
  combinatorially across **10 predefined distance-rank subset patterns**
  (`RANKSETS`) per target, generating a much larger synthetic training set
  than the other solutions' augmentation.
- Largest ensemble: up to **8 members** varying not just seed but channel
  width, encoder depth, trunk depth, and kernel size (true architecture
  diversity, not just seed-bagging), simple-averaged.
- No closed-form statistical/kriging/Wiener baseline at all — relies
  entirely on the neural ensemble (plus the linear-FIR inductive bias baked
  into initialization). Writes and incrementally overwrites a fallback/
  in-progress submission after every ensemble member finishes, prioritizing
  "always have a valid submission on disk" under time pressure.
- Ranked last despite the largest ensemble — same pattern as problem 1's
  rank 5: brute-force ensembling of a single architecture, without a
  physically-motivated closed-form component or explicit metric-decomposed
  sub-model, underperforms solutions that combine domain-aware statistical
  modeling with neural correction and calibrate directly against the score.

---

## 3. Cross-cutting takeaways

1. **Event-level leakage control via exact-waveform hashing is
   non-negotiable** — every solution independently reimplements
   hash-and-union-find grouping before any train/validation split, because
   the same witness recording appears verbatim across multiple rows.
2. **"Any station can be the target" is the single most reused structural
   insight** — four of five solutions turn this physical symmetry into
   substantial free data augmentation (either as resampled training rows or
   as an auxiliary loss term), and it's arguably the biggest lever available
   given how few real training events there are.
3. **The scoring metric is explicitly decomposed into "shape" and
   "magnitude" and both are targeted directly** — via an extra
   `MSE(|pred|,|target|)` loss term (universal), a dedicated magnitude-only
   network (rank 2), or a statistical envelope estimator blended in
   post-hoc (ranks 2, 4). Nobody relies on plain MSE alone to cover both.
4. **Closed-form, physically-motivated baselines (kriging / Wiener
   filtering in the frequency domain) plus a learned neural residual
   correction consistently beat pure end-to-end neural approaches** in this
   leaderboard — ranks 1, 2, and 4 (the top 3) all include such a baseline;
   ranks 3 and 5 (purely neural) placed lower despite one having a novel
   custom transformer and the other the largest ensemble. This echoes
   problem 1's finding that domain-aware structure beats raw model
   capacity.
5. **Calibrating ensemble weights and global scale directly against the
   true competition score** (grid search or numerical optimization on a
   held-out, event-grouped validation split) rather than trusting training
   loss is standard practice among the top solutions — worth doing whenever
   the final metric isn't the same as the training loss.
6. **Physical priors reduce what the network has to learn from scratch**:
   odd-symmetry-in-sign (baked in architecturally or via augmentation),
   near-identity/averaging initialization for output layers (`zeros_` on
   final conv weights so the network starts as a no-op or as a simple
   average), and geometry-based (not just learned) attention weighting all
   appear repeatedly — these act as strong, nearly-free regularizers on a
   small-data problem.
7. **Determinism and defensive engineering are treated as first-class
   requirements**: fixed seeds and deterministic backend flags everywhere,
   an explicit repeat-inference equality check (rank 1), and immediate
   fallback-submission writes before/during training (ranks 4, 5) so a
   timeout or crash still leaves a valid, if weak, submission on disk.

---

## 4. If reusing these ideas elsewhere

- For any "reconstruct a signal at an unobserved sensor from nearby sensors"
  problem: start with a closed-form per-frequency-bin kriging or Wiener
  filter fit from cross-sensor covariance/coherence as a strong, cheap
  baseline, then add a neural residual corrector on top rather than
  replacing it outright.
- If the sensors/roles are physically interchangeable, exploit that
  symmetry for data augmentation (row resampling or an auxiliary
  "predict-a-witness" loss term) before reaching for a bigger model — it
  was the most consistently high-value trick across solutions here.
- When the competition metric has multiple weighted/multiplied components
  (here: shape and magnitude), add a matching loss term for each component
  directly, and consider a dedicated sub-model for the component that's
  hardest to get from a single joint objective.
- Calibrate any blend weights and global output scale against the actual
  metric on held-out data — don't assume equal-weight averaging or raw
  training loss minimization gets you to the true optimum.
