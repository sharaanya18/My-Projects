# Prompt to continue in a new session

Copy everything below the line into a new Claude Code session. Attach the dataset in the same
message: either the split `.partN.rar` / `.7z.00N` files of the `public` folder (with `train/images`
and `test/images`), or say where the folder is if the session runs on your own computer.

---

I'm solving the Shipd "Project Eris" challenge **Filament Instances Across Scales**: instance
segmentation of thin filaments. It's scored by the mean over IoU thresholds 0.50–0.95 of the
micro-F1 of greedy one-to-one mask matching. Training labels come from domain A (filaments ~17 px wide)
and domain B (~4 px wide); the 452 test crops come from an unlabelled domain C (~6 px wide).
Runtime is `python3 solution.py <public_dir> <submission_out>` on one A10G, with about 1 hour for
training plus inference. My goal is a score **above 0.26 on unseen data** without overfitting or gaming the metric.

**Work so far**, on branch `claude/gifted-goodall-94aiwg` of `sharaanya18/My-Projects`, folder `filament-instances/`:
- `solution.py`: the complete pipeline.
  - A U-Net from scratch with filament and junction outputs.
  - Training crops are rescaled to a 4.3–8.5 px filament width, and the masks are re-rasterised from
    the polygons with an exact copy of the task's rasteriser.
  - Touching and crossing filaments are split using the junction output.
  - The thresholds, fragment filter and test-time scale set are searched with the official metric on
    held-out objects rendered at 5–7.5 px (a stand-in for domain C).
  - Prediction is at native size with 8-way flip/rotation TTA, applied one image at a time.
  - Wall-clock guards: training stops at 45 min (`FIL_TRAIN_END`), the search at 52.5 min (`FIL_HPO_END`).
- `APPROACH.md`: the full reasoning, including the rules followed.
- `tools/check_labels.py <public_dir>`: checks against the real labels (needs only the CSVs). Results so far:
  - rasteriser: 0 mismatches out of 1,582 published masks;
  - median filament width: A 16.6 px, B 4.0 px;
  - crops with crossing / touching filaments: 81 / 114;
  - filament splitting with perfect maps scores 0.951, and 0.460 with maps 1 px too fat.
- **Not done yet:** no training run on the real images, so there is no measured score.

**Rules I must follow** (the Solver Guidebook overrides the challenge text where they differ):
- Train from scratch inside the script. No pretrained weights, no external or synthetic training data.
- The test crops are for per-image inference only. No pseudo-labels, self-training, test-time
  adaptation or test-set statistics, even though the challenge text allows self-training.
- Nothing tuned offline may be hard-coded: every search runs inside the script.
- Only use Kaggle-Docker libraries (torch, opencv, scikit-image, scipy, pandas, numpy).
- Stop training around 45–50 min, and the whole run must finish within about 1 hour.
- Don't use ids, `capture_group` or file order as features.
- `submission.csv` must contain all 452 test ids.

**Please do this:**
1. Get the branch: `git fetch origin claude/gifted-goodall-94aiwg && git checkout claude/gifted-goodall-94aiwg`.
2. Unpack the dataset I attached (e.g. `7z x "public (1).part1.rar"` or `7z x public.7z.001`).
   Check that `train/images` has 921 JPEGs and `test/images` has 452, and that the CSVs are present.
3. Run `python3 tools/check_labels.py <public_dir>` and confirm the numbers above.
4. Run the solution:
   - **With a GPU:** run it in full (`python3 solution.py <public_dir> working/submission.csv`) and
     report the logged proxy score, the A-proxy and B-proxy scores, the chosen configuration and the
     runtime.
   - **CPU only:** first run a short smoke test (e.g. `FIL_TRAIN_END=600 FIL_HPO_END=900`) to confirm
     it works on the real images. Then run as long a training as is reasonable, and tell me plainly that
     the CPU proxy score understates what the one-hour GPU run achieves.
5. Look at the log and improve only in ways that generalise, re-checking the proxy score after each
   change. Things to check:
   - Is the training throughput enough, or is the data loader the bottleneck?
   - Is the chosen `t_m` at the edge of the search grid?
   - Are the A-proxy and B-proxy scores far apart?
   - Are there too many false strands, or fragments?

   Don't tune to the leaderboard, and don't hard-code parameters found offline.
6. Commit and push to the same branch. Report honestly: give the measured proxy score, say whether it
   is likely to clear 0.26, and state that the real unseen score comes only from the platform
   (upload the `submission.csv` as the free CSV check).
