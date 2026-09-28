# Kaggle launcher prelude. It finds the uploaded dataset under /kaggle/input and
# lays it out as <data>/{*.csv, images/} so solution.py can run unchanged. The
# solution source follows this block and reads positional args.
import glob as _glob, os as _os, sys as _sys
_csv = sorted(_glob.glob("/kaggle/input/**/train.csv", recursive=True))
assert _csv, "train.csv not found under /kaggle/input"
_src = _os.path.dirname(_csv[0])
_jpgs = _glob.glob("/kaggle/input/**/*.jpg", recursive=True)
assert _jpgs, "no images found under /kaggle/input"
_img_dir = _os.path.dirname(_jpgs[0])
_data = "/kaggle/working/data"
_os.makedirs(_data, exist_ok=True)
for _f in ["train.csv", "train_targets.csv", "test.csv", "sample_submission.csv"]:
    if not _os.path.exists(f"{_data}/{_f}"):
        _os.symlink(f"{_src}/{_f}", f"{_data}/{_f}")
if not _os.path.exists(f"{_data}/images"):
    _os.symlink(_img_dir, f"{_data}/images")
print(f"data: {_src} | images: {_img_dir} ({len(_jpgs)} jpgs)", flush=True)
_sys.argv = [_sys.argv[0], _data, "/kaggle/working/submission.csv"]
# ---------------------------------------------------------------------------
