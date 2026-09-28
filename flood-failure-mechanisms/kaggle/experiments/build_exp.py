# Builds one Kaggle script: data prelude + a driver that runs the backbone sweep
# on GPU 0 and a solution.py variant on GPU 1, in parallel.
import base64, re, sys
here = sys.argv[1]; out = sys.argv[2]; overrides = dict(a.split("=", 1) for a in sys.argv[3:])
prelude = open(f"{here}/../kaggle_prelude.py").read()
sweep = open(f"{here}/sweep_backbones.py").read()
sol = open(f"{here}/../../solution.py").read()
for k, v in overrides.items():
    sol, nsub = re.subn(rf"^{k} = .*$", f"{k} = {v}", sol, count=1, flags=re.M)
    assert nsub == 1, k
enc = lambda s: base64.b64encode(s.encode()).decode()
driver = f'''
import base64, subprocess, os
_d = "/kaggle/working"
open(f"{{_d}}/sweep.py","w").write("import sys as _sys\\n" + base64.b64decode("{enc(sweep)}").decode())
open(f"{{_d}}/sol.py","w").write(base64.b64decode("{enc(sol)}").decode())
env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0", SWEEP_DEVICE="cuda:0")
env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
p0 = subprocess.Popen(["python", f"{{_d}}/sweep.py", _sys.argv[1]], env=env0, stdout=open(f"{{_d}}/sweep.log","w"), stderr=subprocess.STDOUT)
p1 = subprocess.Popen(["python", f"{{_d}}/sol.py", _sys.argv[1], f"{{_d}}/submission_v2.csv"], env=env1, stdout=open(f"{{_d}}/sol_v2.log","w"), stderr=subprocess.STDOUT)
print("sweep exit", p0.wait(), "| sol exit", p1.wait())
for f in ["sweep.log", "sol_v2.log"]:
    print("=" * 30, f); print(open(f"{{_d}}/{{f}}").read())
'''
open(out, "w").write(prelude + driver)
print("overrides:", overrides)
