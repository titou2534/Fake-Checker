"""Make sure we're running on a Python that has Fake Checker's packages.

Import this first in every entry script. If the current Python (e.g. the macOS
system one VS Code sometimes picks) lacks the packages, re-run the same command
with a Python that has them, such as Anaconda's. Kept Python 3.6+ compatible so
it can run under old interpreters.
"""
import importlib.util
import os
import subprocess
import sys

NEEDED = ["pandas", "sklearn", "joblib", "requests", "streamlit"]
CANDIDATES = [
    "/opt/anaconda3/bin/python", "/opt/miniconda3/bin/python",
    os.path.expanduser("~/anaconda3/bin/python"), os.path.expanduser("~/miniconda3/bin/python"),
    "/opt/homebrew/bin/python3", "/usr/local/bin/python3",
]
CHECK = "import sys; assert sys.version_info >= (3, 10); import " + ", ".join(NEEDED)


def _ok_here():
    return sys.version_info >= (3, 10) and all(importlib.util.find_spec(m) for m in NEEDED)


def _find_python():
    for py in CANDIDATES:
        if os.path.exists(py) and os.path.realpath(py) != os.path.realpath(sys.executable):
            if subprocess.run([py, "-c", CHECK], capture_output=True).returncode == 0:
                return py
    return None


if not _ok_here() and not os.environ.get("FAKECHECK_RELAUNCHED"):
    py = _find_python()
    if py is None:
        sys.exit("This Python ({}) is missing Fake Checker's packages and no Anaconda/other Python with them "
                 "was found.\nRun: pip install -r requirements.txt".format(sys.executable))
    print("[fakecheck] {} lacks the packages - switching to {}".format(sys.executable, py), flush=True)
    env = dict(os.environ, FAKECHECK_RELAUNCHED="1")
    try:
        code = subprocess.run([py] + sys.argv, env=env).returncode
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)
