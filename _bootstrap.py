import os
import sys


def ensure_venv():
    root = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(root, ".venv")

    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(venv_python):
        activate_hint = (
            r"  .venv\Scripts\activate"
            if os.name == "nt"
            else "  source .venv/bin/activate"
        )
        sys.exit(
            "No .venv found next to this script.\n"
            "Run the install steps from the README first:\n"
            "  python -m venv .venv\n"
            f"{activate_hint}\n"
            "  pip install -r requirements.txt"
        )

    if os.path.abspath(sys.executable) != os.path.abspath(venv_python):
        if os.name == "nt":
            import subprocess
            result = subprocess.run([venv_python] + sys.argv)
            sys.exit(result.returncode)
        else:
            os.execv(venv_python, [venv_python] + sys.argv)
