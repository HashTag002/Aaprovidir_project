#!/usr/bin/env python3
"""
Script de démarrage — Lance Dash (port 8050) et Django (port 8000) en parallèle.
Usage : python run.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  PESTEL ANALYTICS — Démarrage des serveurs")
print("=" * 60)
print()
print("  ▶ Dash Dashboard  → http://localhost:8050")
print("  ▶ Django App      → http://localhost:8000/dashboard/")
print()
print("  Ctrl+C pour arrêter")
print("=" * 60)

dash_proc = subprocess.Popen(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0,'.'); "
     "import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','pestel_project.settings'); "
     "import django; django.setup(); "
     "from dashboard.dash_app import app; app.run(debug=False, port=8050, host='0.0.0.0')"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)

django_proc = subprocess.Popen(
    [sys.executable, "manage.py", "runserver", "0.0.0.0:8000"],
)

try:
    dash_proc.wait()
    django_proc.wait()
except KeyboardInterrupt:
    dash_proc.terminate()
    django_proc.terminate()
    print("\n  Serveurs arrêtés.")
