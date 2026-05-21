#!/usr/bin/env python3
"""
Script de démarrage — Lance Dash (port 8050) et Django (port 8000) en parallèle.
Libère automatiquement les ports s'ils sont occupés.
"""
import subprocess
import sys
import os
import time
import psutil
from pathlib import Path

# Définition du répertoire de base
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

def kill_process_on_port(port):
    """Tue agressivement tous les processus écoutant sur le port spécifié."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # On vérifie les connexions du processus
            connections = proc.connections(kind='inet')
            for conn in connections:
                if conn.laddr.port == port:
                    print(f"  [!] Libération forcée du port {port} (PID: {proc.info['pid']} - {proc.info['name']})")
                    proc.kill() # SIGKILL pour une libération immédiate
                    proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    time.sleep(1) # Pause de sécurité pour laisser l'OS libérer le socket

# Détection de l'interprète Python
venv_python = BASE_DIR / "env" / "bin" / "python3"
python_exe = str(venv_python) if venv_python.exists() else sys.executable

print("=" * 60)
print("  PESTEL ANALYTICS — Nettoyage et Démarrage")
print("=" * 60)

# Nettoyage des ports
kill_process_on_port(8000)
kill_process_on_port(8050)

print()
print(f"  ▶ Dash Dashboard  → http://localhost:8050")
print(f"  ▶ Django App      → http://localhost:8000/dashboard/")
print()
print("  Ctrl+C pour arrêter")
print("=" * 60)

# Commandes de lancement
dash_cmd = [
    python_exe, "-c",
    "import sys; sys.path.insert(0,'.'); "
    "import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','aaprovidir_project.settings'); "
    "import django; django.setup(); "
    "from dashboard.dash_app import app; app.run(debug=False, port=8050, host='0.0.0.0')"
]

django_cmd = [python_exe, "manage.py", "runserver", "0.0.0.0:8000"]

dash_proc = subprocess.Popen(dash_cmd)
django_proc = subprocess.Popen(django_cmd)

try:
    while True:
        time.sleep(1)
        if dash_proc.poll() is not None:
            print("\n[!] Le serveur Dash s'est arrêté.")
            break
        if django_proc.poll() is not None:
            print("\n[!] Le serveur Django s'est arrêté.")
            break
except KeyboardInterrupt:
    print("\n  Arrêt demandé par l'utilisateur...")
finally:
    dash_proc.terminate()
    django_proc.terminate()
    print("  Serveurs arrêtés.")
