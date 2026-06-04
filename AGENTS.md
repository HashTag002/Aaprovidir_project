# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

Single Python app: **PESTEL Analytics / AaPROVIDIR** — agricultural commodity price analytics (Dash UI + optional Django landing). Data lives in `data/Dataset.csv` (semicolon-separated). Unified dev launcher: `run.py`.

### Services

| Service | Port | Required for analytics E2E |
|---------|------|----------------------------|
| Dash (`dashboard.dash_app`) | 8050 | Yes |
| Django (`manage.py runserver`) | 8000 | No (landing page + link to Dash) |
| SQLite (`db.sqlite3`) | file | No (only admin/auth after `migrate`) |

### Dependency install

The repo includes a committed `env/` virtualenv. **Do not use `env/bin/pip` directly** — its shebang points at another machine's path. Always install with:

```bash
test -x env/bin/python3 || python3 -m venv env
env/bin/python3 -m pip install -r requirements.txt
env/bin/python3 -m pip install scikit-learn statsmodels
```

`scikit-learn` and `statsmodels` are imported by `dashboard/dash_app.py` but are not listed in `requirements.txt`.

### Run (development)

```bash
env/bin/python3 manage.py migrate   # optional; only if using Django admin
env/bin/python3 run.py              # starts Dash :8050 and Django :8000
```

- Dash: http://localhost:8050
- Django shell: http://localhost:8000/dashboard/

`run.py` kills any process already bound to ports 8000/8050 before starting. Use tmux for long-running dev servers.

### Lint / test / build

No project lint config, test suite, or production build step. Useful sanity checks:

```bash
env/bin/python3 manage.py check
env/bin/python3 manage.py test    # no app tests in repo
```

`manage.py check` may warn that `/workspace/static` is missing (`staticfiles.W004`); this does not block dev.

### Hello-world verification

1. Start `run.py`, open http://localhost:8050.
2. Confirm KPI cards and the time-series chart render.
3. Change the **PRODUIT** dropdown (e.g. Maïs → Niébé) and confirm KPIs and chart update.
