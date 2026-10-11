---
id: 01a12780-accd-7b22-92a7-a18029984b06
slug: tasks/environment-setup
title: "Assess and prepare environment setup"
type: task
status: completed
priority: medium
agent_id: agent-2-env-setup
---

Unassigned next task for second agent. Read handoff; inspect current repository and Python state; document concrete setup steps and access blockers. Files: docs/HANDOFF.md and this task; coordinate any shared AGENTS.md changes. Installation and application implementation require further user authorization; no commits or pushes. Acceptance: verified environment facts, proposed .venv/dependency approach, repository access plan, clear next step.

## Findings (agent-2-env-setup, 2026-10-10)

Claimed via `git-kb set tasks/environment-setup status=active agent_id=agent-2-env-setup` — `git-kb assign` returned `{"error":"...","success":false}` because the document is workspace-only (uncommitted); the assign CAS query against the committed DB found 0 rows. This matches AGENTS.md's predicted limitation; `set` is workspace-first and worked instead. No force used.

### Python / pip
- `python3` = 3.12.3 at `/usr/bin/python3`. No `python` binary.
- No `pip`/`pip3` executable, and `python3 -c "import ensurepip"` fails — Ubuntu's distro `python3` package ships without `ensurepip` bundled.
- `python3 -m venv` module itself IS present and runs (`venv --help` succeeds), but a venv created now would have no `pip` for the same reason.
- `apt-cache policy python3-pip` shows candidate `24.0+dfsg-1ubuntu1.3` available in configured Ubuntu repos, not installed. Getting a usable pip requires either `sudo apt install python3-pip` (and likely `python3-venv` / `python3.12-venv`) or downloading `get-pip.py` — both are dependency-installation actions requiring explicit authorization, and the apt path needs `sudo`.

### Proposed setup plan (once authorized)
1. `sudo apt install python3-venv python3-pip` (or confirm `python3.12-venv` package name first) to get a working `ensurepip`/pip.
2. `python3 -m venv .venv --copies` from repo root — use `--copies` rather than symlinks, since the project tree lives on a WSL `9p`/drvfs mount of the Windows C: drive (`findmnt` confirms `fstype=9p`), where symlink support is unreliable.
3. Activate and `pip install --upgrade pip`, then install declared deps (torch, streamlit, numpy, pandas, pytest) — exact versions to be pinned in a `requirements.txt`/`pyproject.toml` once someone writes it (none exists yet; checked `requirements*.txt`, `pyproject.toml`, `setup.py` at depth 2 — none found).
4. Verify with `python3 -c "import torch, streamlit"` and `pytest --version` inside the venv.
5. Add `.venv/` to `.gitignore` (already covers venvs — confirm) before any commit.

### Repository access blocker — escalated, needs user/environment action
`docs/HANDOFF.md` described an "empty read-only `.git` folder." That is no longer accurate: **no `.git` entry exists at all now** — `ls -la`, `find -maxdepth 1 -iname ".git"` from the repo root show nothing. I did not delete or touch any `.git` path; this is a direct observation, not a change I made. This is a change from the prior session's state and needs the user to confirm the intended source of truth (restore the placeholder, point to a different remote/location, or confirm `git init` is wanted) before anyone runs `git init` or similar. Per AGENTS.md this requires the user/environment to resolve — not invented remotes, and not done unilaterally here.

Filesystem notes relevant to repo setup: project root is mounted via WSL2 `9p` (drvfs) from Windows `C:\`; all files show `rwxrwxrwx`/uid 1000 regardless of real permissions (normal for drvfs, not a sign of misconfigured permissions).

### Git-kb facts
- `git-kb status --json`: 7 workspace `added` documents (3 context notes + 4 tasks), `clean: false`, no head commit yet.
- `git-kb list --json`: `[]` — expected per AGENTS.md/.kb/AGENTS.md, since nothing is committed yet.

### Next step
Await user authorization + decision on `.git` path, then (in order): resolve/create Git repo → authorize dependency installation → `sudo apt install` + venv bootstrap per plan above → only then move to `tasks/synthetic-data`.

### Recheck, 2026-10-10 (same session, after product-vision update)
Re-verified: `.git` still absent for me (`ls -la`, `find -maxdepth 1 -iname ".git"`, `stat`, `git status` all agree it doesn't exist). The first agent's parallel note now again reports an "empty read-only `.git` directory," contradicting this. Two agents inspecting the same path are getting different answers in the same timeframe. Flagging this plainly rather than resolving it myself: possible causes include a host-side race on the drvfs/9p mount, a symlink/junction on the Windows side that only one process resolves, or something else about this sandboxed environment — I have no way to confirm which from inside WSL. **Recommend the user check from the Windows side (e.g. `dir /ah` in the actual `C:\...\IDS-ML-Project` folder, or re-run `ls -la` here themselves) rather than trusting either agent's single snapshot.** No git action taken by either agent; none should be taken until this is resolved by the user.

**Stage: Foundation.** This task is the prerequisite for all later stages — nothing in tasks/synthetic-data, tasks/train-evaluate, tasks/inference-replay, tasks/alert-store, or tasks/streamlit-ui can start real implementation until setup is resolved and authorized.

### Resolved, 2026-10-10 — user authorized implementation and deferred the `.git` question
User: "dont worry about github right now im prolly just gonna push the whole project at the end, lets get this thing rolling and get to coding." Treated as authorization for dependency installation and application code; `.git` resolution explicitly deferred by the user, not resolved by either agent.

No `sudo` available (`sudo -n true` → password required, none provided). Avoided it entirely:
1. `python3 -m venv .venv --copies --without-pip` — succeeds (module itself works; only the `ensurepip` seed step was broken).
2. `curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python` — bootstraps pip 26.2.1 straight into the venv, no sudo/apt/root needed.
3. `requirements.txt` added (numpy, pandas, scikit-learn, torch via `--extra-index-url https://download.pytorch.org/whl/cpu`, streamlit, pytest) and installed with `.venv/bin/pip install -r requirements.txt`. Took ~15 min — the 9p/drvfs mount is slow for many-small-file installs, not a failure.
4. `pyproject.toml` (src layout) + `.venv/bin/pip install -e .` so `ids_ml` imports cleanly in tests/scripts.
5. Verified: `pytest tests/ -q` → 21 passed; `python -m ids_ml.pipeline` ran a full generate→split→train→evaluate→save cycle successfully (see tasks/synthetic-data and tasks/train-evaluate for details).

Status: **completed**. Setup is no longer a blocker for Stage 1 work. It remains the dependency for Stage 2 (tasks/inference-replay, tasks/alert-store, tasks/streamlit-ui) only insofar as they need this same venv, which already exists.