# Agent workflow

IDS-ML-Project targets a polished IDS simulation and neural training workbench powered by reliable synthetic Python data and PyTorch. A small classifier is the first milestone, not the final scope. Read README.md, docs/PRODUCT_VISION.md, and docs/HANDOFF.md at the start of each session.

## Coordination
- Run `git-kb status --json` and `git-kb list --json` from the project root. Read committed context and tasks with `git-kb show <slug> --json`, plus checked-out Markdown under `.kb/workspaces/main/context/` and `.kb/workspaces/main/tasks/`: uncommitted documents do not appear in the current list query. Status reports slugs rather than paths.
- Use a unique agent ID. Claim work using `git-kb assign <slug> <agent-id> --json`; never force another agent's assignment. If assignment requires a committed document, coordinate ownership with the user and record it in the handoff instead of bypassing the restriction.
- Record task scope, intended files, acceptance criteria, progress, checks, blockers, and next steps. Use `active` while working and `completed` only after verification.
- Task assignment does not lock files. Coordinate before editing shared files. Only one agent at a time updates shared context or indexes code. Preserve others' changes; never reset, clear, or commit their work.
- Update docs/HANDOFF.md at handoff. Do not spawn additional agents unless requested.

## Git-kb
Git-kb 0.2.12 is initialized locally without a remote. Use `--json` where supported and inspect `--help` for syntax.

Create uniquely named tasks with `git-kb create task --slug tasks/<name> --title '<title>' --body '<scope and criteria>' --json`. Use checkout and edit the Markdown path printed by the CLI. Never edit `.kb/store` or cache databases directly.

New documents are workspace changes until committed. Git commits and Git-kb commits are separate. The original no-commit/no-push restriction remains in effect until the user authorizes those actions. Once authorized, commit only explicit owned slugs: `git-kb commit -m '<message>' <slug> ...`. Never use an all-documents commit in a shared workspace.

After source exists, use `git-kb code index --index-only <source-directory>`; add `--prune` after deletions. Embeddings are disabled. There is no code to index yet. Source files and verified results remain authoritative.

## Project direction
Proposed first version: reproducible synthetic data, binary normal/suspicious labels, train/validation/test split before fitting preprocessing, a small network, and evaluation. Keep Streamlit separate from training. Record design changes and their reasons in Git-kb.

Use a local `.venv` and declared dependencies when installation is authorized. Exclude environments, secrets, generated data, and model outputs from source control. Run checks appropriate to each change and record commands and results. Synthetic evaluation does not establish production intrusion detection accuracy.

## Setup boundary
Documentation and local KB initialization are authorized. Application code, dependency installation, commits, and pushes have not been authorized in this coordination step. Python is 3.12.3 via `python3` on WSL2/Bash; pip and project libraries are absent. The empty read-only `.git` directory is not a usable repository. Do not delete it, change its permissions, or invent remotes; resolve access with the user/environment.
