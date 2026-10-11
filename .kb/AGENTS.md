# Local Git-kb guide

Read root AGENTS.md and docs/HANDOFF.md first. They define the project workflow and authorization limits. This project-specific guide replaces the generated generic template.

Use git-kb status --json and git-kb list --json. New documents remain uncommitted workspace changes; read .kb/workspaces/main/context/*.md and .kb/workspaces/main/tasks/*.md for their latest content. Status reports slugs rather than paths; list currently returns an empty array. Use --json where supported.

Claim tasks before work; coordinate overlapping files. Edit documents through checkout, never .kb/store or database/cache files. Preserve other agents changes. No commits or pushes are authorized yet. Once authorized, commit only your explicit document slugs.

No remote is configured. Embeddings are disabled. Code indexing starts after Python source exists, using git-kb code index --index-only <source-directory>.
