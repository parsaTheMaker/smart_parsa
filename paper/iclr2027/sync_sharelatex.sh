#!/usr/bin/env bash
# Keep the isolated ShareLaTeX paper clone synchronized without touching the
# parent research repository or its unrelated experiment files.
set -euo pipefail

repo_dir="/home/parsa/smart_parsa/paper/iclr2027/sharelatex_project"
log_dir="/home/parsa/smart_parsa/paper/iclr2027/sync_logs"
log_file="${log_dir}/sharelatex_sync.log"

mkdir -p "${log_dir}"
exec >>"${log_file}" 2>&1

echo "[$(date --iso-8601=seconds)] ShareLaTeX sync started"
cd "${repo_dir}"

if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    git stash push --include-untracked --message "scheduled-sharelatex-sync-$(date +%s)"
    restore_stash=1
else
    restore_stash=0
fi

# Remote changes are applied before any local changes are committed or pushed.
git fetch origin master
git rebase origin/master

if [[ "${restore_stash}" -eq 1 ]]; then
    git stash pop
fi

if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    git add --all
    git -c commit.gpgSign=false commit -m "chore(paper): scheduled ShareLaTeX sync"
fi

if [[ "$(git rev-list --count origin/master..HEAD)" -gt 0 ]]; then
    if ! git push origin HEAD:master; then
        # ShareLaTeX changed during the push. Rebase local paper work on the
        # newer remote version before one safe retry.
        git fetch origin master
        git rebase origin/master
        git push origin HEAD:master
    fi
fi
echo "[$(date --iso-8601=seconds)] ShareLaTeX sync completed"
