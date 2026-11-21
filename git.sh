
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/strata-moeboost"

BRANCH="$(git branch --show-current)"
[[ -n "$BRANCH" ]] || {
    echo "ERROR: No current Git branch."
    exit 1
}

BACKUP="../strata-moeboost-changes-backup-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo " STRATA MOEBOOST - COMPARE BACKUP AND PUSH"
echo "=========================================="
echo "Project: $(pwd)"
echo "Branch:  $BRANCH"

# ------------------------------------------
# 1. FETCH
# ------------------------------------------
echo
echo "===== 1. FETCH GITHUB ====="

git fetch origin --prune

# Fetch upstream if configured
if git remote get-url upstream >/dev/null 2>&1; then
    git fetch upstream --prune
fi

# ------------------------------------------
# 2. FAST COMPARISON
# ------------------------------------------
echo
echo "===== 2. FAST BRANCH COMPARISON ====="

for RB in origin/main upstream/main; do
    if git show-ref --verify --quiet "refs/remotes/$RB"; then
        echo
        echo "--- HEAD vs $RB ---"

        git rev-list --left-right --count "HEAD...$RB"

        echo "Changed files (first 30):"
        git diff --name-status "HEAD...$RB" | head -30 || true
    fi
done

echo
echo "===== YOUR GITHUB BRANCH STATUS ====="

if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    git rev-list --left-right --count "HEAD...origin/$BRANCH"
else
    echo "Remote feature branch does not exist yet."
fi

# ------------------------------------------
# 3. BACKUP CHANGED FILES ONLY
# ------------------------------------------
echo
echo "===== 3. BACKUP CHANGED FILES ====="

mkdir -p "$BACKUP"

# Back up tracked modifications and deletions
git diff HEAD --name-only -z |
while IFS= read -r -d '' FILE; do
    mkdir -p "$BACKUP/$(dirname "$FILE")"

    if [[ -f "$FILE" ]]; then
        cp -a "$FILE" "$BACKUP/$FILE"
    else
        git show "HEAD:$FILE" > "$BACKUP/$FILE"
    fi
done

# Back up new untracked files
git ls-files --others --exclude-standard -z |
while IFS= read -r -d '' FILE; do
    case "$FILE" in
        moeboost_tools_cpp/CMakeFiles/*|\
        moeboost_tools_cpp/CMakeCache.txt|\
        moeboost_tools_cpp/Makefile|\
        moeboost_tools_cpp/cmake_install.cmake|\
        moeboost_tools_cpp/*.o|\
        moeboost_tools_cpp/*.bin|\
        moeboost_tools_cpp/moeboost_cli|\
        moeboost_tools_cpp/moeboost_tools|\
        moeboost_tools_cpp/test_each_tool_runner)
            continue
            ;;
    esac

    mkdir -p "$BACKUP/$(dirname "$FILE")"
    cp -a "$FILE" "$BACKUP/$FILE"
done

echo "Backup: $BACKUP"
echo "Backed-up files:"
find "$BACKUP" -type f -print

# ------------------------------------------
# 4. AUTOMATIC STAGING
# ------------------------------------------
echo
echo "===== 4. AUTO-DETECT AND STAGE ====="

git add -A

# Exclude generated CMake files and binaries
git reset -q -- \
    moeboost_tools_cpp/CMakeFiles \
    moeboost_tools_cpp/CMakeCache.txt \
    moeboost_tools_cpp/Makefile \
    moeboost_tools_cpp/cmake_install.cmake \
    moeboost_tools_cpp/moeboost_cli \
    moeboost_tools_cpp/moeboost_tools \
    moeboost_tools_cpp/test_each_tool_runner

# ------------------------------------------
# 5. WHITESPACE CLEANUP
# ------------------------------------------
echo
echo "===== 5. CLEAN STAGED TEXT FILES ====="

python3 - <<'PY'
import subprocess
from pathlib import Path

files = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only", "-z"]
).decode().split("\0")

for name in files:
    if not name:
        continue

    path = Path(name)

    if not path.is_file():
        continue

    try:
        content = path.read_text()
    except (UnicodeDecodeError, OSError):
        continue

    cleaned = "\n".join(
        line.rstrip() for line in content.splitlines()
    ).rstrip() + "\n"

    if content != cleaned:
        path.write_text(cleaned)
        print("Cleaned:", name)
PY

# Re-stage any cleaned files
git add -u

# ------------------------------------------
# 6. REVIEW AND VALIDATE
# ------------------------------------------
echo
echo "===== 6. REVIEW STAGED CHANGES ====="

git status --short
git diff --cached --check
git diff --cached --stat

if git diff --cached --quiet; then
    echo "No changes to commit."
else
    # --------------------------------------
    # 7. COMMIT
    # --------------------------------------
    echo
    echo "===== 7. COMMIT ====="

    git commit -m "feat: update MoEBoost runtime and tooling"
fi

# ------------------------------------------
# 8. CHECK TOKEN
# ------------------------------------------
echo
echo "===== 8. GITHUB TOKEN ====="

if [[ -z "${GH_TOKEN:-}" ]]; then
    read -rsp "Enter GitHub Personal Access Token: " GH_TOKEN
    echo
fi

[[ -n "$GH_TOKEN" ]] || {
    echo "ERROR: Empty GitHub token."
    exit 1
}

export GH_TOKEN

# ------------------------------------------
# 9. TEMPORARY ASKPASS
# ------------------------------------------
ASKPASS="$(mktemp)"
chmod 700 "$ASKPASS"

cleanup() {
    rm -f "$ASKPASS"
    unset GH_TOKEN
}
trap cleanup EXIT

cat > "$ASKPASS" <<'ASKPASS_EOF'
#!/bin/sh
case "$1" in
    *Username*)
        printf '%s\n' "x-access-token"
        ;;
    *Password*)
        printf '%s\n' "$GH_TOKEN"
        ;;
    *)
        exit 1
        ;;
esac
ASKPASS_EOF

# ------------------------------------------
# 10. PUSH
# ------------------------------------------
echo
echo "===== 9. PUSH TO YOUR GITHUB ====="

GIT_ASKPASS="$ASKPASS" \
GIT_TERMINAL_PROMPT=0 \
git push -u origin "$BRANCH"

# ------------------------------------------
# 11. VERIFY
# ------------------------------------------
echo
echo "===== 10. FINAL VERIFICATION ====="

git status -sb

echo
echo "Latest commit:"
git log -1 --oneline

echo
echo "=========================================="
echo " COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "Branch: $BRANCH"
echo "Backup: $BACKUP"
echo "Remote: origin"
echo "=========================================="
