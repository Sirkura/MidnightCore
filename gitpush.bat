@echo on
setlocal EnableExtensions EnableDelayedExpansion

REM === CONFIG ===
set "REPO_DIR=G:\Experimental\Production\MidnightCore"
set "REMOTE_URL=https://github.com/Sirkura/MidnightCore.git"
set "BRANCH=main"

REM Optional: extra Git diagnostics (uncomment if needed)
REM set GIT_TRACE=1
REM set GIT_CURL_VERBOSE=1

where git || (echo [ERROR] Git not found in PATH & pause & exit /b 1)
if not exist "%REPO_DIR%" (echo [ERROR] Repo dir not found: "%REPO_DIR%" & pause & exit /b 1)

pushd "%REPO_DIR%" || (echo [ERROR] cd failed & pause & exit /b 1)

REM Init if needed
git rev-parse --is-inside-work-tree 1>nul 2>nul || git init || (echo [ERROR] git init failed & goto :fail)

REM Ensure branch & remote
git branch -M "%BRANCH%" 1>nul 2>nul
git remote get-url origin 1>nul 2>nul && (
  git remote set-url origin "%REMOTE_URL%" || goto :fail
) || (
  git remote add origin "%REMOTE_URL%" || goto :fail
)

REM (Optional) identify user config so commits don’t fail
git config user.name || git config user.name "Your Name"
git config user.email || git config user.email "you@example.com"

REM Fetch, then pull only if remote branch exists
git fetch origin "%BRANCH%" 1>nul 2>nul
git rev-parse --verify "origin/%BRANCH%" 1>nul 2>nul && (
  git pull --rebase origin "%BRANCH%" || goto :fail
)

REM Stage & commit if needed
git add -A || goto :fail
for /f %%A in ('git status --porcelain') do set CHANGED=1
if defined CHANGED (
  git commit -m "Update %date% %time%" || goto :fail
) else (
  echo [INFO] No changes to commit.
)

REM Push
git push -u origin "%BRANCH%" || goto :fail

echo [DONE] Push complete.
popd
pause
exit /b 0

:fail
echo.
echo [ERROR] A step failed. See the message(s) above for the exact cause.
popd
pause
exit /b 1
