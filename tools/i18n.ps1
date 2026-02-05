<#
tools/i18n.ps1 (PowerShell-only, no bash)

Extract/merge/compile gettext catalogs using MSYS2 toolchain executables directly.
This avoids all bash -lc quoting/arg-splitting issues on Windows.

Prereqs:
- MSYS2 installed (e.g. C:\msys64)
- gettext installed in the selected toolchain:
  pacman -S --needed gettext

Run (from repo root):
  .\tools\i18n.ps1 -Action all -MsysRoot "C:\msys64" -Toolchain ucrt64
  .\tools\i18n.ps1 -Action extract -MsysRoot "C:\msys64"
  .\tools\i18n.ps1 -Action merge   -MsysRoot "C:\msys64"
  .\tools\i18n.ps1 -Action compile -MsysRoot "C:\msys64"
#>

param(
  [ValidateSet("extract","merge","compile","all")]
  [string]$Action = "all",

  [string[]]$Languages = @("zh_CN","en_US"),

  # MSYS2 install directory (e.g. C:\msys64). If empty, auto-detect.
  [string]$MsysRoot = "",

  # Which MSYS2 toolchain provides gettext binaries.
  [ValidateSet("ucrt64","mingw64")]
  [string]$Toolchain = "ucrt64"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-MsysRoot {
  param([string]$Provided)

  function Is-RootedPath {
    param([string]$p)
    if (-not $p) { return $false }
    return ($p -match '^[A-Za-z]:\\' -or $p -match '^\\\\')
  }

  # Note: avoid calling Test-Path on empty string (PowerShell can error)
  if (Is-RootedPath $Provided) {
    if (Test-Path $Provided) {
      return (Resolve-Path $Provided).Path
    }
  }

  # IMPORTANT: ignore invalid values like MSYS2_ROOT="C"
  $rawCandidates = @()
  $envRoot = $env:MSYS2_ROOT
  if ($envRoot -and ($envRoot -match '^[A-Za-z]:\\' -or $envRoot -match '^\\\\')) {
    $rawCandidates += $envRoot
  }
  $rawCandidates += @(
    "C:\msys64",
    "D:\msys64",
    "E:\msys64"
  )

  $candidates = $rawCandidates | Where-Object { Test-Path $_ }

  if (-not $candidates) {
    throw "MSYS2 not found. Set -MsysRoot (e.g. C:\msys64)."
  }

  return (Resolve-Path $candidates[0]).Path
}

function Get-GettextTool {
  param([string]$MsysRoot, [string]$Toolchain, [string]$ExeName)

  $p = Join-Path $MsysRoot (Join-Path "$Toolchain\bin" $ExeName)
  if (-not (Test-Path $p)) {
    throw "Missing $ExeName at: $p`nInstall in MSYS2: pacman -S --needed gettext"
  }
  return $p
}

# Repo root is parent of tools/
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$msys = Resolve-MsysRoot $MsysRoot

$xgettext = Get-GettextTool $msys $Toolchain "xgettext.exe"
$msgmerge = Get-GettextTool $msys $Toolchain "msgmerge.exe"
$msginit  = Get-GettextTool $msys $Toolchain "msginit.exe"
$msgfmt   = Get-GettextTool $msys $Toolchain "msgfmt.exe"

$potPath = Join-Path $repoRoot "src\locale\messages.pot"

function Ensure-Dir {
  param([string]$Dir)
  if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Path $Dir | Out-Null }
}

if ($Action -in @("extract","all")) {
  Ensure-Dir (Join-Path $repoRoot "src\locale")

  $tmpList = [System.IO.Path]::GetTempFileName()

  try {
    # Collect all src/**/*.py, write as relative paths using forward slashes
    $files = Get-ChildItem -Path (Join-Path $repoRoot "src") -Recurse -File -Filter "*.py"
    $rel = $files | ForEach-Object {
      $r = $_.FullName.Substring($repoRoot.Length).TrimStart('\','/')
      ($r -replace '\\','/')
    }

    [System.IO.File]::WriteAllLines($tmpList, $rel, (New-Object System.Text.UTF8Encoding($false)))

    Write-Host ("Extracting POT from {0} Python files..." -f $files.Count)

    & $xgettext `
      --from-code=UTF-8 -L Python `
      --keyword=tr_ `
      --keyword=ngettext:1,2 `
      --keyword=pgettext:1c,2 `
      --files-from=$tmpList `
      --output=$potPath

    if ($LASTEXITCODE -ne 0) { throw "xgettext failed (exit $LASTEXITCODE)." }

    Write-Host "OK: extracted -> src/locale/messages.pot"
  }
  finally {
    if (Test-Path $tmpList) { Remove-Item $tmpList -Force -ErrorAction SilentlyContinue }
  }
}

if ($Action -in @("merge","all")) {
  foreach ($lang in $Languages) {
    $poDir = Join-Path $repoRoot ("src\locale\{0}\LC_MESSAGES" -f $lang)
    $poPath = Join-Path $poDir "messages.po"

    Ensure-Dir $poDir

    if (-not (Test-Path $poPath)) {
      & $msginit --no-translator --input=$potPath --locale=$lang --output-file=$poPath
      if ($LASTEXITCODE -ne 0) { throw "msginit failed for $lang (exit $LASTEXITCODE)." }
    }

    & $msgmerge --update --backup=none $poPath $potPath
    if ($LASTEXITCODE -ne 0) { throw "msgmerge failed for $lang (exit $LASTEXITCODE)." }

    Write-Host ("OK: merged -> src/locale/{0}/LC_MESSAGES/messages.po" -f $lang)
  }
}

if ($Action -in @("compile","all")) {
  foreach ($lang in $Languages) {
    $poPath = Join-Path $repoRoot ("src\locale\{0}\LC_MESSAGES\messages.po" -f $lang)
    $moPath = Join-Path $repoRoot ("src\locale\{0}\LC_MESSAGES\messages.mo" -f $lang)

    & $msgfmt -c -o $moPath $poPath
    if ($LASTEXITCODE -ne 0) { throw "msgfmt failed for $lang (exit $LASTEXITCODE)." }

    Write-Host ("OK: compiled -> src/locale/{0}/LC_MESSAGES/messages.mo" -f $lang)
  }
}