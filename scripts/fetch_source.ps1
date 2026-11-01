# Not run on its own: dot-sourced by the build scripts that build from a
# pinned source archive (build_ffmpeg_decoders.ps1, build_dav1d.ps1), as
#   . (Join-Path $PSScriptRoot 'fetch_source.ps1')
# so that its functions are defined in the caller's scope (run with &, they
# would go with the child scope).
#
# Get-PinnedArchive returns the path of a verified copy of an archive (its
# SHA-256 the pinned one) in the downloads folder, which outlives the
# scripts' work folders: an archive downloaded once is not downloaded again,
# so a rebuild works while its site is down. -Given (the scripts' -Archive)
# puts a copy of one on disk there first. A run killed outright can leave a
# <name>.partial-<process id> there, harmless.

# Whether the file at $Path is there and has the SHA-256 $Sha256.
function Test-ArchiveHash([string]$Path, [string]$Sha256) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    # (Windows PowerShell's Get-FileHash returns nothing for a file it cannot read, unless told to stop)
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash
    return $hash.ToLowerInvariant() -eq $Sha256.ToLowerInvariant()
}

function Get-PinnedArchive {
    param(
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$Name,  # the archive's file name
        [Parameter(Mandatory)] [string]$Sha256,
        [string]$Given = '',  # a copy of it on disk, kept in the downloads folder instead of downloading it
        [string]$CacheDirectory = (Join-Path $env:LOCALAPPDATA 'VideoMetricsLab-build/downloads')
    )
    New-Item -ItemType Directory -Force -Path $CacheDirectory | Out-Null
    $target = Join-Path (Resolve-Path -LiteralPath $CacheDirectory).ProviderPath $Name
    if ($Given) {
        if (-not (Test-Path -LiteralPath $Given -PathType Leaf)) { throw "no archive at $Given" }
        $Given = (Resolve-Path -LiteralPath $Given).ProviderPath
        if (-not (Test-ArchiveHash $Given $Sha256)) { throw "$Given is not $Name of SHA-256 $Sha256" }
        if ($Given -eq $target) {
            Write-Host "using $target (SHA-256 ok)"
            return $target
        }
    } elseif (Test-ArchiveHash $target $Sha256) {
        Write-Host "using cached $target (SHA-256 ok)"
        return $target
    } elseif (Test-Path -LiteralPath $target) {
        Write-Host "cached $target has another SHA-256: downloading it again"
    }
    # Copied or downloaded next to the archive under another name, checked, then renamed: the archive's own name
    # only ever holds a whole, verified file
    $partial = "$target.partial-$PID"
    try {
        if ($Given) {
            Copy-Item -LiteralPath $Given -Destination $partial
        } else {
            Write-Host "downloading $Url"
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12  # (for the process)
            $ProgressPreference = 'SilentlyContinue'  # (this function's: the progress bar slows a large download)
            Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing
        }
        if (-not (Test-ArchiveHash $partial $Sha256)) { throw "$(if ($Given) { $Given } else { $Url }) gave no $Name of SHA-256 $Sha256" }
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force }  # (-Force: read-only too)
        Move-Item -LiteralPath $partial -Destination $target
    } finally {
        # (what a failed transfer or a wrong download left)
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    }
    if ($Given) { Write-Host "using $Given (SHA-256 ok), kept as $target" }
    return $target
}
