param(
    [int]$Dataset = 2,
    [double[]]$Taus = @(0.6, 0.8),
    [int]$Threads = 32,
    [int]$TimeoutSeconds = 180,
    [switch]$Build
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RepoRoot 'no-use/build'
$ResultDir = Join-Path $BuildDir 'alg4-validation'
$CanonicalExe = Join-Path $BuildDir 'alg4-canonical.exe'
$LayeredExe = Join-Path $BuildDir 'alg4-layered-limited.exe'
$MemoryLimitBytes = 16GB

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

function Invoke-Compiler {
    param([string[]]$CompilerArgs)
    & g++ @CompilerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Compilation failed with exit code $LASTEXITCODE"
    }
}

if ($Build) {
    Push-Location $RepoRoot
    try {
        Invoke-Compiler @(
            '-std=c++17', '-fopenmp', '-mavx2', '-O3',
            'src/alg4-raw.cpp', 'src/semantic_graph.cpp',
            '-o', 'no-use/build/alg4-canonical.exe'
        )
        Invoke-Compiler @(
            '-std=c++17', '-fopenmp', '-mavx2', '-O3', '-Isrc',
            'tests/alg4_layered_limited.cpp', 'src/semantic_graph.cpp',
            '-o', 'no-use/build/alg4-layered-limited.exe'
        )
    }
    finally {
        Pop-Location
    }
}

if (!(Test-Path -LiteralPath $CanonicalExe)) {
    throw "Missing $CanonicalExe. Run with -Build."
}
if (!(Test-Path -LiteralPath $LayeredExe)) {
    throw "Missing $LayeredExe. Run with -Build."
}

function Invoke-Miner {
    param(
        [string]$Name,
        [string]$Executable,
        [double]$Tau
    )

    $TauText = $Tau.ToString('0.################',
        [System.Globalization.CultureInfo]::InvariantCulture)
    $Prefix = "ds${Dataset}_tau$($TauText.Replace('.', '_'))_$Name"
    $SinkRelative = "no-use/build/alg4-validation/$Prefix.cliques"
    $SinkAbsolute = Join-Path $RepoRoot $SinkRelative
    $LogAbsolute = Join-Path $ResultDir "$Prefix.log"

    $StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $StartInfo.FileName = $Executable
    $StartInfo.WorkingDirectory = $RepoRoot
    $StartInfo.UseShellExecute = $false
    $StartInfo.RedirectStandardInput = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $StartInfo.CreateNoWindow = $true
    $StartInfo.Arguments = "dataset $Dataset threads $Threads timeout $TimeoutSeconds"

    $Process = [System.Diagnostics.Process]::new()
    $Process.StartInfo = $StartInfo
    $null = $Process.Start()
    $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
    $StderrTask = $Process.StandardError.ReadToEndAsync()
    # Windows PowerShell writes a BOM on redirected stdin. Consume it on a
    # harmless first line so it cannot prefix the sink command.
    $Process.StandardInput.WriteLine('')
    $Process.StandardInput.WriteLine("sink $SinkRelative")
    $Process.StandardInput.WriteLine("mine $TauText")
    $Process.StandardInput.WriteLine('quit')
    $Process.StandardInput.Close()

    $Watch = [System.Diagnostics.Stopwatch]::StartNew()
    [int64]$PeakWorkingSet = 0
    $Status = 'completed'
    while (!$Process.WaitForExit(100)) {
        try {
            $Process.Refresh()
            if ($Process.WorkingSet64 -gt $PeakWorkingSet) {
                $PeakWorkingSet = $Process.WorkingSet64
            }
        }
        catch {
        }
        if ($PeakWorkingSet -gt $MemoryLimitBytes) {
            $Status = 'memory-limit'
            $Process.Kill()
            break
        }
        if ($Watch.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
            $Status = 'timeout'
            $Process.Kill()
            break
        }
    }
    $Process.WaitForExit()
    $Watch.Stop()
    try {
        $Process.Refresh()
        if ($Process.PeakWorkingSet64 -gt $PeakWorkingSet) {
            $PeakWorkingSet = $Process.PeakWorkingSet64
        }
    }
    catch {
    }

    $Stdout = $StdoutTask.Result
    $Stderr = $StderrTask.Result
    [System.IO.File]::WriteAllText($LogAbsolute, $Stdout + $Stderr)
    if ($Status -eq 'completed' -and $Process.ExitCode -ne 0) {
        $Status = "exit-$($Process.ExitCode)"
    }

    $TotalMs = $null
    $Matches = [regex]::Matches($Stdout, '=== TOTAL:\s+(\d+) ms')
    if ($Matches.Count -gt 0) {
        $TotalMs = [int64]$Matches[$Matches.Count - 1].Groups[1].Value
    }

    [pscustomobject]@{
        Name = $Name
        Status = $Status
        ExitCode = if ($Process.HasExited) { $Process.ExitCode } else { $null }
        WallSeconds = [math]::Round($Watch.Elapsed.TotalSeconds, 3)
        TotalMs = $TotalMs
        PeakGiB = [math]::Round($PeakWorkingSet / 1GB, 3)
        Sink = $SinkAbsolute
        Log = $LogAbsolute
    }
}

function Get-CliqueSummary {
    param([string]$Path)
    $Set = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal)
    $Sizes = [System.Collections.Generic.SortedDictionary[int, int64]]::new()
    foreach ($Line in [System.IO.File]::ReadLines($Path)) {
        $Clique = $Line.Trim()
        if ($Clique.Length -eq 0) { continue }
        if (!$Set.Add($Clique)) {
            throw "Duplicate clique in $Path`: $Clique"
        }
        $Size = ($Clique -split ' ').Count
        if (!$Sizes.ContainsKey($Size)) { $Sizes[$Size] = 0 }
        $Sizes[$Size]++
    }
    [pscustomobject]@{ Set = $Set; Sizes = $Sizes; Count = $Set.Count }
}

function Compare-CliqueOutputs {
    param([string]$ExpectedPath, [string]$ActualPath)
    $Expected = Get-CliqueSummary $ExpectedPath
    $Actual = Get-CliqueSummary $ActualPath
    $Missing = 0
    foreach ($Clique in $Expected.Set) {
        if (!$Actual.Set.Contains($Clique)) { $Missing++ }
    }
    $Extra = 0
    foreach ($Clique in $Actual.Set) {
        if (!$Expected.Set.Contains($Clique)) { $Extra++ }
    }
    $SizeMatch = $Expected.Sizes.Count -eq $Actual.Sizes.Count
    if ($SizeMatch) {
        foreach ($Pair in $Expected.Sizes.GetEnumerator()) {
            if (!$Actual.Sizes.ContainsKey($Pair.Key) -or
                $Actual.Sizes[$Pair.Key] -ne $Pair.Value) {
                $SizeMatch = $false
                break
            }
        }
    }
    [pscustomobject]@{
        ExactMatch = ($Missing -eq 0 -and $Extra -eq 0)
        SizeMatch = $SizeMatch
        ExpectedCount = $Expected.Count
        ActualCount = $Actual.Count
        Missing = $Missing
        Extra = $Extra
        ExpectedSizes = ($Expected.Sizes.GetEnumerator() | ForEach-Object {
            "$($_.Key):$($_.Value)" }) -join ','
        ActualSizes = ($Actual.Sizes.GetEnumerator() | ForEach-Object {
            "$($_.Key):$($_.Value)" }) -join ','
    }
}

$Results = @()
foreach ($Tau in $Taus) {
    Write-Host "Running layered reference: dataset=$Dataset tau=$Tau"
    $Layered = Invoke-Miner 'layered' $LayeredExe $Tau
    if ($Layered.Status -ne 'completed') {
        Write-Host "Skipping tau=$Tau because the reference status is $($Layered.Status)."
        $Results += [pscustomobject]@{ Tau = $Tau; Layered = $Layered; Canonical = $null; Compare = $null }
        continue
    }

    Write-Host "Running canonical DFS: dataset=$Dataset tau=$Tau"
    $Canonical = Invoke-Miner 'canonical' $CanonicalExe $Tau
    $Comparison = $null
    if ($Canonical.Status -eq 'completed') {
        $Comparison = Compare-CliqueOutputs $Layered.Sink $Canonical.Sink
    }
    $Results += [pscustomobject]@{
        Tau = $Tau
        Layered = $Layered
        Canonical = $Canonical
        Compare = $Comparison
    }
}

$ValidationFailed = $false
foreach ($Result in $Results) {
    Write-Host ""
    Write-Host "tau=$($Result.Tau)"
    Write-Host "  layered : status=$($Result.Layered.Status) total_ms=$($Result.Layered.TotalMs) peak_gib=$($Result.Layered.PeakGiB)"
    if ($null -ne $Result.Canonical) {
        Write-Host "  canonical: status=$($Result.Canonical.Status) total_ms=$($Result.Canonical.TotalMs) peak_gib=$($Result.Canonical.PeakGiB)"
    }
    if ($null -ne $Result.Compare) {
        Write-Host "  exact=$($Result.Compare.ExactMatch) sizes=$($Result.Compare.SizeMatch) count=$($Result.Compare.ActualCount)"
        Write-Host "  by-size=$($Result.Compare.ActualSizes)"
        if (!$Result.Compare.ExactMatch -or !$Result.Compare.SizeMatch) {
            $ValidationFailed = $true
            Write-Warning "Output mismatch at tau=$($Result.Tau): missing=$($Result.Compare.Missing) extra=$($Result.Compare.Extra)"
        }
        if ($Result.Canonical.TotalMs -ge $Result.Layered.TotalMs) {
            Write-Warning "Canonical DFS is not faster at tau=$($Result.Tau)."
        }
        if ($Result.Canonical.PeakGiB -ge $Result.Layered.PeakGiB) {
            Write-Warning "Canonical DFS does not use less peak working set at tau=$($Result.Tau)."
        }
    }
}

if ($ValidationFailed) {
    throw 'One or more output comparisons failed. See the complete summary above.'
}
