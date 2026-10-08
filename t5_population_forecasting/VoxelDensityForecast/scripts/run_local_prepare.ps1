<#
  本地数据预处理编排脚本。

  该脚本按顺序调用 3 个 Python 入口：
    1) 原始数据审计
    2) 状态缓存构建
    3) 闭合集群分片构建

  它会透传关键覆盖参数，并在任一步骤非零退出时立即终止。
#>

param(
    [string]$Python = 'python',
    [string]$Config = 'configs\default.yaml',
    [string]$SelectionFile = '',
    [string]$Start = '',
    [string]$End = '',
    [switch]$NoSourceHash,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
# 强制在仓库根目录执行，保证相对路径稳定。
Set-Location -LiteralPath $projectRoot

function Invoke-Checked {
    param([string[]]$CommandArgs)
    # 执行单个 Python 命令，非零退出则立即报错。
    & $Python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Python $($CommandArgs -join ' ')"
    }
}

# 阶段 1/2：审计原始文件并构建状态缓存（共享配置与筛选参数）。
$auditArgs = @('scripts\audit_raw_data.py', '--config', $Config)
$stateArgs = @('scripts\prepare_states.py', '--config', $Config)
if ($SelectionFile) {
    $auditArgs += @('--selection-file', $SelectionFile)
    $stateArgs += @('--selection-file', $SelectionFile)
}
if ($NoSourceHash) { $stateArgs += '--no-source-hash' }
if ($Overwrite) { $stateArgs += '--overwrite' }
Invoke-Checked $auditArgs
Invoke-Checked $stateArgs

# 阶段 3：构建闭合集群月度分片。
$shardArgs = @('scripts\prepare_closed_cohort.py', '--config', $Config)
if ($Start) { $shardArgs += @('--start', $Start) }
if ($End) { $shardArgs += @('--end', $End) }
if ($Overwrite) { $shardArgs += '--overwrite' }
Invoke-Checked $shardArgs
Write-Host 'Local preprocessing completed.'
