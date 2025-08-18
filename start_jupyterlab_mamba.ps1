# Nombre del entorno conda
$EnvName = "ds"

# Detectar instalación de Miniforge/Miniconda/Anaconda
$candidates = @(
    "$HOME\miniforge3",
    "$HOME\mambaforge",
    "$HOME\miniconda3",
    "$HOME\anaconda3"
)
$CondaRoot = $null
foreach($c in $candidates){
    if(Test-Path "$c\condabin\conda.bat"){ $CondaRoot = $c; break }
}
if (-not $CondaRoot) {
    Write-Error "❌ No se encontró Miniforge/Miniconda/Anaconda."
    exit 1
}

# Cargar conda en PowerShell
& "$CondaRoot\condabin\conda.bat" shell.powershell hook | Out-String | Invoke-Expression

# Activar el entorno
conda activate $EnvName

# Abrir Jupyter Lab en la carpeta actual (.) con el entorno ya activado
jupyter lab
