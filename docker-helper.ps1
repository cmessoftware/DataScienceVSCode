# Data Science VSCode - Docker Helper (PowerShell)
# Script para facilitar el uso de Docker en entorno de Data Science

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$RemainingArgs
)

# Colores para output
function Write-ColorOutput($Color, $Message) {
    Write-Host $Message -ForegroundColor $Color
}

function Show-Help {
    Write-ColorOutput Blue "Data Science VSCode - Docker Helper"
    Write-Host ""
    Write-Host "Uso: .\docker-helper.ps1 [COMANDO]"
    Write-Host ""
    Write-Host "Comandos disponibles:"
    Write-Host "  build           Construir la imagen Docker de Data Science"
    Write-Host "  start           Iniciar Jupyter Lab"
    Write-Host "  stop            Detener todos los servicios"
    Write-Host "  restart         Reiniciar los servicios"
    Write-Host "  logs            Mostrar logs de los servicios"
    Write-Host "  shell           Abrir una shell en el contenedor"
    Write-Host "  python          Abrir una consola Python interactiva"
    Write-Host "  notebook        Abrir un nuevo notebook"
    Write-Host "  clean           Limpiar contenedores e imágenes no utilizados"
    Write-Host "  status          Mostrar estado de los contenedores"
    Write-Host "  install         Instalar paquetes adicionales"
    Write-Host "  backup          Crear backup de notebooks"
    Write-Host "  help            Mostrar esta ayuda"
    Write-Host ""
    Write-Host "Ejemplos:"
    Write-Host "  .\docker-helper.ps1 build         # Construir la imagen"
    Write-Host "  .\docker-helper.ps1 start         # Iniciar Jupyter Lab en http://localhost:8888"
    Write-Host "  .\docker-helper.ps1 install pandas # Instalar paquete adicional"
}

function Test-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-ColorOutput Red "Error: Docker no está instalado o no está en el PATH"
        exit 1
    }
    
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        if (-not (docker compose version 2>$null)) {
            Write-ColorOutput Red "Error: Docker Compose no está disponible"
            exit 1
        }
    }
}

function Build-Images {
    Write-ColorOutput Blue "Construyendo imagen Docker de Data Science..."
    docker-compose build --no-cache
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Imagen construida exitosamente"
        Write-ColorOutput Yellow "Imagen lista para Data Science con Jupyter Lab"
    } else {
        Write-ColorOutput Red "✗ Error al construir la imagen"
    }
}

function Start-Services {
    Write-ColorOutput Blue "Iniciando entorno de Data Science..."
    docker-compose up -d datascience-vscode
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Jupyter Lab iniciado exitosamente"
        Write-ColorOutput Yellow "Jupyter Lab disponible en: http://localhost:8888"
        Write-ColorOutput Yellow "Token: datascience2024"
        Write-ColorOutput Yellow "Para ver los logs: .\docker-helper.ps1 logs"
        Write-ColorOutput Cyan "Notebooks disponibles en: UTN-elearning-analisis-datos-avanzado/"
    } else {
        Write-ColorOutput Red "✗ Error al iniciar servicios"
    }
}

function Stop-Services {
    Write-ColorOutput Blue "Deteniendo servicios..."
    docker-compose down
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Servicios detenidos"
    } else {
        Write-ColorOutput Red "✗ Error al detener servicios"
    }
}

function Restart-Services {
    Write-ColorOutput Blue "Reiniciando servicios..."
    docker-compose restart
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Servicios reiniciados"
    } else {
        Write-ColorOutput Red "✗ Error al reiniciar servicios"
    }
}

function Show-Logs {
    Write-ColorOutput Blue "Mostrando logs de los servicios..."
    docker-compose logs -f
}

function Open-Shell {
    Write-ColorOutput Blue "Abriendo shell en el contenedor..."
    docker-compose exec datascience-vscode /bin/bash
}

function Open-Python {
    Write-ColorOutput Blue "Abriendo consola Python interactiva..."
    docker-compose exec datascience-vscode python
}

function Create-Notebook {
    param([string]$NotebookName = "nuevo_notebook.ipynb")
    
    Write-ColorOutput Blue "Creando nuevo notebook: $NotebookName"
    docker-compose exec datascience-vscode jupyter notebook --generate-config
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Notebook creado: $NotebookName"
        Write-ColorOutput Yellow "Accede a http://localhost:8888 para abrirlo"
    } else {
        Write-ColorOutput Red "✗ Error al crear notebook"
    }
}

function Install-Package {
    param([string]$PackageName)
    
    if (-not $PackageName) {
        Write-ColorOutput Red "Debe especificar un paquete para instalar"
        return
    }
    
    Write-ColorOutput Blue "Instalando paquete: $PackageName"
    docker-compose exec datascience-vscode pip install $PackageName
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Paquete $PackageName instalado exitosamente"
    } else {
        Write-ColorOutput Red "✗ Error al instalar paquete $PackageName"
    }
}

function Backup-Notebooks {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = "backup_notebooks_$timestamp"
    
    Write-ColorOutput Blue "Creando backup de notebooks..."
    New-Item -ItemType Directory -Path $backupDir -Force
    Copy-Item -Path "UTN-elearning-analisis-datos-avanzado" -Destination $backupDir -Recurse
    
    if (Test-Path $backupDir) {
        Write-ColorOutput Green "✓ Backup creado en: $backupDir"
    } else {
        Write-ColorOutput Red "✗ Error al crear backup"
    }
}

function Show-Jupyter-Info {
    Write-ColorOutput Blue "Información de Jupyter Lab:"
    Write-ColorOutput Yellow "URL: http://localhost:8888"
    Write-ColorOutput Yellow "Token: datascience2024"
    Write-ColorOutput Cyan "Notebooks: UTN-elearning-analisis-datos-avanzado/"
    Write-ColorOutput Cyan "Kaggle Competitions: kaggle_competitions/"
}

function Clean-Docker {
    Write-ColorOutput Blue "Limpiando contenedores e imágenes no utilizados..."
    docker system prune -f
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ Limpieza completada"
    } else {
        Write-ColorOutput Red "✗ Error en la limpieza"
    }
}

function Show-Status {
    Write-ColorOutput Blue "Estado de los contenedores:"
    docker-compose ps
}

# Verificar Docker
Test-Docker

# Procesar comando
switch ($Command.ToLower()) {
    "build" { Build-Images }
    "start" { Start-Services }
    "stop" { Stop-Services }
    "restart" { Restart-Services }
    "logs" { Show-Logs }
    "shell" { Open-Shell }
    "python" { Open-Python }
    "notebook" { Create-Notebook }
    "install" { 
        if ($RemainingArgs.Count -gt 0) {
            $packageName = $RemainingArgs[0]
            Write-ColorOutput Blue "Instalando paquete: $packageName"
            Install-Package $packageName
        } else {
            Write-ColorOutput Red "Debe especificar un paquete para instalar"
            Write-ColorOutput Yellow "Ejemplo: .\docker-helper.ps1 install pandas"
        }
    }
    "backup" { Backup-Notebooks }
    "info" { Show-Jupyter-Info }
    "clean" { Clean-Docker }
    "status" { Show-Status }
    "help" { Show-Help }
    "--help" { Show-Help }
    "-h" { Show-Help }
    default {
        Write-ColorOutput Red "Comando no reconocido: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}
