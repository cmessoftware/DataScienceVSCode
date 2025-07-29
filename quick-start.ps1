# Inicio rápido para Data Science VSCode - Windows PowerShell
# Script para configurar y iniciar el entorno completo

param(
    [switch]$SkipBuild,
    [switch]$OpenBrowser,
    [switch]$ShowLogs,
    [switch]$Build,
    [switch]$Help
)

# Colores para output
function Write-ColorOutput($Color, $Message) {
    Write-Host $Message -ForegroundColor $Color
}

function Show-Help {
    Write-Host ""
    Write-ColorOutput Cyan "🚀 DATA SCIENCE VSCODE - INICIO RÁPIDO"
    Write-ColorOutput Cyan "======================================"
    Write-Host ""
    Write-ColorOutput Yellow "USO:"
    Write-Host "  .\quick-start.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput Yellow "OPCIONES:"
    Write-Host "  -SkipBuild      Omitir construcción de imagen (usar imagen existente)"
    Write-Host "  -OpenBrowser    Abrir Jupyter Lab en navegador automáticamente (default: true)"
    Write-Host "  -ShowLogs       Mostrar logs al finalizar"
    Write-Host "  -Help           Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput Yellow "EJEMPLOS:"
    Write-Host "  .\quick-start.ps1 -Build            # Inicio completo"
    Write-Host "  .\quick-start.ps1 -SkipBuild        # Solo iniciar servicios"
    Write-Host "  .\quick-start.ps1 -ShowLogs         # Mostrar logs al final"
    Write-Host "  .\quick-start.ps1 -OpenBrowser:`$false # No abrir navegador"
    Write-Host ""
    exit 0
}

function Show-Banner {
    Write-Host ""
    Write-ColorOutput Cyan "🚀 DATA SCIENCE VSCODE - INICIO RÁPIDO"
    Write-ColorOutput Cyan "======================================"
    Write-Host ""
}

function Test-DockerAvailable {
    Write-ColorOutput Blue "🔍 Verificando Docker..."
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-ColorOutput Red "❌ Docker no está instalado o no está en el PATH"
        Write-ColorOutput Yellow "📥 Descarga Docker Desktop desde: https://www.docker.com/products/docker-desktop"
        exit 1
    }
    
    # Verificar que Docker esté ejecutándose
    try {
        $dockerTest = docker version --format json 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrEmpty($dockerTest)) {
            Write-ColorOutput Red "❌ Docker no está ejecutándose"
            Write-ColorOutput Yellow "🔧 Inicia Docker Desktop y vuelve a intentar"
            exit 1
        }
    } catch {
        Write-ColorOutput Red "❌ Error al conectar con Docker"
        Write-ColorOutput Yellow "🔧 Asegúrate de que Docker Desktop esté ejecutándose"
        exit 1
    }
    
    Write-ColorOutput Green "✅ Docker está disponible y ejecutándose"
}

function Build-Environment {
    if ($SkipBuild -and -not $Build) {
        Write-ColorOutput Yellow "⏭️ Saltando construcción de imagen..."
        return
    }
    
    Write-ColorOutput Blue "📊 Construyendo imagen Docker de Data Science..."
    Write-ColorOutput Yellow "⏳ Esto puede tomar varios minutos la primera vez..."
    
    & ".\docker-helper.ps1" build
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✅ Imagen construida exitosamente"
    } else {
        Write-ColorOutput Red "❌ Error al construir imagen"
        Write-ColorOutput Yellow "🔧 Revisa los logs arriba para más detalles"
        exit 1
    }
}

function Start-Environment {
    Write-ColorOutput Blue "🎯 Iniciando entorno de Jupyter Lab..."
    
    & ".\docker-helper.ps1" start
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✅ Entorno iniciado exitosamente"
        Show-AccessInfo
    } else {
        Write-ColorOutput Red "❌ Error al iniciar servicios"
        Write-ColorOutput Yellow "🔧 Intenta: .\docker-helper.ps1 logs"
        exit 1
    }
}

function Show-AccessInfo {
    Write-Host ""
    Write-ColorOutput Green "🎉 ¡ENTORNO LISTO!"
    Write-ColorOutput Cyan "=================="
    Write-Host ""
    Write-ColorOutput Yellow "🌐 Jupyter Lab: http://localhost:8888"
    Write-ColorOutput Yellow "🔑 Token: datascience2024"
    Write-ColorOutput Yellow "📁 Notebooks: UTN-elearning-analisis-datos-avanzado/"
    Write-Host ""
    Write-ColorOutput Cyan "📋 COMANDOS ÚTILES:"
    Write-ColorOutput White "  .\docker-helper.ps1 stop      # Detener servicios"
    Write-ColorOutput White "  .\quick-stop.ps1             # Detener con script"
    Write-ColorOutput White "  .\docker-helper.ps1 logs     # Ver logs"
    Write-ColorOutput White "  .\docker-helper.ps1 shell    # Abrir terminal"
    Write-ColorOutput White "  .\docker-helper.ps1 status   # Ver estado"
    Write-ColorOutput White "  .\docker-helper.ps1 help     # Ayuda completa"
    Write-Host ""
    
    # OpenBrowser es true por defecto, solo false si se especifica explícitamente
    if (-not $OpenBrowser.IsPresent -or $OpenBrowser) {
        Write-ColorOutput Blue "🌐 Abriendo Jupyter Lab en el navegador..."
        Start-Process "http://localhost:8888/?token=datascience2024"
    }
}

function Test-JupyterAccess {
    Write-ColorOutput Blue "🔍 Verificando acceso a Jupyter Lab..."
    
    Start-Sleep -Seconds 5
    
    try {
        Invoke-WebRequest -Uri "http://localhost:8888" -TimeoutSec 10 -ErrorAction Stop | Out-Null
        Write-ColorOutput Green "✅ Jupyter Lab responde correctamente"
    } catch {
        Write-ColorOutput Yellow "⚠️ Jupyter Lab aún se está iniciando..."
        Write-ColorOutput Yellow "🔧 Espera unos segundos e intenta acceder manualmente"
    }
}

function Main {
    # Mostrar ayuda si se solicita o no hay parámetros
    if ($Help -or (-not $SkipBuild -and -not $OpenBrowser.IsPresent -and -not $ShowLogs -and -not $Build)) {
        Show-Help
        return
    }
    
    # Si se especifica Build, construir entorno completo
    if ($Build) {
        Show-Banner
        Test-DockerAvailable
        Build-Environment
        Start-Environment
        Test-JupyterAccess
        
        if ($ShowLogs) {
            Write-ColorOutput Blue "📋 Mostrando logs..."
            & ".\docker-helper.ps1" logs
        }
        
        Write-ColorOutput Green "🚀 Proceso completado. ¡Feliz análisis de datos!"
        Write-ColorOutput Yellow "💡 Para detener: .\quick-stop.ps1"
        return
    }
    
    Show-Banner
    
    # Verificar prerrequisitos
    Test-DockerAvailable
    
    # Solo construir si no se especifica SkipBuild
    if (-not $SkipBuild) {
        Build-Environment
    }
    
    # Iniciar servicios
    Start-Environment
    
    # Verificar acceso
    Test-JupyterAccess
    
    # Mostrar logs si se solicita
    if ($ShowLogs) {
        Write-ColorOutput Blue "📋 Mostrando logs..."
        & ".\docker-helper.ps1" logs
    }
    
    Write-ColorOutput Green "🚀 Proceso completado. ¡Feliz análisis de datos!"
    Write-ColorOutput Yellow "💡 Para detener: .\quick-stop.ps1"
}

# Ejecutar función principal
Main



