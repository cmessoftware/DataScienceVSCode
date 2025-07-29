# Script para detener el entorno Data Science VSCode - Windows PowerShell
# Script para detener los servicios Docker del entorno

param(
    [switch]$Help
)

# Colores para output
function Write-ColorOutput($Color, $Message) {
    Write-Host $Message -ForegroundColor $Color
}

function Show-Help {
    Write-Host ""
    Write-ColorOutput Cyan "� DATA SCIENCE VSCODE - DETENER SERVICIOS"
    Write-ColorOutput Cyan "=========================================="
    Write-Host ""
    Write-ColorOutput Yellow "USO:"
    Write-Host "  .\quick-stop.ps1 [OPCIONES]"
    Write-Host ""
    Write-ColorOutput Yellow "OPCIONES:"
    Write-Host "  -Help           Mostrar esta ayuda"
    Write-Host ""
    Write-ColorOutput Yellow "DESCRIPCIÓN:"
    Write-Host "  Detiene todos los servicios Docker del entorno Data Science"
    Write-Host ""
    exit 0
}

function Show-Banner {
    Write-Host ""
    Write-ColorOutput Cyan "� DATA SCIENCE VSCODE - DETENER SERVICIOS"
    Write-ColorOutput Cyan "=========================================="
    Write-Host ""
}

function Stop-Environment {
    Write-ColorOutput Blue "� Deteniendo servicios Docker..."
    
    & ".\docker-helper.ps1" stop
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✅ Servicios detenidos exitosamente"
    } else {
        Write-ColorOutput Red "❌ Error al detener servicios"
        Write-ColorOutput Yellow "🔧 Intenta: docker-compose down"
        exit 1
    }
}

function Show-StopInfo {
    Write-Host ""
    Write-ColorOutput Green "🛑 ¡SERVICIOS DETENIDOS!"
    Write-ColorOutput Cyan "========================"
    Write-Host ""
    Write-ColorOutput Yellow " COMANDOS ÚTILES:"
    Write-ColorOutput White "  .\quick-start.ps1            # Reiniciar entorno"
    Write-ColorOutput White "  .\docker-helper.ps1 start    # Solo iniciar servicios"
    Write-ColorOutput White "  .\docker-helper.ps1 status   # Ver estado"
    Write-ColorOutput White "  .\docker-helper.ps1 help     # Ayuda completa"
    Write-Host ""
}

function Main {
    # Mostrar ayuda si se solicita
    if ($Help) {
        Show-Help
        return
    }
    
    Show-Banner
    
    # Detener servicios
    Stop-Environment
    
    # Mostrar información
    Show-StopInfo
    
    Write-ColorOutput Green "� Proceso completado. Servicios detenidos."
    Write-ColorOutput Yellow "💡 Para reiniciar: .\quick-start.ps1"
}

# Ejecutar función principal
Main



