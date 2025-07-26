# Detener entorno Data Science VSCode - Windows PowerShell
# Script para detener servicios de forma limpia

function Write-ColorOutput($Color, $Message) {
    Write-Host $Message -ForegroundColor $Color
}

function Show-Banner {
    Write-Host ""
    Write-ColorOutput Red "🛑 DATA SCIENCE VSCODE - DETENER SERVICIOS"
    Write-ColorOutput Red "==========================================="
    Write-Host ""
}

function Stop-Environment {
    Write-ColorOutput Blue "🔄 Deteniendo servicios..."
    
    & ".\docker-helper.ps1" stop
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✅ Servicios detenidos exitosamente"
    } else {
        Write-ColorOutput Red "❌ Error al detener servicios"
        Write-ColorOutput Yellow "🔧 Intenta: docker-compose down"
    }
}

function Show-Status {
    Write-ColorOutput Blue "📊 Estado actual de contenedores:"
    & ".\docker-helper.ps1" status
}

function Main {
    Show-Banner
    Stop-Environment
    Write-Host ""
    Show-Status
    Write-Host ""
    Write-ColorOutput Green "🏁 Proceso completado. Servicios detenidos."
    Write-ColorOutput Yellow "💡 Para reiniciar: .\quick-start.ps1"
}

# Ejecutar función principal
Main
