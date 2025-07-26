# Test simple para verificar Docker
Write-Host "🔍 Probando Docker..." -ForegroundColor Blue

# Test 1: Comando docker existe
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Write-Host "✅ Comando 'docker' encontrado" -ForegroundColor Green
} else {
    Write-Host "❌ Comando 'docker' no encontrado" -ForegroundColor Red
    exit 1
}

# Test 2: Docker version
Write-Host "🔍 Verificando versión de Docker..." -ForegroundColor Blue
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker version: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error obteniendo versión de Docker" -ForegroundColor Red
    exit 1
}

# Test 3: Docker status con timeout
Write-Host "🔍 Verificando estado de Docker (con timeout)..." -ForegroundColor Blue
try {
    $job = Start-Job -ScriptBlock { docker version }
    $result = Wait-Job -Job $job -Timeout 10
    
    if ($result) {
        Receive-Job -Job $job | Out-Null
        Remove-Job -Job $job
        Write-Host "✅ Docker está ejecutándose correctamente" -ForegroundColor Green
    } else {
        Remove-Job -Job $job -Force
        Write-Host "❌ Timeout al verificar Docker (Docker Desktop puede no estar ejecutándose)" -ForegroundColor Red
        Write-Host "🔧 Solución: Inicia Docker Desktop y espera a que se complete el inicio" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ Error al verificar estado de Docker: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 Docker está funcionando correctamente!" -ForegroundColor Green
