import os
import sys
import warnings
from pathlib import Path

    # 🔧 Configurar codificación para emojis en Windows
    if sys.platform.startswith("win"):
        import io
    
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
        # Configurar variables de entorno
        os.environ["PYTHONIOENCODING"] = "utf-8"
    
    # 🔧 Configurar warnings
    warnings.filterwarnings("ignore")
    # Suprimir warnings específicos de joblib
    warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
    import os
    
    os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"  # Usar carpeta temporal específica
    
    
    # 🎨 Detectar soporte de emojis en el terminal
    def supports_emojis():
        """Detecta si el entorno actual soporta emojis"""
        try:
            if sys.platform.startswith("win") and "TERM" not in os.environ:
                return False
            return True
        except:
            return False
    
    
    USE_EMOJIS = supports_emojis()
    
    
    def safe_print(message):
        """
        Función helper para imprimir mensajes con emojis de manera segura
        """
        if not USE_EMOJIS:
            # Remover emojis para compatibilidad con PowerShell Windows
            import re
    
            clean_message = re.sub(r"[^\x00-\x7F]+", "[*]", message)
            print(clean_message)
        else:
            print(message)
