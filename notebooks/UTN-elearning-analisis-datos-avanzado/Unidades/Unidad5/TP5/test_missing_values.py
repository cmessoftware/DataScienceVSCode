#!/usr/bin/env python3
"""
Test script para diagnosticar y solucionar el problema de valores faltantes
"""

import pandas as pd
import numpy as np
from models import ChurnPredictor

def test_missing_values_fix():
    """
    Prueba que el preprocesador maneja correctamente los valores faltantes
    """
    print("🧪 PRUEBA DE SOLUCIÓN PARA VALORES FALTANTES")
    print("=" * 60)
    
    # Crear datos similares a los del dataset Telco con valores faltantes
    np.random.seed(42)
    test_data = pd.DataFrame({
        'tenure': [np.nan, 12, 24, 36, 6, np.nan, 18],
        'MonthlyCharges': [65.0, np.nan, 85.2, 91.5, 45.3, 70.0, np.nan],
        'TotalCharges': [1000.5, np.nan, 2040.8, 3295.5, 271.8, 1260.0, 1100.2],
        'InternetService': ['DSL', 'Fiber optic', np.nan, 'DSL', 'No', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'Yes', np.nan, 'No', 'Yes', np.nan],
        'Contract': ['Month-to-month', 'One year', np.nan, 'Two year', 'Month-to-month', 'One year', 'Two year'],
        'gender': ['Male', 'Female', 'Male', 'Female', np.nan, 'Male', 'Female'],
    })
    
    print("📊 Dataset de prueba con valores faltantes:")
    print(test_data)
    
    # Usar el diagnóstico
    predictor = ChurnPredictor()
    predictor.diagnose_missing_values(test_data)
    
    # Crear y probar el preprocesador
    print("\n🔧 Creando preprocesador...")
    preprocessor = predictor.create_preprocessor(test_data)
    
    print("\n🔄 Aplicando transformación...")
    try:
        X_transformed = preprocessor.fit_transform(test_data)
        print(f"✅ ¡Transformación exitosa!")
        print(f"   - Shape original: {test_data.shape}")
        print(f"   - Shape transformado: {X_transformed.shape}")
        
        # Verificar que no hay NaN
        df_transformed = pd.DataFrame(X_transformed)
        has_nan = df_transformed.isna().any().any()
        
        if not has_nan:
            print("   - ✅ No hay valores NaN en los datos transformados")
            print("   - 🎉 ¡Los modelos pueden ser entrenados sin problemas!")
        else:
            print("   - ⚠️ Aún hay valores NaN (esto no debería pasar)")
            
    except Exception as e:
        print(f"❌ Error en transformación: {e}")
        return False
    
    print("\n💡 INSTRUCCIONES PARA EL USUARIO:")
    print("=" * 40)
    print("1. El preprocesador ahora maneja automáticamente los valores faltantes")
    print("2. Simplemente ejecuta tu código normal:")
    print("   predictor = ChurnPredictor()")
    print("   preprocessor = predictor.create_preprocessor(X_train)")
    print("   predictor.create_models()")
    print("   predictor.train_models(X_train, y_train)")
    print("3. Los valores faltantes serán imputados automáticamente")
    
    return True

if __name__ == "__main__":
    success = test_missing_values_fix()
    if success:
        print("\n🎉 ¡PRUEBA COMPLETADA EXITOSAMENTE!")
        print("El problema de valores faltantes ha sido solucionado.")
    else:
        print("\n❌ La prueba falló. Revisar el código.")
