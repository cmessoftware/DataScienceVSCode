#!/usr/bin/env python3
"""
🚀 SCRIPT RÁPIDO: Diagnóstico de NaN en tus datos
===============================================

Úsalo para diagnosticar rápidamente problemas de valores faltantes
y verificar que el preprocesador los maneja correctamente.

Uso:
    python diagnostico_nan_rapido.py
"""

import pandas as pd
import numpy as np
from models import ChurnPredictor

def diagnosticar_datos(X_data, nombre_dataset="Tu dataset"):
    """
    Diagnostica rápidamente valores NaN en cualquier dataset
    
    Args:
        X_data: DataFrame con tus datos
        nombre_dataset: Nombre descriptivo del dataset
    """
    print(f'🔍 DIAGNÓSTICO RÁPIDO: {nombre_dataset}')
    print('=' * 50)
    
    # Información básica
    print(f'📊 Shape: {X_data.shape}')
    print(f'📊 Columnas: {list(X_data.columns)}')
    
    # Diagnóstico de NaN
    predictor = ChurnPredictor()
    predictor.diagnose_missing_values(X_data)
    
    # Probar el preprocesador
    print('\n🔧 Probando preprocesador...')
    try:
        preprocessor = predictor.create_preprocessor(X_data)
        X_transformed = preprocessor.fit_transform(X_data)
        
        # Verificar resultado
        df_result = pd.DataFrame(X_transformed)
        tiene_nan = df_result.isna().any().any()
        
        print(f'✅ Transformación exitosa!')
        print(f'   - Shape transformado: {X_transformed.shape}')
        print(f'   - ¿Tiene NaN después?: {tiene_nan}')
        
        if not tiene_nan:
            print('   - 🎉 ¡Perfecto! Listo para ML')
        else:
            print('   - ⚠️ Aún hay NaN (revisar)')
            
    except Exception as e:
        print(f'❌ Error en preprocesamiento: {e}')

def crear_datos_ejemplo():
    """Crea datos de ejemplo para probar"""
    return pd.DataFrame({
        'tenure': [1, np.nan, 24, 36, 12],
        'MonthlyCharges': [29.85, 56.95, np.nan, 91.5, 70.0],
        'gender': ['Female', np.nan, 'Male', 'Female', 'Male'],
        'OnlineSecurity': ['No', 'Yes', np.nan, 'No', 'Yes'],
        'Contract': ['Month-to-month', np.nan, 'One year', 'Two year', 'Month-to-month']
    })

def main():
    """Función principal"""
    print('🚀 DIAGNÓSTICO RÁPIDO DE VALORES NaN')
    print('=' * 40)
    
    # Ejemplo con datos de prueba
    print('📝 Ejemplo con datos de prueba:')
    datos_ejemplo = crear_datos_ejemplo()
    diagnosticar_datos(datos_ejemplo, "Datos de ejemplo")
    
    print('\n' + '='*60)
    print('💡 INSTRUCCIONES DE USO:')
    print('Para usar con tus propios datos, modifica este script:')
    print('')
    print('# Cargar tus datos')
    print('mis_datos = pd.read_csv("mi_dataset.csv")')
    print('# O usar los datos que ya tienes cargados')
    print('# mis_datos = X_train_split  # por ejemplo')
    print('')
    print('# Diagnosticar')
    print('diagnosticar_datos(mis_datos, "Mi Dataset")')
    print('')
    print('🎯 Resultado esperado: "¡Perfecto! Listo para ML"')

if __name__ == "__main__":
    main()
