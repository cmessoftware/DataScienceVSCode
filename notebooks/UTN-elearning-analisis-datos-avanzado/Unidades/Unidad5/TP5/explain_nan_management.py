#!/usr/bin/env python3
"""
🔍 SCRIPT: Explicación del Manejo Automático de Valores NaN
=========================================================

Este script demuestra paso a paso qué significa que "Los valores NaN se manejan 
automáticamente en el pipeline de preprocesamiento" en modelos de Machine Learning.

Autor: Asistente IA
Fecha: Julio 2025
"""

import pandas as pd
import numpy as np
from models import ChurnPredictor

def explain_nan_management():
    """
    Explica paso a paso cómo el pipeline maneja automáticamente los valores NaN
    """
    print('🔍 EXPLICACIÓN: ¿QUÉ SIGNIFICA "MANEJO AUTOMÁTICO DE NaN"?')
    print('=' * 65)
    
    # PASO 1: DATOS ORIGINALES CON NaN
    print('📊 PASO 1: DATOS ORIGINALES CON NaN')
    print('-' * 40)
    
    original_data = pd.DataFrame({
        'tenure': [1, np.nan, 24, 36, 12],          # Numérica con NaN
        'MonthlyCharges': [29.85, 56.95, np.nan, 91.5, 70.0],  # Numérica con NaN
        'TotalCharges': [100.0, np.nan, 800.0, 1200.0, 850.0], # Numérica con NaN
        'gender': ['Female', np.nan, 'Male', 'Female', 'Male'],   # Categórica con NaN
        'InternetService': ['DSL', 'Fiber optic', np.nan, 'DSL', 'No'], # Categórica con NaN
        'OnlineSecurity': ['No', 'Yes', np.nan, 'No', 'Yes'],     # Binaria con NaN
        'Contract': ['Month-to-month', np.nan, 'One year', 'Two year', 'Month-to-month']
    })
    
    print("Datos con valores NaN:")
    print(original_data)
    print(f"\n🚨 Valores NaN por columna:")
    print(original_data.isnull().sum())
    
    print('\n❌ PROBLEMA: Los modelos de ML NO pueden trabajar con NaN')
    print('   Si intentas: LogisticRegression.fit(X_con_nan) → ValueError!')
    
    # PASO 2: EL PIPELINE AUTOMÁTICO
    print('\n⚙️ PASO 2: EL PIPELINE AUTOMÁTICO')
    print('-' * 40)
    
    predictor = ChurnPredictor()
    
    # Mostrar diagnóstico
    print("\n🔍 Diagnóstico automático de valores faltantes:")
    predictor.diagnose_missing_values(original_data)
    
    # Crear preprocesador
    print('\n🔧 Creando preprocesador con manejo automático...')
    preprocessor = predictor.create_preprocessor(original_data)
    
    # PASO 3: TRANSFORMACIÓN AUTOMÁTICA
    print('\n🔄 PASO 3: TRANSFORMACIÓN AUTOMÁTICA')
    print('-' * 45)
    print('Durante preprocessor.fit_transform() sucede AUTOMÁTICAMENTE:')
    print('   1️⃣ SimpleImputer detecta valores NaN')
    print('   2️⃣ Los reemplaza según la estrategia:')
    print('      • Numéricas: NaN → mediana de la columna')
    print('      • Categóricas: NaN → valor más frecuente')
    print('   3️⃣ Aplica StandardScaler a numéricas')
    print('   4️⃣ Aplica OneHotEncoder a categóricas')
    print('   5️⃣ Retorna datos completamente numéricos')
    
    # Aplicar transformación
    print('\n⚡ Ejecutando transformación...')
    transformed_X = preprocessor.fit_transform(original_data)
    
    # PASO 4: RESULTADO FINAL
    print('\n📊 PASO 4: RESULTADO FINAL')
    print('-' * 32)
    print(f'Shape original: {original_data.shape}')
    print(f'Shape transformado: {transformed_X.shape}')
    print(f'Tipo de datos: {type(transformed_X)} con dtype {transformed_X.dtype}')
    
    # Verificar que no hay NaN
    result_df = pd.DataFrame(transformed_X)
    has_nan = result_df.isna().any().any()
    print(f'¿Contiene NaN después?: {has_nan}')
    
    print('\n📋 Primeras filas del resultado transformado:')
    print(result_df.head())
    
    print('\n✅ RESULTADO FINAL:')
    print('   • Todos los valores son numéricos (float64)')
    print('   • No hay ningún NaN')
    print('   • Los modelos ML pueden entrenar sin errores')
    print('   • Los datos están escalados y normalizados')
    
    # PASO 5: SIGNIFICADO DE "AUTOMÁTICO"
    print('\n💡 PASO 5: SIGNIFICADO DE "AUTOMÁTICO"')
    print('-' * 45)
    print('🤔 ¿Qué significa "automático"?')
    print('   📝 TÚ escribes simplemente:')
    print('      model.fit(X_train, y_train)')
    print('      model.predict(X_test)')
    print('')
    print('   🤖 EL PIPELINE hace internamente (SIN código extra):')
    print('      1. Detecta NaN → Los imputa automáticamente')
    print('      2. Escala números → StandardScaler')
    print('      3. Codifica categorías → OneHotEncoder')
    print('      4. Envía datos limpios al modelo')
    print('')
    print('   ✨ Todo esto sucede TRANSPARENTEMENTE')
    print('      No necesitas escribir código de limpieza')
    print('      No necesitas imputer manualmente')
    print('      El pipeline se encarga de todo')
    
    return transformed_X, original_data

def demonstrate_with_models():
    """
    Demuestra cómo los modelos entrenan sin problemas después del preprocesamiento
    """
    print('\n🚀 DEMOSTRACIÓN: ENTRENAMIENTO SIN PROBLEMAS')
    print('=' * 55)
    
    # Crear datos con NaN
    np.random.seed(42)
    data_with_nan = pd.DataFrame({
        'tenure': [1, np.nan, 24, 36, 12, 8, np.nan, 15],
        'MonthlyCharges': [29.85, 56.95, np.nan, 91.5, 70.0, 45.2, 85.3, np.nan],
        'gender': ['Female', np.nan, 'Male', 'Female', 'Male', 'Female', np.nan, 'Male'],
        'OnlineSecurity': ['No', 'Yes', np.nan, 'No', 'Yes', 'No', 'Yes', np.nan],
        'Contract': ['Month-to-month', np.nan, 'One year', 'Two year', 'Month-to-month', 'One year', np.nan, 'Two year']
    })
    
    # Variable objetivo (sin NaN)
    target_y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    
    print('📊 Datos de ejemplo para entrenamiento:')
    print(f'   Shape: {data_with_nan.shape}')
    print(f'   NaN por columna: {data_with_nan.isnull().sum().sum()} total')
    
    # Crear predictor y entrenar
    predictor = ChurnPredictor()
    
    print('\n🔧 Creando preprocesador...')
    preprocessor = predictor.create_preprocessor(data_with_nan)
    
    print('\n🤖 Creando modelos...')
    models = predictor.create_models()
    
    print('\n🎯 ENTRENANDO MODELOS (con datos que tienen NaN):')
    try:
        predictor.train_models(data_with_nan, target_y)
        print('\n🎉 ¡ÉXITO! Todos los modelos entrenaron correctamente')
        print('   ✅ Los valores NaN fueron manejados automáticamente')
        print('   ✅ No hubo errores de ValueError')
        print('   ✅ Los modelos están listos para hacer predicciones')
        
    except Exception as e:
        print(f'\n❌ Error inesperado: {e}')
        
    return predictor

def real_case_example():
    """
    Simula un caso real del dataset Telco con problemas típicos
    """
    print('\n📋 CASO REAL: DATASET TELCO CON PROBLEMAS TÍPICOS')
    print('=' * 60)
    
    # Simular problemas típicos del dataset Telco
    problematic_telco = pd.DataFrame({
        'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK', '7795-CFOCW'],
        'tenure': [1, 34, 2, np.nan],  # NaN común
        'MonthlyCharges': [29.85, 56.95, 53.85, 42.30],
        'TotalCharges': [29.85, ' ', 108.15, np.nan],  # Espacios → NaN
        'gender': ['Female', 'Male', 'Male', np.nan],
        'InternetService': ['DSL', 'DSL', np.nan, 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes', 'Yes', np.nan],
        'StreamingTV': ['No', 'No', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', np.nan, 'Month-to-month']
    })
    
    print('🚨 Problemas típicos encontrados:')
    print('   • TotalCharges con espacios en blanco')
    print('   • Valores NaN en columnas numéricas y categóricas')
    print('   • Mezcla de tipos de datos')
    
    # Convertir TotalCharges (simular el problema real)
    problematic_telco['TotalCharges'] = pd.to_numeric(
        problematic_telco['TotalCharges'], errors='coerce'
    )
    
    print(f'\n📊 Dataset después de conversión numérica:')
    print(problematic_telco)
    print(f'\nValores NaN por columna:')
    print(problematic_telco.isnull().sum())
    
    # Usar el predictor
    predictor = ChurnPredictor()
    print('\n🔍 Diagnóstico automático:')
    predictor.diagnose_missing_values(problematic_telco.drop('customerID', axis=1))
    
    return problematic_telco

def main():
    """
    Función principal que ejecuta todas las demostraciones
    """
    print('🎯 SCRIPT DE DEMOSTRACIÓN: MANEJO AUTOMÁTICO DE VALORES NaN')
    print('=' * 70)
    print('Este script te muestra exactamente qué significa que el pipeline')
    print('maneja automáticamente los valores NaN en Machine Learning.\n')
    
    # Explicación principal
    transformed_X, original_data = explain_nan_management()
    
    # Demostración con modelos
    predictor = demonstrate_with_models()
    
    # Caso real
    telco_data = real_case_example()
    
    print('\n🎉 RESUMEN FINAL')
    print('=' * 20)
    print('✅ Los valores NaN se manejan AUTOMÁTICAMENTE significa:')
    print('   1. No necesitas escribir código de limpieza')
    print('   2. No necesitas imputer manualmente')
    print('   3. No necesitas preocuparte por errores de NaN')
    print('   4. El pipeline se encarga de todo transparentemente')
    print('   5. Solo escribes: model.fit(X, y) y funciona')
    print('')
    print('💡 Estrategias automáticas aplicadas:')
    print('   • Numéricas: NaN → mediana')
    print('   • Categóricas: NaN → valor más frecuente')
    print('   • Binarias: NaN → valor más frecuente')
    print('')
    print('🚀 ¡Tu código de ML funciona sin cambios!')

if __name__ == "__main__":
    main()
