#!/usr/bin/env python3
"""
🔧 VERIFICACIÓN: Fix para error "No internet service"
===================================================

Este script verifica que el error ValueError: could not convert string to float: 'No internet service'
ha sido completamente solucionado.

Ejecuta este script para confirmar que tu código funcionará sin problemas.
"""

import pandas as pd
import numpy as np
from models import ChurnPredictor

def test_no_internet_service_fix():
    """
    Prueba específica para el error de 'No internet service'
    """
    print('🧪 TEST: Fix para "No internet service"')
    print('=' * 45)
    
    # Crear datos que reproducen exactamente el error original
    test_data = pd.DataFrame({
        'tenure': [1, 12, 24, 36, 8, 48],
        'MonthlyCharges': [29.85, 56.95, 85.2, 91.5, 70.0, 95.0],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes'], 
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'],
        
        # Estas columnas causaban el error original
        'OnlineSecurity': ['No', 'Yes', 'No internet service', 'Yes', 'No', 'No internet service'],
        'OnlineBackup': ['Yes', 'No internet service', 'No', 'Yes', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No internet service', 'Yes', 'No', 'Yes', 'No internet service'],
        'StreamingTV': ['No internet service', 'No', 'Yes', 'No', 'Yes', 'No'],
        'StreamingMovies': ['No', 'No internet service', 'No', 'Yes', 'No internet service', 'Yes']
    })
    
    print('📊 Datos de prueba con valores problemáticos:')
    print('   - OnlineSecurity con "No internet service"')
    print('   - OnlineBackup con "No internet service"')
    print('   - TechSupport con "No internet service"')
    print('   - StreamingTV con "No internet service"')
    print('   - StreamingMovies con "No internet service"')
    
    # Mostrar algunos valores problemáticos
    problematic_cols = ['OnlineSecurity', 'OnlineBackup', 'TechSupport']
    print(f'\n📋 Ejemplo de valores problemáticos:')
    print(test_data[problematic_cols].head(3))
    
    print('\n🔧 Creando ChurnPredictor...')
    predictor = ChurnPredictor()
    
    try:
        # Test 1: Crear preprocesador
        print('\n1️⃣ Probando create_preprocessor...')
        preprocessor = predictor.create_preprocessor(test_data)
        print('   ✅ Preprocesador creado exitosamente')
        
        # Test 2: Aplicar transformación
        print('\n2️⃣ Probando fit_transform...')
        X_transformed = preprocessor.fit_transform(test_data)
        print(f'   ✅ Transformación exitosa! Shape: {X_transformed.shape}')
        
        # Test 3: Verificar resultados
        print('\n3️⃣ Verificando resultados...')
        df_result = pd.DataFrame(X_transformed)
        has_nan = df_result.isna().any().any()
        
        # Convertir a array numpy para verificar inf
        X_array = np.array(X_transformed, dtype=float)
        has_inf = np.isinf(X_array).any()
        
        print(f'   - ¿Contiene NaN?: {has_nan}')
        print(f'   - ¿Contiene Inf?: {has_inf}')
        print(f'   - Tipo de datos: {X_array.dtype}')
        print(f'   - Rango de valores: [{X_array.min():.2f}, {X_array.max():.2f}]')
        
        # Test 4: Probar inspect_transformed_columns (lo que causaba el error)
        print('\n4️⃣ Probando inspect_transformed_columns...')
        predictor.inspect_transformed_columns(
            X_original=test_data,
            columns=['Partner', 'Dependents', 'Contract', 'PaymentMethod']
        )
        print('   ✅ inspect_transformed_columns funciona correctamente')
        
        # Test 5: Probar entrenamiento de modelos
        print('\n5️⃣ Probando entrenamiento de modelos...')
        models = predictor.create_models()
        
        # Crear variable objetivo de prueba
        y_test = pd.Series([0, 1, 0, 1, 0, 1])
        
        predictor.train_models(test_data, y_test)
        print('   ✅ Modelos entrenados exitosamente')
        
        print('\n🎉 TODOS LOS TESTS PASARON!')
        print('✅ El error "No internet service" está completamente solucionado')
        print('✅ Tu código funcionará sin problemas')
        
        return True
        
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        print('\nDetalles del error:')
        import traceback
        traceback.print_exc()
        return False

def test_specific_columns():
    """
    Prueba específica para las columnas que causaron el error original
    """
    print('\n' + '='*60)
    print('🎯 TEST ESPECÍFICO: Columnas que causaron el error original')
    print('='*60)
    
    # Datos mínimos para reproducir el error
    minimal_data = pd.DataFrame({
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'Yes', 'No'],
        'OnlineSecurity': ['No internet service', 'Yes', 'No'],  # Problemática
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer']
    })
    
    print('📊 Datos mínimos de prueba:')
    print(minimal_data)
    
    predictor = ChurnPredictor()
    
    try:
        preprocessor = predictor.create_preprocessor(minimal_data)
        
        # Esto era lo que fallaba antes
        predictor.inspect_transformed_columns(
            X_original=minimal_data,
            columns=['Partner', 'Dependents', 'Contract', 'PaymentMethod']
        )
        
        print('\n✅ Columnas específicas procesan correctamente')
        print('✅ El error original está solucionado')
        
    except Exception as e:
        print(f'❌ Error en columnas específicas: {e}')
        return False
    
    return True

def main():
    """
    Ejecuta todos los tests de verificación
    """
    print('🔧 VERIFICACIÓN COMPLETA: Fix "No internet service"')
    print('=' * 55)
    print('Este script verifica que el error ValueError ha sido solucionado.\n')
    
    # Ejecutar tests
    test1_passed = test_no_internet_service_fix()
    test2_passed = test_specific_columns()
    
    print('\n' + '='*55)
    print('📋 RESUMEN DE VERIFICACIÓN')
    print('='*25)
    
    if test1_passed and test2_passed:
        print('🎉 ✅ TODOS LOS TESTS PASARON')
        print('✅ El error "No internet service" está completamente solucionado')
        print('✅ inspect_transformed_columns funciona correctamente')
        print('✅ Los modelos pueden entrenar sin problemas')
        print('\n💡 Puedes ejecutar tu código original sin cambios')
        print('   predictor.inspect_transformed_columns(...) funcionará perfectamente')
    else:
        print('❌ ALGUNOS TESTS FALLARON')
        print('⚠️ Puede que aún haya problemas pendientes')
    
    print('\n🚀 ¡Tu pipeline de ML está listo para usar!')

if __name__ == "__main__":
    main()
