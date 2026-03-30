"""
TALLER SEMANA 03: Computación de Alto Rendimiento (HPC) en el Agro
===================================================================
Implementa operaciones vectorizadas con NumPy para procesar 
grandes volúmenes de datos agrícolas.
"""

import numpy as np


# ============================================================================
# TAREA 1: Limpieza de datos con np.clip
# ============================================================================

def limpiar_datos(datos, minimo=0, maximo=100):
    """
    Limpia los datos de humedad eliminando valores fuera de rango.
    
    Args:
        datos: Array numpy con valores de humedad
        minimo: Valor mínimo permitido (default: 0)
        maximo: Valor máximo permitido (default: 100)
    
    Returns:
        Array numpy con valores clipados entre minimo y maximo
    """
    return np.clip(datos, minimo, maximo)


# ============================================================================
# TAREA 2: Detección de sequía con máscaras booleanas
# ============================================================================

def detectar_sequia(datos, umbral=30):
    """
    Detecta zonas con condición de sequía (humedad < umbral).
    
    Args:
        datos: Array numpy con valores de humedad
        umbral: Valor de humedad bajo el cual se considera sequía (default: 30)
    
    Returns:
        Tupla con (mascara_booleana, conteo_de_zonas_secas)
    """
    mascara = datos < umbral
    conteo = np.sum(mascara)
    return mascara, int(conteo)


# ============================================================================
# TAREA 3: Aplicación de riego con np.where
# ============================================================================

def aplicar_riego(datos, umbral_seco=30, umbral_humedo=80):
    """
    Clasifica las zonas y recomienda acción de riego usando np.where.
    
    Args:
        datos: Array numpy con valores de humedad
        umbral_seco: Humedad bajo la cual se necesita riego (default: 30)
        umbral_humedo: Humedad sobre la cual se necesita drenaje (default: 80)
    
    Returns:
        Array numpy con clasificación: 'RIEGO', 'DRENAJE', o 'OK'
    """
    return np.where(
        datos < umbral_seco, 'RIEGO',
        np.where(datos > umbral_humedo, 'DRENAJE', 'OK')
    )


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    print("=" * 60)
    print("TALLER SEMANA 03: HPC EN EL AGRO")
    print("=" * 60)
    
    # Cargar datos
    print("\n[1/4] Cargando datos de humedad_finca.npy...")
    datos = np.load('humedad_finca.npy')
    print(f"  ✓ Matriz cargada: {datos.shape}")
    print(f"  ✓ Total de lecturas: {datos.size}")
    
    # Tarea 1: Limpieza
    print("\n[2/4] Limpiando datos con np.clip...")
    datos_limpios = limpiar_datos(datos)
    print(f"  ✓ Valores fuera de rango corregidos")
    print(f"  ✓ Rango actual: [{datos_limpios.min():.2f}, {datos_limpios.max():.2f}]")
    
    # Tarea 2: Detección de sequía
    print("\n[3/4] Detectando zonas de sequía...")
    mascara, conteo = detectar_sequia(datos_limpios)
    porcentaje = (conteo / datos_limpios.size) * 100
    print(f"  ⚠ Zonas en sequía: {conteo} ({porcentaje:.2f}%)")
    
    # Tarea 3: Aplicación de riego
    print("\n[4/4] Clasificando zonas para riego...")
    clasificacion = aplicar_riego(datos_limpios)
    unico, conteos = np.unique(clasificacion, return_counts=True)
    
    print("\n  📊 Resumen de clasificación:")
    for tipo, count in zip(unico, conteos):
        pct = (count / datos_limpios.size) * 100
        print(f"    {tipo}: {count} lecturas ({pct:.2f}%)")
    
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPLETADO")
    print("=" * 60)
    
    return {
        'datos_limpios': datos_limpios,
        'zonas_sequia': conteo,
        'clasificacion': clasificacion
    }


if __name__ == "__main__":
    main()