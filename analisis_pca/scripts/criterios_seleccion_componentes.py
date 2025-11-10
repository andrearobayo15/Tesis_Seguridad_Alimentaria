"""
Criterios Estadísticos para Selección de Componentes Principales
Análisis detallado para determinar número óptimo de componentes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def revisar_criterios_estadisticos():
    print("=" * 60)
    print("CRITERIOS ESTADÍSTICOS PARA SELECCIÓN DE COMPONENTES")
    print("=" * 60)
    
    # Cargar datos originales para re-ejecutar PCA
    archivo = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
    df = pd.read_csv(archivo)
    
    # Preparar datos
    variables_excluir = ['departamento', 'año', 'mes', 'fecha', 'clave']
    variables_numericas = [col for col in df.columns if col not in variables_excluir]
    X = df[variables_numericas].copy()
    
    # Imputar y estandarizar
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # PCA completo
    pca = PCA()
    pca.fit(X_scaled)
    
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)
    eigenvalues = pca.explained_variance_
    
    print(f"Variables originales: {len(variables_numericas)}")
    print(f"Componentes calculados: {len(varianza_explicada)}")
    
    return pca, varianza_explicada, varianza_acumulada, eigenvalues

def aplicar_criterios_seleccion(varianza_explicada, varianza_acumulada, eigenvalues):
    print(f"\n" + "=" * 50)
    print("APLICACIÓN DE CRITERIOS ESTADÍSTICOS")
    print("=" * 50)
    
    # CRITERIO 1: Regla de Kaiser (Eigenvalues > 1)
    kaiser_components = np.sum(eigenvalues > 1)
    print(f"\n1. CRITERIO DE KAISER (Eigenvalues > 1):")
    print(f"   Componentes recomendados: {kaiser_components}")
    print(f"   Eigenvalues > 1:")
    for i, eigenval in enumerate(eigenvalues):
        if eigenval > 1:
            print(f"     PC{i+1}: {eigenval:.3f}")
        else:
            break
    
    # CRITERIO 2: Varianza Explicada Acumulada
    idx_70 = np.where(varianza_acumulada >= 0.70)[0][0] + 1
    idx_80 = np.where(varianza_acumulada >= 0.80)[0][0] + 1
    idx_90 = np.where(varianza_acumulada >= 0.90)[0][0] + 1
    
    print(f"\n2. CRITERIO DE VARIANZA EXPLICADA:")
    print(f"   70% varianza: {idx_70} componentes ({varianza_acumulada[idx_70-1]:.3f})")
    print(f"   80% varianza: {idx_80} componentes ({varianza_acumulada[idx_80-1]:.3f})")
    print(f"   90% varianza: {idx_90} componentes ({varianza_acumulada[idx_90-1]:.3f})")
    
    # CRITERIO 3: Regla del Codo (Scree Test)
    diferencias = np.diff(varianza_explicada)
    diferencias_2 = np.diff(diferencias)  # Segunda derivada
    
    # Buscar el punto donde la pendiente se estabiliza
    threshold = 0.01  # Umbral para considerar estabilización
    codo_idx = 1
    for i in range(1, len(diferencias)):
        if abs(diferencias[i]) < threshold:
            codo_idx = i + 1
            break
    
    print(f"\n3. CRITERIO DEL CODO (Scree Test):")
    print(f"   Componentes recomendados: ~{codo_idx}")
    print(f"   Diferencias en varianza explicada:")
    for i in range(min(10, len(diferencias))):
        print(f"     PC{i+1} -> PC{i+2}: {diferencias[i]:.4f}")
    
    # CRITERIO 4: Análisis de Componentes Individuales
    print(f"\n4. ANÁLISIS INDIVIDUAL DE COMPONENTES:")
    print(f"   Componentes con varianza > 5%:")
    componentes_significativos = 0
    for i, var in enumerate(varianza_explicada):
        if var > 0.05:  # 5% de varianza
            componentes_significativos += 1
            print(f"     PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        else:
            break
    
    return kaiser_components, idx_70, idx_80, idx_90, codo_idx, componentes_significativos

def analizar_variables_objetivo():
    print(f"\n" + "=" * 50)
    print("ANÁLISIS DE VARIABLES OBJETIVO")
    print("=" * 50)
    
    # Cargar datos originales para ver las variables FIES
    archivo = "d:/Tesis maestria/Tesis codigo/imputaciones_amelia/resultados/BASE_MASTER_FINAL_TESIS.csv"
    df = pd.read_csv(archivo)
    
    # Verificar variables FIES objetivo
    variables_fies = [col for col in df.columns if 'FIES' in col]
    print(f"Variables FIES disponibles:")
    for var in variables_fies:
        print(f"  - {var}")
    
    # Verificar si existen las variables objetivo
    objetivo_1 = 'FIES_moderado_grave'
    objetivo_2 = 'FIES_grave'
    
    if objetivo_1 in df.columns:
        print(f"\n✓ {objetivo_1} encontrada")
        # Estadísticas básicas
        stats_1 = df[objetivo_1].describe()
        print(f"  Estadísticas: min={stats_1['min']:.1f}, max={stats_1['max']:.1f}, mean={stats_1['mean']:.1f}")
    else:
        print(f"\n✗ {objetivo_1} NO encontrada")
    
    if objetivo_2 in df.columns:
        print(f"✓ {objetivo_2} encontrada")
        stats_2 = df[objetivo_2].describe()
        print(f"  Estadísticas: min={stats_2['min']:.1f}, max={stats_2['max']:.1f}, mean={stats_2['mean']:.1f}")
    else:
        print(f"✗ {objetivo_2} NO encontrada")
    
    return variables_fies, objetivo_1 in df.columns, objetivo_2 in df.columns

def recomendar_numero_componentes(kaiser_components, idx_70, idx_80, idx_90, codo_idx, componentes_significativos):
    print(f"\n" + "=" * 60)
    print("RECOMENDACIÓN FINAL DE COMPONENTES")
    print("=" * 60)
    
    criterios = {
        'Kaiser (Eigenvalues > 1)': kaiser_components,
        'Varianza 70%': idx_70,
        'Varianza 80%': idx_80,
        'Varianza 90%': idx_90,
        'Regla del Codo': codo_idx,
        'Componentes > 5% varianza': componentes_significativos
    }
    
    print("RESUMEN DE CRITERIOS:")
    for criterio, valor in criterios.items():
        print(f"  - {criterio}: {valor} componentes")
    
    # Análisis de consenso
    valores = list(criterios.values())
    componentes_comunes = []
    
    # Buscar valores que aparecen en múltiples criterios
    for i in range(1, 16):  # Revisar hasta 15 componentes
        count = valores.count(i)
        if count >= 2:
            componentes_comunes.append((i, count))
    
    print(f"\nCONSENSO ENTRE CRITERIOS:")
    if componentes_comunes:
        componentes_comunes.sort(key=lambda x: x[1], reverse=True)
        for comp, count in componentes_comunes:
            print(f"  - {comp} componentes: {count} criterios coinciden")
    
    # Recomendación basada en literatura estadística
    print(f"\nRECOMENDACIÓN ESTADÍSTICA:")
    
    # Para modelos predictivos, se recomienda:
    # 1. Kaiser como mínimo
    # 2. 80% varianza como objetivo estándar
    # 3. Considerar interpretabilidad
    
    recomendacion_conservadora = kaiser_components
    recomendacion_balanceada = idx_80
    recomendacion_completa = idx_90
    
    print(f"  - CONSERVADORA (Kaiser): {recomendacion_conservadora} componentes")
    print(f"  - BALANCEADA (80% varianza): {recomendacion_balanceada} componentes")
    print(f"  - COMPLETA (90% varianza): {recomendacion_completa} componentes")
    
    print(f"\nRECOMENDACIÓN FINAL PARA MODELADO:")
    print(f"  Usar {recomendacion_balanceada} componentes principales (PC1-PC{recomendacion_balanceada})")
    print(f"  Justificación:")
    print(f"    - Captura {varianza_acumulada[recomendacion_balanceada-1]*100:.1f}% de la varianza")
    print(f"    - Cumple criterio estándar de 80% varianza explicada")
    print(f"    - Balance entre reducción dimensional y retención de información")
    print(f"    - Evita sobreajuste manteniendo interpretabilidad")
    
    return recomendacion_balanceada

def crear_visualizacion_criterios(varianza_explicada, varianza_acumulada, eigenvalues, kaiser_components, idx_80):
    print(f"\n" + "=" * 40)
    print("CREANDO VISUALIZACIONES")
    print("=" * 40)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Scree Plot
    n_show = min(15, len(varianza_explicada))
    componentes = range(1, n_show + 1)
    
    ax1.plot(componentes, varianza_explicada[:n_show], 'bo-', linewidth=2, markersize=6)
    ax1.axvline(x=kaiser_components, color='r', linestyle='--', alpha=0.7, label=f'Kaiser ({kaiser_components})')
    ax1.axvline(x=idx_80, color='g', linestyle='--', alpha=0.7, label=f'80% varianza ({idx_80})')
    ax1.set_xlabel('Componente Principal')
    ax1.set_ylabel('Varianza Explicada')
    ax1.set_title('Scree Plot - Criterio del Codo')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Varianza Acumulada
    ax2.plot(componentes, varianza_acumulada[:n_show], 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='80% varianza')
    ax2.axvline(x=idx_80, color='g', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Número de Componentes')
    ax2.set_ylabel('Varianza Acumulada')
    ax2.set_title('Varianza Explicada Acumulada')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Eigenvalues (Criterio Kaiser)
    ax3.bar(componentes, eigenvalues[:n_show], alpha=0.7, color='skyblue')
    ax3.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Kaiser (eigenvalue = 1)')
    ax3.set_xlabel('Componente Principal')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('Criterio de Kaiser - Eigenvalues')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Comparación de Criterios
    criterios_nombres = ['Kaiser', '80% Var', 'Codo']
    criterios_valores = [kaiser_components, idx_80, 5]  # Aproximado para el codo
    
    ax4.bar(criterios_nombres, criterios_valores, alpha=0.7, color=['red', 'green', 'blue'])
    ax4.set_ylabel('Número de Componentes')
    ax4.set_title('Comparación de Criterios')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:/Tesis maestria/Tesis codigo/analisis_pca/resultados/criterios_seleccion_componentes.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualización guardada: criterios_seleccion_componentes.png")

def main():
    # 1. Revisar criterios estadísticos
    pca, varianza_explicada, varianza_acumulada, eigenvalues = revisar_criterios_estadisticos()
    
    # 2. Aplicar criterios de selección
    kaiser_components, idx_70, idx_80, idx_90, codo_idx, componentes_significativos = aplicar_criterios_seleccion(
        varianza_explicada, varianza_acumulada, eigenvalues)
    
    # 3. Analizar variables objetivo
    variables_fies, tiene_moderado_grave, tiene_grave = analizar_variables_objetivo()
    
    # 4. Recomendar número de componentes
    componentes_recomendados = recomendar_numero_componentes(
        kaiser_components, idx_70, idx_80, idx_90, codo_idx, componentes_significativos)
    
    # 5. Crear visualizaciones
    crear_visualizacion_criterios(varianza_explicada, varianza_acumulada, eigenvalues, 
                                 kaiser_components, idx_80)
    
    return componentes_recomendados, tiene_moderado_grave, tiene_grave

if __name__ == "__main__":
    componentes_recomendados, tiene_moderado_grave, tiene_grave = main()
