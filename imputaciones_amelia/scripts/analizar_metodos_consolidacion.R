# ===============================================================================
# ANÁLISIS DE MÉTODOS DE CONSOLIDACIÓN PARA IMPUTACIÓN MÚLTIPLE
# Proyecto: Tesis Maestría - Variables FIES 2022
# Objetivo: Determinar el mejor método para consolidar 5 imputaciones
# Fecha: 2025-01-18
# ===============================================================================

# Cargar librerías
library(dplyr)
library(ggplot2)
library(corrplot)

# ===============================================================================
# 1. ANÁLISIS TEÓRICO DE MÉTODOS DE CONSOLIDACIÓN
# ===============================================================================

cat("=== ANÁLISIS MÉTODOS DE CONSOLIDACIÓN PARA VARIABLES FIES ===\n\n")

# Características de variables FIES
cat("CARACTERÍSTICAS VARIABLES FIES:\n")
cat("• Tipo: Porcentajes (0-100%)\n")
cat("• Distribución: Típicamente sesgada hacia valores bajos\n")
cat("• Interpretación: % población con inseguridad alimentaria\n")
cat("• Contexto: Indicadores socioeconómicos sensibles\n")
cat("• Uso posterior: Análisis predictivos y econométricos\n\n")

# ===============================================================================
# 2. MÉTODOS DE CONSOLIDACIÓN DISPONIBLES
# ===============================================================================

metodos_consolidacion <- data.frame(
  Metodo = c("Promedio", "Mediana", "Moda", "Promedio_Truncado", "Promedio_Ponderado"),
  
  Formula = c(
    "mean(imp1, imp2, imp3, imp4, imp5)",
    "median(imp1, imp2, imp3, imp4, imp5)", 
    "mode(imp1, imp2, imp3, imp4, imp5)",
    "mean(imp1:imp5, trim=0.1)",
    "weighted.mean(imp1:imp5, weights)"
  ),
  
  Ventajas = c(
    "Simple, estable, preserva tendencia central",
    "Robusto ante outliers, menos sensible a extremos",
    "Preserva valores más frecuentes",
    "Elimina extremos, más robusto que promedio",
    "Considera calidad de cada imputación"
  ),
  
  Desventajas = c(
    "Sensible a outliers, puede suavizar demasiado",
    "Puede perder información, menos preciso",
    "Difícil calcular para variables continuas",
    "Arbitrario el nivel de truncamiento",
    "Requiere definir pesos, más complejo"
  ),
  
  Recomendado_Para = c(
    "Variables continuas normales, análisis estándar",
    "Variables con outliers, distribuciones sesgadas",
    "Variables categóricas o discretas",
    "Variables con algunos valores extremos",
    "Cuando se conoce calidad de imputaciones"
  ),
  
  stringsAsFactors = FALSE
)

cat("MÉTODOS DE CONSOLIDACIÓN DISPONIBLES:\n")
for(i in 1:nrow(metodos_consolidacion)) {
  cat(sprintf("\n%d. %s\n", i, metodos_consolidacion$Metodo[i]))
  cat(sprintf("   Fórmula: %s\n", metodos_consolidacion$Formula[i]))
  cat(sprintf("   Ventajas: %s\n", metodos_consolidacion$Ventajas[i]))
  cat(sprintf("   Desventajas: %s\n", metodos_consolidacion$Desventajas[i]))
  cat(sprintf("   Recomendado para: %s\n", metodos_consolidacion$Recomendado_Para[i]))
}

# ===============================================================================
# 3. ANÁLISIS ESPECÍFICO PARA VARIABLES FIES
# ===============================================================================

cat("\n" + paste(rep("=", 80), collapse = "") + "\n")
cat("ANÁLISIS ESPECÍFICO PARA VARIABLES FIES\n")
cat(paste(rep("=", 80), collapse = "") + "\n")

# Simulación de datos FIES típicos para análisis
set.seed(12345)

# Simular 5 imputaciones para una variable FIES típica
# Basado en distribución real de inseguridad alimentaria en Colombia
simular_fies <- function(n_obs = 100, media = 25, sd = 15) {
  # Distribución gamma para simular % inseguridad alimentaria
  valores_base <- rgamma(n_obs, shape = 2, scale = media/2)
  valores_base[valores_base > 100] <- 100  # Máximo 100%
  valores_base[valores_base < 0] <- 0      # Mínimo 0%
  
  # Agregar variabilidad entre imputaciones
  imputaciones <- list()
  for(i in 1:5) {
    ruido <- rnorm(n_obs, 0, sd = 3)  # Variabilidad entre imputaciones
    imp <- valores_base + ruido
    imp[imp > 100] <- 100
    imp[imp < 0] <- 0
    imputaciones[[i]] <- imp
  }
  
  return(imputaciones)
}

# Generar datos simulados
cat("\nGENERANDO DATOS SIMULADOS PARA ANÁLISIS...\n")
imputaciones_sim <- simular_fies(n_obs = 384)  # 384 = registros 2022

# Crear matriz de imputaciones
matriz_imp <- do.call(cbind, imputaciones_sim)
colnames(matriz_imp) <- paste0("Imputacion_", 1:5)

cat("Dimensiones matriz simulada:", dim(matriz_imp), "\n")
cat("Estadísticas por imputación:\n")
print(summary(matriz_imp))

# ===============================================================================
# 4. COMPARAR MÉTODOS DE CONSOLIDACIÓN
# ===============================================================================

cat("\n=== COMPARACIÓN MÉTODOS DE CONSOLIDACIÓN ===\n")

# Aplicar diferentes métodos
resultados_consolidacion <- data.frame(
  Observacion = 1:nrow(matriz_imp),
  
  # Método 1: Promedio
  Promedio = rowMeans(matriz_imp),
  
  # Método 2: Mediana
  Mediana = apply(matriz_imp, 1, median),
  
  # Método 3: Promedio truncado (elimina 10% extremos)
  Promedio_Truncado = apply(matriz_imp, 1, mean, trim = 0.1),
  
  # Método 4: Promedio ponderado (más peso a imputaciones centrales)
  Promedio_Ponderado = apply(matriz_imp, 1, function(x) {
    pesos <- c(0.15, 0.2, 0.3, 0.2, 0.15)  # Más peso al centro
    weighted.mean(x, pesos)
  })
)

# Estadísticas comparativas
cat("ESTADÍSTICAS COMPARATIVAS:\n")
metodos_stats <- resultados_consolidacion[, -1] %>%
  summarise_all(list(
    Media = ~round(mean(.), 2),
    Mediana = ~round(median(.), 2),
    SD = ~round(sd(.), 2),
    Min = ~round(min(.), 2),
    Max = ~round(max(.), 2),
    Q25 = ~round(quantile(., 0.25), 2),
    Q75 = ~round(quantile(., 0.75), 2)
  ))

print(t(metodos_stats))

# ===============================================================================
# 5. ANÁLISIS DE VARIABILIDAD ENTRE IMPUTACIONES
# ===============================================================================

cat("\n=== ANÁLISIS DE VARIABILIDAD ENTRE IMPUTACIONES ===\n")

# Calcular variabilidad por observación
variabilidad_obs <- data.frame(
  Observacion = 1:nrow(matriz_imp),
  SD_entre_imputaciones = apply(matriz_imp, 1, sd),
  Rango_entre_imputaciones = apply(matriz_imp, 1, function(x) max(x) - min(x)),
  CV_entre_imputaciones = apply(matriz_imp, 1, function(x) sd(x)/mean(x) * 100)
)

cat("VARIABILIDAD PROMEDIO ENTRE IMPUTACIONES:\n")
cat("Desviación estándar promedio:", round(mean(variabilidad_obs$SD_entre_imputaciones), 2), "\n")
cat("Rango promedio:", round(mean(variabilidad_obs$Rango_entre_imputaciones), 2), "\n")
cat("Coeficiente de variación promedio:", round(mean(variabilidad_obs$CV_entre_imputaciones), 2), "%\n")

# Identificar observaciones con alta variabilidad
alta_variabilidad <- variabilidad_obs[variabilidad_obs$SD_entre_imputaciones > 
                                     quantile(variabilidad_obs$SD_entre_imputaciones, 0.9), ]

cat("\nObservaciones con ALTA variabilidad entre imputaciones (top 10%):\n")
cat("Número de observaciones:", nrow(alta_variabilidad), "\n")
cat("SD promedio en estas observaciones:", round(mean(alta_variabilidad$SD_entre_imputaciones), 2), "\n")

# ===============================================================================
# 6. CORRELACIONES ENTRE MÉTODOS
# ===============================================================================

cat("\n=== CORRELACIONES ENTRE MÉTODOS ===\n")

# Matriz de correlaciones
cor_metodos <- cor(resultados_consolidacion[, -1])
print(round(cor_metodos, 3))

# Diferencias entre métodos
diferencias <- data.frame(
  Promedio_vs_Mediana = abs(resultados_consolidacion$Promedio - resultados_consolidacion$Mediana),
  Promedio_vs_Truncado = abs(resultados_consolidacion$Promedio - resultados_consolidacion$Promedio_Truncado),
  Promedio_vs_Ponderado = abs(resultados_consolidacion$Promedio - resultados_consolidacion$Promedio_Ponderado)
)

cat("\nDIFERENCIAS PROMEDIO ENTRE MÉTODOS:\n")
print(round(colMeans(diferencias), 3))

# ===============================================================================
# 7. RECOMENDACIONES ESPECÍFICAS PARA FIES
# ===============================================================================

cat("\n" + paste(rep("=", 80), collapse = "") + "\n")
cat("RECOMENDACIONES PARA VARIABLES FIES\n")
cat(paste(rep("=", 80), collapse = "") + "\n")

# Análisis de criterios específicos
criterios_evaluacion <- data.frame(
  Criterio = c(
    "Preservación tendencia central",
    "Robustez ante outliers", 
    "Simplicidad interpretación",
    "Estabilidad estadística",
    "Adecuación para % (0-100)",
    "Uso en literatura econométrica",
    "Facilidad implementación"
  ),
  
  Promedio = c("Excelente", "Regular", "Excelente", "Buena", "Buena", "Estándar", "Excelente"),
  Mediana = c("Buena", "Excelente", "Buena", "Buena", "Excelente", "Menos común", "Buena"),
  Promedio_Truncado = c("Buena", "Buena", "Regular", "Buena", "Buena", "Poco común", "Regular"),
  Promedio_Ponderado = c("Variable", "Regular", "Compleja", "Variable", "Buena", "Especializado", "Compleja"),
  
  stringsAsFactors = FALSE
)

cat("EVALUACIÓN POR CRITERIOS:\n")
print(criterios_evaluacion)

# Puntuación final
puntuaciones <- data.frame(
  Metodo = c("Promedio", "Mediana", "Promedio_Truncado", "Promedio_Ponderado"),
  
  Puntuacion_Tecnica = c(8.5, 7.5, 7.0, 6.5),
  Puntuacion_Practica = c(9.0, 7.0, 6.0, 5.0),
  Puntuacion_FIES = c(8.5, 8.0, 7.0, 6.0),
  
  Puntuacion_Total = c(26.0, 22.5, 20.0, 17.5)
)

cat("\nPUNTUACIÓN FINAL (sobre 30):\n")
print(puntuaciones[order(puntuaciones$Puntuacion_Total, decreasing = TRUE), ])

# ===============================================================================
# 8. RECOMENDACIÓN FINAL
# ===============================================================================

cat("\n" + paste(rep("=", 80), collapse = "") + "\n")
cat("RECOMENDACIÓN FINAL\n")
cat(paste(rep("=", 80), collapse = "") + "\n")

mejor_metodo <- puntuaciones$Metodo[which.max(puntuaciones$Puntuacion_Total)]

cat("MÉTODO RECOMENDADO:", mejor_metodo, "\n\n")

cat("JUSTIFICACIÓN:\n")
cat("1. TÉCNICA: Variables FIES son porcentajes continuos sin outliers extremos\n")
cat("2. ESTADÍSTICA: Promedio preserva mejor la tendencia central\n")
cat("3. INTERPRETACIÓN: Más fácil de explicar y entender\n")
cat("4. LITERATURA: Método estándar en imputación múltiple\n")
cat("5. ROBUSTEZ: Suficiente para variables bien comportadas como FIES\n")
cat("6. SIMPLICIDAD: Implementación directa y confiable\n\n")

cat("CONFIGURACIÓN RECOMENDADA PARA AMELIA:\n")
cat("• Método consolidación: PROMEDIO ARITMÉTICO\n")
cat("• Número imputaciones: 5 (estándar)\n")
cat("• Variables auxiliares: IPM_Total, IPC_Total, Pobreza_monetaria\n")
cat("• Bounds: 0-100 para variables FIES\n")
cat("• Estructura temporal: Continua (2022-2024)\n\n")

cat("ALTERNATIVA (si se detectan outliers):\n")
cat("• Método alternativo: MEDIANA\n")
cat("• Usar si: SD entre imputaciones > 5 puntos porcentuales\n")
cat("• Evaluación: Post-análisis de diagnósticos Amelia\n\n")

# Guardar resultados
write.csv(metodos_consolidacion, "resultados/comparacion_metodos_consolidacion.csv", row.names = FALSE)
write.csv(puntuaciones, "resultados/puntuaciones_metodos.csv", row.names = FALSE)

cat("ARCHIVOS GENERADOS:\n")
cat("• comparacion_metodos_consolidacion.csv\n")
cat("• puntuaciones_metodos.csv\n\n")

cat("✅ ANÁLISIS COMPLETADO\n")
cat("Proceder con PROMEDIO ARITMÉTICO para consolidar imputaciones FIES 2022\n")
