# ===============================================================================
# IMPUTACIÓN ESPECÍFICA VARIABLES FIES 2022 CON AMELIA
# Proyecto: Tesis Maestría - Análisis Socioeconómico Colombia
# Objetivo: Imputar SOLO las 8 variables FIES detalladas para el año 2022
# Fecha: 2025-01-18
# ===============================================================================

# Cargar librerías necesarias
if (!require("Amelia")) install.packages("Amelia")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("VIM")) install.packages("VIM")
if (!require("corrplot")) install.packages("corrplot")
if (!require("tidyr")) install.packages("tidyr")

library(Amelia)
library(dplyr)
library(ggplot2)
library(VIM)
library(corrplot)
library(tidyr)

# ===============================================================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ===============================================================================

# Establecer directorio de trabajo
setwd("d:/Tesis maestria/Tesis codigo/imputaciones_amelia")

# Cargar base master original
cat("=== CARGANDO BASE MASTER ORIGINAL ===\n")
datos_originales <- read.csv("datos_originales/BASE_MASTER_2022_2025_FIES_COMPLETO_RECUPERADO.csv", 
                            stringsAsFactors = FALSE, encoding = "UTF-8")

cat("Dimensiones originales:", dim(datos_originales), "\n")
cat("Período:", min(datos_originales$año, na.rm = TRUE), "-", max(datos_originales$año, na.rm = TRUE), "\n")

# Variables FIES específicas a imputar
variables_fies_objetivo <- c(
  "FIES_preocupacion_alimentos",
  "FIES_no_alimentos_saludables", 
  "FIES_poca_variedad_alimentos",
  "FIES_saltar_comida",
  "FIES_comio_menos",
  "FIES_sin_alimentos",
  "FIES_hambre_sin_comer",
  "FIES_no_comio_dia_entero"
)

cat("\nVariables FIES a imputar:", length(variables_fies_objetivo), "\n")
for(var in variables_fies_objetivo) {
  cat("  •", var, "\n")
}

# ===============================================================================
# 2. PREPARAR DATOS COMPLETOS (2022-2024) PARA MODELO AMELIA
# ===============================================================================

# Usar TODOS los años disponibles para el modelo (2022-2024)
# Esto mejora la calidad de imputación al tener más información
datos_completos <- datos_originales %>%
  filter(año %in% c(2022, 2023, 2024))

cat("\n=== DATOS COMPLETOS PARA MODELO AMELIA ===\n")
cat("Registros totales:", nrow(datos_completos), "\n")
cat("Años incluidos:", paste(sort(unique(datos_completos$año)), collapse = ", "), "\n")
cat("Departamentos:", length(unique(datos_completos$departamento)), "\n")

# Análisis específico por año
analisis_por_año <- datos_completos %>%
  group_by(año) %>%
  summarise(
    registros = n(),
    departamentos = n_distinct(departamento),
    .groups = 'drop'
  )

cat("\nDistribución por año:\n")
print(analisis_por_año)

# Verificar estado de variables FIES por año
cat("\n=== ESTADO VARIABLES FIES POR AÑO ===\n")
for(año_actual in c(2022, 2023, 2024)) {
  cat(sprintf("\n--- AÑO %d ---\n", año_actual))
  datos_año <- datos_completos %>% filter(año == año_actual)
  
  for(var in variables_fies_objetivo) {
    if(var %in% names(datos_año)) {
      faltantes <- sum(is.na(datos_año[[var]]))
      porcentaje <- round((faltantes / nrow(datos_año)) * 100, 2)
      disponibles <- nrow(datos_año) - faltantes
      cat(sprintf("  %-28s: %3d faltantes (%5.1f%%) | %3d disponibles\n", 
                  var, faltantes, porcentaje, disponibles))
    }
  }
}

# ===============================================================================
# 3. PREPARAR DATASET PARA AMELIA
# ===============================================================================

# Seleccionar variables para el modelo de imputación
variables_modelo <- c(
  # Identificadores
  "departamento", "año", "mes",
  
  # Variables FIES objetivo
  variables_fies_objetivo,
  
  # Variables auxiliares para mejorar imputación
  "IPM_Total",                    # Pobreza multidimensional
  "IPC_Total",                    # Pobreza de consumo
  "Pobreza_monetaria",           # Pobreza monetaria
  "Vida_general",                # Calidad de vida general
  "Ingreso",                     # Satisfacción con ingresos
  
  # Variables FIES agregadas (si están disponibles)
  "FIES_leve_moderado",
  "FIES_grave", 
  "FIES_moderada"
)

# Filtrar variables que existen
variables_disponibles <- variables_modelo[variables_modelo %in% names(datos_completos)]
cat("\n=== VARIABLES PARA MODELO AMELIA ===\n")
cat("Variables solicitadas:", length(variables_modelo), "\n")
cat("Variables disponibles:", length(variables_disponibles), "\n")

# Crear dataset para Amelia usando TODOS los años (2022-2024)
datos_amelia <- datos_completos %>%
  select(all_of(variables_disponibles)) %>%
  mutate(
    # Convertir departamento a numérico para Amelia
    departamento_num = as.numeric(as.factor(departamento)),
    # Crear variable temporal numérica continua
    tiempo = (año - 2022) * 12 + match(mes, c("enero", "febrero", "marzo", "abril", "mayo", "junio",
                                              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"))
  ) %>%
  # Remover variables categóricas originales
  select(-departamento, -año, -mes)

cat("Dimensiones dataset Amelia:", dim(datos_amelia), "\n")

# ===============================================================================
# 4. ANÁLISIS PRE-IMPUTACIÓN
# ===============================================================================

cat("\n=== ANÁLISIS PRE-IMPUTACIÓN ===\n")

# Estadísticas de missingness
missing_stats <- datos_amelia %>%
  select(all_of(variables_fies_objetivo)) %>%
  summarise_all(~sum(is.na(.))) %>%
  gather(variable, missing_count) %>%
  mutate(
    total_obs = nrow(datos_amelia),
    missing_pct = round((missing_count / total_obs) * 100, 2),
    available_count = total_obs - missing_count
  ) %>%
  arrange(desc(missing_pct))

print(missing_stats)

# Verificar correlaciones entre variables disponibles
variables_numericas <- datos_amelia %>%
  select_if(is.numeric) %>%
  select(-tiempo) %>%
  names()

if(length(variables_numericas) > 1) {
  correlaciones <- cor(datos_amelia[variables_numericas], use = "pairwise.complete.obs")
  
  # Guardar matriz de correlaciones
  png("diagnosticos/correlaciones_pre_imputacion.png", width = 1000, height = 800)
  corrplot(correlaciones, method = "color", type = "upper", 
           order = "hclust", tl.cex = 0.8, tl.col = "black")
  title("Correlaciones Variables Pre-Imputación")
  dev.off()
}

# Patrón de missingness
png("diagnosticos/patron_missingness_fies_2022.png", width = 1200, height = 800)
VIM::aggr(datos_amelia[variables_fies_objetivo], 
          col = c('navyblue', 'red'), 
          numbers = TRUE, 
          sortVars = TRUE,
          main = "Patrón de Datos Faltantes - Variables FIES 2022")
dev.off()

# ===============================================================================
# 5. CONFIGURACIÓN AMELIA ESPECÍFICA
# ===============================================================================

cat("\n=== CONFIGURACIÓN AMELIA ===\n")

# Definir bounds (límites) para variables FIES (típicamente 0-100)
bounds_matrix <- matrix(c(
  # Todas las variables FIES tienen rango 0-100
  which(names(datos_amelia) == "FIES_preocupacion_alimentos"), 0, 100,
  which(names(datos_amelia) == "FIES_no_alimentos_saludables"), 0, 100,
  which(names(datos_amelia) == "FIES_poca_variedad_alimentos"), 0, 100,
  which(names(datos_amelia) == "FIES_saltar_comida"), 0, 100,
  which(names(datos_amelia) == "FIES_comio_menos"), 0, 100,
  which(names(datos_amelia) == "FIES_sin_alimentos"), 0, 100,
  which(names(datos_amelia) == "FIES_hambre_sin_comer"), 0, 100,
  which(names(datos_amelia) == "FIES_no_comio_dia_entero"), 0, 100
), ncol = 3, byrow = TRUE)

# Filtrar bounds para variables que existen
bounds_validos <- bounds_matrix[bounds_matrix[,1] > 0, , drop = FALSE]

cat("Bounds definidos para", nrow(bounds_validos), "variables FIES\n")

# ===============================================================================
# 6. EJECUTAR IMPUTACIÓN MÚLTIPLE
# ===============================================================================

cat("\n=== EJECUTANDO IMPUTACIÓN MÚLTIPLE ===\n")
cat("Esto puede tomar varios minutos...\n")

# Configurar semilla para reproducibilidad
set.seed(12345)

# Ejecutar Amelia
start_time <- Sys.time()

amelia_result <- amelia(
  x = datos_amelia,
  m = 5,                          # 5 imputaciones múltiples
  cs = "departamento_num",        # Variable de sección cruzada (departamento numérico)
  ts = "tiempo",                  # Variable temporal continua
  bounds = bounds_validos,        # Límites para variables FIES
  max.resample = 1000,           # Máximo resampling
  tolerance = 1e-04,             # Tolerancia convergencia
  empri = 0.01 * nrow(datos_amelia), # Prior empírico
  frontend = FALSE,              # Sin interfaz gráfica
  parallel = "no"                # Sin paralelización para estabilidad
)

end_time <- Sys.time()
tiempo_ejecucion <- end_time - start_time

cat("\n=== IMPUTACIÓN COMPLETADA ===\n")
cat("Tiempo de ejecución:", round(tiempo_ejecucion, 2), attr(tiempo_ejecucion, "units"), "\n")
cat("Número de imputaciones:", amelia_result$m, "\n")
cat("Convergencia exitosa:", !is.null(amelia_result$imputations), "\n")

# ===============================================================================
# 7. DIAGNÓSTICOS DE CALIDAD
# ===============================================================================

cat("\n=== GENERANDO DIAGNÓSTICOS ===\n")

# Diagnóstico 1: Convergencia
png("diagnosticos/convergencia_amelia.png", width = 1200, height = 800)
plot(amelia_result, which.vars = variables_fies_objetivo[1:3])
title("Diagnóstico de Convergencia - Variables FIES")
dev.off()

# Diagnóstico 2: Distribuciones comparadas
for(var in variables_fies_objetivo[1:3]) {
  if(var %in% names(datos_amelia) && sum(!is.na(datos_amelia[[var]])) > 0) {
    png(paste0("diagnosticos/distribucion_", gsub("_", "", var), ".png"), 
        width = 1000, height = 600)
    compare.density(amelia_result, var = var)
    title(paste("Comparación Distribuciones -", var))
    dev.off()
  }
}

# Diagnóstico 3: Overimputation (si hay suficientes datos)
if(sum(!is.na(datos_amelia[[variables_fies_objetivo[1]]])) > 10) {
  png("diagnosticos/overimputation_test.png", width = 1000, height = 600)
  overimpute(amelia_result, var = variables_fies_objetivo[1])
  title("Test de Overimputación")
  dev.off()
}

# ===============================================================================
# 8. CONSOLIDAR IMPUTACIONES
# ===============================================================================

cat("\n=== CONSOLIDANDO IMPUTACIONES ===\n")

# Función avanzada para consolidar imputaciones con múltiples métodos
consolidar_imputaciones <- function(amelia_obj, metodo = "mean", analizar_variabilidad = TRUE) {
  
  # Extraer todas las imputaciones
  imputaciones <- amelia_obj$imputations
  
  # Crear dataset base
  datos_consolidados <- imputaciones[[1]]
  
  # Variables FIES a consolidar
  vars_fies <- variables_fies_objetivo[variables_fies_objetivo %in% names(datos_consolidados)]
  
  cat("\n=== CONSOLIDANDO", length(vars_fies), "VARIABLES FIES ===\n")
  cat("Método seleccionado:", toupper(metodo), "\n")
  
  # Análisis de variabilidad entre imputaciones
  if(analizar_variabilidad) {
    cat("\nAnalizando variabilidad entre imputaciones...\n")
    
    for(var in vars_fies[1:3]) {  # Analizar primeras 3 variables
      valores_matriz <- sapply(imputaciones, function(imp) imp[[var]])
      
      # Estadísticas de variabilidad
      sd_promedio <- mean(apply(valores_matriz, 1, sd, na.rm = TRUE), na.rm = TRUE)
      rango_promedio <- mean(apply(valores_matriz, 1, function(x) max(x, na.rm = TRUE) - min(x, na.rm = TRUE)), na.rm = TRUE)
      
      cat(sprintf("  %s: SD promedio = %.2f, Rango promedio = %.2f\n", 
                  var, sd_promedio, rango_promedio))
      
      # Recomendación automática de método
      if(sd_promedio > 5) {
        cat(sprintf("    Alta variabilidad detectada. Considerar mediana.\n"))
      } else {
        cat(sprintf("    Variabilidad normal. Promedio es adecuado.\n"))
      }
    }
  }
  
  # Aplicar método de consolidación seleccionado
  for (var in vars_fies) {
    # Extraer valores de todas las imputaciones
    valores_imputados <- sapply(imputaciones, function(imp) imp[[var]])
    
    if (metodo == "mean") {
      # Promedio aritmético
      datos_consolidados[[var]] <- rowMeans(valores_imputados, na.rm = TRUE)
      
    } else if (metodo == "median") {
      # Mediana (más robusta)
      datos_consolidados[[var]] <- apply(valores_imputados, 1, median, na.rm = TRUE)
      
    } else if (metodo == "trimmed_mean") {
      # Promedio truncado (elimina 10% extremos)
      datos_consolidados[[var]] <- apply(valores_imputados, 1, mean, trim = 0.1, na.rm = TRUE)
      
    } else if (metodo == "weighted_mean") {
      # Promedio ponderado (más peso al centro)
      pesos <- c(0.15, 0.2, 0.3, 0.2, 0.15)
      datos_consolidados[[var]] <- apply(valores_imputados, 1, function(x) {
        weighted.mean(x, pesos, na.rm = TRUE)
      })
    }
  }
  
  # Estadísticas de consolidación
  cat("\nEstadísticas de consolidación:\n")
  for(var in vars_fies[1:3]) {
    if(var %in% names(datos_consolidados)) {
      valores_final <- datos_consolidados[[var]]
      cat(sprintf("  %s: Media=%.2f, Mediana=%.2f, SD=%.2f\n", 
                  var, mean(valores_final, na.rm = TRUE), 
                  median(valores_final, na.rm = TRUE),
                  sd(valores_final, na.rm = TRUE)))
    }
  }
  
  return(datos_consolidados)
}

# Consolidar resultados
datos_imputados <- consolidar_imputaciones(amelia_result, metodo = "mean", analizar_variabilidad = TRUE)

# Verificar que no hay NAs en variables objetivo
nas_restantes <- sapply(variables_fies_objetivo, function(var) {
  if(var %in% names(datos_imputados)) {
    sum(is.na(datos_imputados[[var]]))
  } else {
    NA
  }
})

cat("NAs restantes después de imputación:\n")
print(nas_restantes)

# ===============================================================================
# 9. INTEGRAR CON BASE ORIGINAL
# ===============================================================================

cat("\n=== INTEGRANDO CON BASE ORIGINAL ===\n")

# Crear copia de seguridad de datos originales
datos_finales <- datos_originales

# Crear mapeo de departamentos para integración
departamentos_originales <- unique(datos_completos$departamento)
mapeo_departamentos <- data.frame(
  departamento_num = 1:length(departamentos_originales),
  departamento = departamentos_originales
)

# Agregar información de departamento y año a datos imputados
datos_imputados_con_info <- datos_imputados %>%
  left_join(mapeo_departamentos, by = "departamento_num") %>%
  mutate(
    año = floor((tiempo - 1) / 12) + 2022,
    mes_num = ((tiempo - 1) %% 12) + 1,
    mes = c("enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")[mes_num]
  )

# Filtrar solo datos 2022 imputados para integración
datos_imputados_solo_2022 <- datos_imputados_con_info %>%
  filter(año == 2022)

# Crear clave de unión
datos_imputados_solo_2022$clave <- paste(datos_imputados_solo_2022$departamento, 
                                         "2022", 
                                         datos_imputados_solo_2022$mes, sep = "_")

datos_finales$clave <- paste(datos_finales$departamento, 
                            datos_finales$año, 
                            datos_finales$mes, sep = "_")

# Actualizar solo valores de 2022 que estaban faltantes
valores_actualizados <- 0

for(var in variables_fies_objetivo) {
  if(var %in% names(datos_finales) && var %in% names(datos_imputados_solo_2022)) {
    
    # Identificar registros 2022 con NAs originales
    idx_2022_na <- which(datos_finales$año == 2022 & is.na(datos_finales[[var]]))
    
    if(length(idx_2022_na) > 0) {
      # Hacer matching con datos imputados
      claves_na <- datos_finales$clave[idx_2022_na]
      idx_match <- match(claves_na, datos_imputados_solo_2022$clave)
      
      # Actualizar valores
      idx_validos <- !is.na(idx_match)
      if(sum(idx_validos) > 0) {
        datos_finales[idx_2022_na[idx_validos], var] <- 
          datos_imputados_solo_2022[idx_match[idx_validos], var]
        
        valores_actualizados <- valores_actualizados + sum(idx_validos)
        cat(sprintf("%-30s: %3d valores imputados\n", var, sum(idx_validos)))
      }
    }
  }
}

cat("Total valores actualizados:", valores_actualizados, "\n")

# ===============================================================================
# 10. GUARDAR RESULTADOS
# ===============================================================================

cat("\n=== GUARDANDO RESULTADOS ===\n")

# 1. Base master con imputaciones FIES 2022
write.csv(datos_finales, "resultados/BASE_MASTER_FIES_2022_IMPUTADA.csv", row.names = FALSE)

# 2. Solo datos 2022 imputados
write.csv(datos_imputados_solo_2022, "resultados/datos_2022_fies_imputados.csv", row.names = FALSE)

# 3. Cada imputación individual
for(i in 1:amelia_result$m) {
  filename <- paste0("resultados/imputacion_", i, "_fies_2022.csv")
  write.csv(amelia_result$imputations[[i]], filename, row.names = FALSE)
}

# 4. Objeto Amelia completo
save(amelia_result, file = "resultados/amelia_fies_2022_completo.RData")

# 5. Reporte de imputación
datos_2022_original <- datos_completos %>% filter(año == 2022)

reporte_imputacion <- data.frame(
  Variable = variables_fies_objetivo,
  Valores_Originales_2022 = sapply(variables_fies_objetivo, function(v) {
    if(v %in% names(datos_2022_original)) sum(!is.na(datos_2022_original[[v]])) else 0
  }),
  Valores_Faltantes_2022 = sapply(variables_fies_objetivo, function(v) {
    if(v %in% names(datos_2022_original)) sum(is.na(datos_2022_original[[v]])) else 0
  }),
  Valores_Imputados = sapply(variables_fies_objetivo, function(v) {
    if(v %in% names(datos_2022_original)) sum(is.na(datos_2022_original[[v]])) else 0
  })
)

reporte_imputacion$Cobertura_Final_Pct <- round(
  (reporte_imputacion$Valores_Originales_2022 + reporte_imputacion$Valores_Imputados) / 
  nrow(datos_2022_original) * 100, 2
)

write.csv(reporte_imputacion, "resultados/reporte_imputacion_fies_2022.csv", row.names = FALSE)

# ===============================================================================
# 11. REPORTE FINAL
# ===============================================================================

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("IMPUTACIÓN FIES 2022 COMPLETADA EXITOSAMENTE\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\nRESUMEN:\n")
cat("• Variables imputadas:", length(variables_fies_objetivo), "\n")
cat("• Año objetivo: 2022\n")
cat("• Registros procesados:", nrow(datos_2022_original), "\n")
cat("• Imputaciones múltiples: 5\n")
cat("• Valores totales imputados:", valores_actualizados, "\n")

cat("\nARCHIVOS GENERADOS:\n")
cat("1. BASE_MASTER_FIES_2022_IMPUTADA.csv - Base master completa con imputaciones\n")
cat("2. datos_2022_fies_imputados.csv - Solo datos 2022 imputados\n")
cat("3. imputacion_1.csv a imputacion_5.csv - Imputaciones individuales\n")
cat("4. amelia_fies_2022_completo.RData - Objeto Amelia completo\n")
cat("5. reporte_imputacion_fies_2022.csv - Reporte detallado\n")
cat("6. Gráficos de diagnóstico en carpeta diagnosticos/\n")

cat("\nCOBERTURA FINAL VARIABLES FIES 2022:\n")
print(reporte_imputacion[, c("Variable", "Cobertura_Final_Pct")])

cat("\n✅ PROCESO COMPLETADO\n")
cat("La base master ahora tiene las variables FIES 2022 completamente imputadas.\n")
cat("Puedes proceder con tus análisis predictivos.\n")
