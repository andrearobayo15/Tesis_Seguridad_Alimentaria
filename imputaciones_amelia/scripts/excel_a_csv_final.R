# ===============================================================================
# CONVERTIR EXCEL MODIFICADO A CSV FINAL
# ===============================================================================
# Convierte el archivo Excel modificado por el usuario a formato CSV final

# Cargar librerías necesarias
library(openxlsx)
library(dplyr)

cat("=== CONVIRTIENDO EXCEL MODIFICADO A CSV FINAL ===\n")

# Leer el archivo Excel modificado
archivo_excel <- "resultados/BASE_MASTER_FIES_2022_IMPUTADA.xlsx"

# Verificar que el archivo existe
if(!file.exists(archivo_excel)) {
  stop("Error: No se encuentra el archivo Excel modificado")
}

cat("Leyendo archivo Excel modificado...\n")

# Leer la primera hoja (Base_Completa) que debería contener los datos modificados
base_final <- read.xlsx(archivo_excel, sheet = 1)

cat("Dimensiones de la base final:", dim(base_final), "\n")
cat("Columnas disponibles:", ncol(base_final), "\n")

# Mostrar las primeras columnas para verificar
cat("\nPrimeras columnas:\n")
print(head(names(base_final), 10))

# Verificar si hay variables FIES
variables_fies <- c(
  "FIES_preocupacion_alimentos",
  "FIES_no_alimentos_saludables", 
  "FIES_poca_variedad_alimentos",
  "FIES_saltar_comida",
  "FIES_comio_menos",
  "FIES_sin_alimentos",
  "FIES_hambre_sin_comer",
  "FIES_no_comio_dia_entero"
)

variables_fies_presentes <- variables_fies[variables_fies %in% names(base_final)]
cat("\nVariables FIES encontradas:", length(variables_fies_presentes), "\n")
if(length(variables_fies_presentes) > 0) {
  cat("Variables FIES presentes:\n")
  for(var in variables_fies_presentes) {
    cat("  •", var, "\n")
  }
}

# Verificar datos por año
if("año" %in% names(base_final)) {
  distribucion_años <- table(base_final$año)
  cat("\nDistribución por años:\n")
  print(distribucion_años)
  
  # Verificar completitud FIES en 2022
  if(length(variables_fies_presentes) > 0) {
    datos_2022 <- base_final[base_final$año == 2022, ]
    cat("\nCompletitud FIES en 2022:\n")
    for(var in variables_fies_presentes) {
      if(var %in% names(datos_2022)) {
        na_count <- sum(is.na(datos_2022[[var]]))
        total_count <- nrow(datos_2022)
        completitud <- round((total_count - na_count) / total_count * 100, 1)
        cat(sprintf("  • %-30s: %3d/%3d (%5.1f%% completo)\n", 
                   var, total_count - na_count, total_count, completitud))
      }
    }
  }
}

# Guardar como CSV final
archivo_csv_final <- "resultados/BASE_MASTER_FINAL_TESIS.csv"
write.csv(base_final, archivo_csv_final, row.names = FALSE)

cat("\n=== CONVERSIÓN COMPLETADA ===\n")
cat("Archivo CSV final generado:", archivo_csv_final, "\n")
cat("Dimensiones finales:", dim(base_final), "\n")

# Crear copia de respaldo con timestamp
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
archivo_respaldo <- paste0("resultados/BASE_MASTER_FINAL_TESIS_", timestamp, ".csv")
write.csv(base_final, archivo_respaldo, row.names = FALSE)
cat("Copia de respaldo:", archivo_respaldo, "\n")

cat("\n✅ Base final lista para análisis de tesis!\n")
cat("Archivo principal: BASE_MASTER_FINAL_TESIS.csv\n")
cat("Archivo respaldo: BASE_MASTER_FINAL_TESIS_", timestamp, ".csv\n", sep = "")
