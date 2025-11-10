# ===============================================================================
# CONVERTIR BASE MASTER IMPUTADA A EXCEL
# ===============================================================================
# Convierte la base master con imputaciones FIES 2022 a formato Excel
# para facilitar la verificación y análisis visual

# Cargar librerías necesarias
if (!require(openxlsx)) {
  install.packages("openxlsx")
  library(openxlsx)
}

if (!require(dplyr)) {
  library(dplyr)
}

cat("=== CONVIRTIENDO BASE MASTER A EXCEL ===\n")

# Leer la base master imputada
base_imputada <- read.csv("resultados/BASE_MASTER_FIES_2022_IMPUTADA.csv", 
                         stringsAsFactors = FALSE)

cat("Dimensiones base imputada:", dim(base_imputada), "\n")

# Variables FIES para verificación
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

# Crear workbook de Excel
wb <- createWorkbook()

# ===============================================================================
# HOJA 1: BASE COMPLETA
# ===============================================================================
addWorksheet(wb, "Base_Completa")
writeData(wb, "Base_Completa", base_imputada)

# Formatear encabezados
headerStyle <- createStyle(
  fontSize = 11,
  fontColour = "white",
  fgFill = "#4472C4",
  halign = "center",
  valign = "center",
  textDecoration = "bold",
  border = "TopBottomLeftRight"
)

addStyle(wb, "Base_Completa", headerStyle, rows = 1, cols = 1:ncol(base_imputada))

# ===============================================================================
# HOJA 2: SOLO DATOS 2022 CON FIES IMPUTADAS
# ===============================================================================
datos_2022 <- base_imputada %>%
  filter(año == 2022) %>%
  select(departamento, año, mes, all_of(variables_fies), 
         IPM_Total, IPC_Total, Pobreza_monetaria)

addWorksheet(wb, "FIES_2022_Imputadas")
writeData(wb, "FIES_2022_Imputadas", datos_2022)

# Formatear encabezados
addStyle(wb, "FIES_2022_Imputadas", headerStyle, rows = 1, cols = 1:ncol(datos_2022))

# Resaltar columnas FIES con color diferente
fiesStyle <- createStyle(
  fgFill = "#E2EFDA",
  border = "TopBottomLeftRight"
)

# Encontrar columnas FIES
fies_cols <- which(names(datos_2022) %in% variables_fies)
if(length(fies_cols) > 0) {
  addStyle(wb, "FIES_2022_Imputadas", fiesStyle, 
           rows = 2:(nrow(datos_2022) + 1), cols = fies_cols, gridExpand = TRUE)
}

# ===============================================================================
# HOJA 3: ESTADÍSTICAS DESCRIPTIVAS FIES 2022
# ===============================================================================
estadisticas_fies <- data.frame(
  Variable = variables_fies,
  Media = sapply(variables_fies, function(v) {
    if(v %in% names(datos_2022)) round(mean(datos_2022[[v]], na.rm = TRUE), 2) else NA
  }),
  Mediana = sapply(variables_fies, function(v) {
    if(v %in% names(datos_2022)) round(median(datos_2022[[v]], na.rm = TRUE), 2) else NA
  }),
  Desv_Estandar = sapply(variables_fies, function(v) {
    if(v %in% names(datos_2022)) round(sd(datos_2022[[v]], na.rm = TRUE), 2) else NA
  }),
  Minimo = sapply(variables_fies, function(v) {
    if(v %in% names(datos_2022)) round(min(datos_2022[[v]], na.rm = TRUE), 2) else NA
  }),
  Maximo = sapply(variables_fies, function(v) {
    if(v %in% names(datos_2022)) round(max(datos_2022[[v]], na.rm = TRUE), 2) else NA
  }),
  Valores_NA = sapply(variables_fies, function(v) {
    if(v %in% names(datos_2022)) sum(is.na(datos_2022[[v]])) else NA
  })
)

addWorksheet(wb, "Estadisticas_FIES_2022")
writeData(wb, "Estadisticas_FIES_2022", estadisticas_fies)

# Formatear encabezados
addStyle(wb, "Estadisticas_FIES_2022", headerStyle, rows = 1, cols = 1:ncol(estadisticas_fies))

# ===============================================================================
# HOJA 4: COMPARACIÓN POR DEPARTAMENTO
# ===============================================================================
comparacion_dept <- datos_2022 %>%
  group_by(departamento) %>%
  summarise(
    FIES_preocupacion_promedio = round(mean(FIES_preocupacion_alimentos, na.rm = TRUE), 2),
    FIES_no_saludables_promedio = round(mean(FIES_no_alimentos_saludables, na.rm = TRUE), 2),
    FIES_poca_variedad_promedio = round(mean(FIES_poca_variedad_alimentos, na.rm = TRUE), 2),
    FIES_saltar_comida_promedio = round(mean(FIES_saltar_comida, na.rm = TRUE), 2),
    FIES_comio_menos_promedio = round(mean(FIES_comio_menos, na.rm = TRUE), 2),
    FIES_sin_alimentos_promedio = round(mean(FIES_sin_alimentos, na.rm = TRUE), 2),
    FIES_hambre_promedio = round(mean(FIES_hambre_sin_comer, na.rm = TRUE), 2),
    FIES_no_comio_dia_promedio = round(mean(FIES_no_comio_dia_entero, na.rm = TRUE), 2),
    .groups = 'drop'
  ) %>%
  arrange(departamento)

addWorksheet(wb, "Comparacion_Departamentos")
writeData(wb, "Comparacion_Departamentos", comparacion_dept)

# Formatear encabezados
addStyle(wb, "Comparacion_Departamentos", headerStyle, rows = 1, cols = 1:ncol(comparacion_dept))

# ===============================================================================
# HOJA 5: INFORMACIÓN DEL PROCESO
# ===============================================================================
info_proceso <- data.frame(
  Aspecto = c(
    "Fecha de procesamiento",
    "Variables imputadas",
    "Año objetivo",
    "Registros procesados 2022",
    "Valores totales imputados",
    "Método de imputación",
    "Número de imputaciones múltiples",
    "Método de consolidación",
    "Bounds aplicados",
    "Variables auxiliares utilizadas",
    "Convergencia",
    "Archivo original"
  ),
  Detalle = c(
    format(Sys.Date(), "%Y-%m-%d"),
    "8 variables FIES detalladas",
    "2022",
    "384 registros",
    "3,072 valores",
    "Amelia (Multiple Imputation)",
    "5 imputaciones",
    "Promedio aritmético",
    "0-100 (porcentajes)",
    "IPM_Total, IPC_Total, Pobreza_monetaria",
    "Exitosa",
    "BASE_MASTER_2022_2025_FIES_COMPLETO_RECUPERADO.csv"
  )
)

addWorksheet(wb, "Info_Proceso")
writeData(wb, "Info_Proceso", info_proceso)

# Formatear encabezados
addStyle(wb, "Info_Proceso", headerStyle, rows = 1, cols = 1:ncol(info_proceso))

# ===============================================================================
# GUARDAR ARCHIVO EXCEL
# ===============================================================================
archivo_excel <- "resultados/BASE_MASTER_FIES_2022_IMPUTADA.xlsx"
saveWorkbook(wb, archivo_excel, overwrite = TRUE)

cat("\n=== CONVERSIÓN COMPLETADA ===\n")
cat("Archivo Excel generado:", archivo_excel, "\n")
cat("\nHojas incluidas:\n")
cat("1. Base_Completa - Toda la base master con imputaciones\n")
cat("2. FIES_2022_Imputadas - Solo datos 2022 con variables FIES\n")
cat("3. Estadisticas_FIES_2022 - Estadísticas descriptivas\n")
cat("4. Comparacion_Departamentos - Promedios por departamento\n")
cat("5. Info_Proceso - Información del proceso de imputación\n")

cat("\n✅ Listo para verificación en Excel!\n")
