@echo off
echo ===============================================================================
echo EJECUTANDO IMPUTACION FIES 2022 CON AMELIA
echo ===============================================================================

cd /d "d:\Tesis maestria\Tesis codigo\imputaciones_amelia"

echo Iniciando R...
R.exe --vanilla < scripts\imputacion_fies_2022_amelia.R

echo.
echo ===============================================================================
echo PROCESO COMPLETADO
echo Revisar carpeta resultados\ para archivos generados
echo ===============================================================================
pause
