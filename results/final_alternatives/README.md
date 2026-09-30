# Comparación de alternativas finales

El análisis ejecutable está en `code/06_evaluation/final_alternatives_analysis.py`.

La carpeta está separada de los resultados existentes. El script conserva los CSV originales y genera una copia filtrada y trazable de las seis configuraciones solicitadas, recalcula la sensibilidad del ranking con pesos ARI 0 %, 10 % y 25 %, ajusta las alternativas con todos los clientes, exporta asignaciones/perfiles, calcula ARI de concordancia entre configuraciones y tablas de contingencia, y crea figuras por conjunto de datos.

Distinciones metodológicas:

- Las métricas de validación y la estabilidad entre semillas se leen del estudio multimodelo existente sobre la muestra común.
- Los perfiles y asignaciones se calculan con el ajuste completo y semilla 42.
- En Fuzzy C-Means las etiquetas son `argmax` de la matriz de pertenencias.
- El ARI de las tablas de concordancia compara configuraciones distintas; no es estabilidad entre semillas.
- Los ceros de intervalos se cuantifican, pero no se clasifican automáticamente como imputados.

Ejecución, desde la raíz del proyecto y con el entorno definido en `environment.yml`:

```powershell
py -3.11 code/06_evaluation/final_alternatives_analysis.py
```

En el entorno de validación actual el lanzador de Python está instalado, pero Windows impide iniciar el ejecutable de Microsoft Store; por ello los archivos generados por la ejecución aún no están presentes y no se deben interpretar como resultados calculados.
