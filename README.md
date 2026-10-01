# Segmentación de clientes mediante clustering

Repositorio del Trabajo de Fin de Máster sobre segmentación no supervisada de clientes a partir de datos transaccionales de H&M e Instacart.

El proyecto compara distintas representaciones de clientes —RFM/DFA y variables ampliadas—, siete algoritmos de clustering y diferentes valores de `k`. La evaluación utiliza métricas internas, selección de variables y análisis de estabilidad para obtener segmentos interpretables.

## Contenido del repositorio

El repositorio contiene el código reproducible y la documentación del análisis:

- `code/01_data_understanding/`: análisis exploratorio y revisión de la calidad de los datos.
- `code/02_data_preparation/`: construcción de representaciones, ingeniería de variables, transformaciones, escalado y selección de características.
- `code/03_modeling/`: funciones compartidas y notebooks de K-Means, MiniBatch K-Means, Bisecting K-Means, BIRCH, CLARA, Fuzzy C-Means y Gaussian Mixture Model.
- `code/04_iterative_data_preparation/`: selección iterativa de variables después del modelado inicial.
- `code/05_modeling_validation/`: reentrenamiento y validación multimodelo de los subconjuntos seleccionados.
- `code/06_evaluation/`: comparación de resultados e interpretación de la mejor segmentación.

## Datos y resultados

Los datos originales, los datasets procesados y los resultados exportados en CSV no forman parte del repositorio. El archivo `.gitignore` excluye cualquier `*.csv`, así como `CRISP_DM_REPORT.md`.

Para ejecutar los notebooks es necesario disponer localmente de los datos de H&M e Instacart y colocarlos en la estructura esperada:

```text
data/
├── hym/
└── instacart/
```

Las carpetas de datos intermedios (`data_without_nulls/`, `simple_datasets/`, `extended_datasets/` y `scaled_data/`) y `results/` se generan durante el flujo de trabajo y permanecen excluidas si contienen CSV. Los resultados se organizan localmente en `results/clustering/`, `results/feature_selection/`, `results/multimodel/`, `results/profiles/` y `results/final_alternatives/`.

### Organización y generación de `results/`

Los CSV de esta carpeta son artefactos generados por los notebooks y no datos fuente. Se excluyen del repositorio mediante `.gitignore`.

#### `results/clustering/`

Contiene la evaluación general de las 252 configuraciones de clustering (`7 algoritmos × 2 datasets × 2 representaciones × 9 valores de k`):

- `clustering_metrics.csv`: métricas de cada ejecución (`silhouette`, `Calinski-Harabasz`, `Davies-Bouldin`, tiempo y métricas específicas del algoritmo). Lo generan los notebooks de `code/03_modeling/` mediante `save_clustering_metrics()`, que consolida las ejecuciones y sustituye duplicados de una misma combinación.
- `clustering_top10.csv` y `clustering_bottom10.csv`: diez mejores y diez peores configuraciones por dataset, obtenidas al ordenar el ranking del estudio general.
- `clustering_algorithm_comparison.csv`: rendimiento agregado por algoritmo.
- `clustering_version_comparison.csv`: comparación agregada entre las representaciones simple y ampliada.

Los cuatro últimos archivos se generan con `code/06_evaluation/01_simple_vs_extended_clustering_comparison.ipynb` a partir de `clustering_metrics.csv`. El ranking combina las tres métricas internas y normaliza sus posiciones dentro de cada dataset.

#### `results/feature_selection/`

Contiene el estudio iterativo de selección y ablación de variables, generado por `code/04_iterative_data_preparation/feature_selection_study.ipynb` con MiniBatch K-Means. Se ejecuta después del modelado inicial y utiliza una muestra reproducible de hasta 50.000 clientes, `k=2..6` y las semillas `42`, `123` y `2026`.

- `feature_selection_results.csv`: resultados de todos los subconjuntos candidatos, incluyendo bases RFM/DFA, variables individuales, combinaciones, adiciones y retiradas de variables.
- `recommended_features.csv`: mejores subconjuntos preliminares por dataset, restringidos a un mínimo de tres variables para que los perfiles sean interpretables.

#### `results/multimodel/`

Contiene la validación de los subconjuntos seleccionados con los siete algoritmos. La genera `code/05_modeling_validation/multimodel_subset_validation.ipynb`, usando hasta 30.000 clientes, `k=2..6` y las tres semillas anteriores.

- `multi_model_feature_validation_detail.csv`: resultado detallado por dataset, subconjunto, algoritmo, `k` y semilla.
- `multi_model_feature_validation.csv`: resultados agregados y estabilidad media mediante ARI.
- `multi_model_recommended_features.csv`: recomendación final de variables, algoritmo y `k` para cada dataset.
- `multi_model_top10.csv` y `multi_model_bottom10.csv`: extremos del ranking multimodelo.
- `multi_model_subset_comparison.csv`: rendimiento agregado de cada subconjunto.
- `multi_model_algorithm_comparison.csv`: rendimiento y estabilidad agregados por algoritmo.
- `multi_model_variable_comparison.csv`: presencia de variables en los rankings y efectos de las comparaciones de ablación.

Estos archivos constituyen la validación multimodelo completa y se conservan como referencia. Sus rankings no deben interpretarse como una selección definitiva aislada cuando las diferencias sean pequeñas.

#### `results/final_alternatives/`

Contiene la comparación específica de las seis alternativas finales, generada por `code/06_evaluation/final_alternatives_analysis.py`. Esta carpeta complementa, sin sustituir, `results/multimodel/` y `results/profiles/`:

- `validation_metrics_requested.csv` y `validation_detail_requested.csv`: copia filtrada y trazable de los resultados multimodelo existentes.
- `ranking_sensitivity_ari_weights.csv`, `ranking_sensitivity_top10.csv` y `ranking_sensitivity_requested.csv`: sensibilidad del ranking al asignar al ARI un peso del 0 %, 10 % o 25 %.
- `assignments_*.csv`, `profiles_*.csv` y `profiles_all_alternatives.csv`: asignaciones y perfiles del ajuste completo para cada alternativa.
- `partition_concordance_ari.csv` y `partition_contingency_tables.csv`: concordancia entre configuraciones distintas y tablas de contingencia; no representan estabilidad entre semillas.
- `interval_quality_by_cluster.csv`: frecuencia de pocos pedidos y ceros observados en las variables de intervalos, sin inferir imputaciones solo a partir de un cero.
- `group_sizes_*.png` y `concordance_*.png`: figuras comparativas separadas por dataset.

Los archivos anteriores siguen siendo útiles para documentar el estudio completo, sus semillas y el contexto de todos los candidatos. Los nuevos archivos se utilizan para la comparación final de alternativas.
#### `results/profiles/`

Contiene los perfiles finales de los segmentos, generados por `code/06_evaluation/02_best_clustering_interpretation.ipynb` después de seleccionar la configuración ganadora:

- `hym_best_clustering_cluster_profiles.csv`: tamaño, porcentaje y medias de las variables de los clusters de H&M.
- `instacart_best_clustering_cluster_profiles.csv`: equivalente para Instacart.

Estos perfiles se calculan uniendo las etiquetas con los datos originales y permiten interpretar los clusters también con variables que no se utilizaron para construirlos.

## Entorno e instalación

El entorno reproducible está definido en [environment.yml](environment.yml). Incluye Python 3.10.14, las versiones de las librerías utilizadas y las dependencias específicas de CLARA y Fuzzy C-Means.

Con [Conda](https://docs.conda.io/projects/conda/en/latest/) o [Miniconda](https://docs.anaconda.com/miniconda/) instalado:

```bash
conda env create -f environment.yml
conda activate tfm-clustering
jupyter lab
```

Si se modifica el archivo de entorno, se puede actualizar la instalación con:

```bash
conda env update -f environment.yml --prune
```

## Flujo de ejecución

El flujo actual está organizado como una secuencia de notebooks: deben ejecutarse en orden porque cada etapa genera archivos que utiliza la siguiente. No existe todavía un script único que lance todo el proceso automáticamente.

1. Ejecutar los notebooks de `code/01_data_understanding/`.
2. Generar las representaciones simples y ampliadas desde `code/02_data_preparation/`.
3. Preparar las variables para el clustering:
   - Ejecutar los notebooks `01_rfm_hym.ipynb` y `01_dfa_instacart.ipynb` para generar las representaciones base de cada dataset.
   - Ejecutar `02_extended_hym.ipynb` y `02_extended_instacart.ipynb` para construir las variables ampliadas.
   - Ejecutar `03_variable_distributions.ipynb` como análisis previo de las distribuciones.
   - Ejecutar `04_scaling.ipynb`. Este notebook ya contiene las reglas de transformación (`log1p`, raíz cuadrada o Yeo–Johnson) y las aplica automáticamente; después estandariza las variables con `StandardScaler`.
   - El notebook genera los CSV escalados de `scaled_data/`, que son los que utilizan los algoritmos de clustering. No es necesario transformar las variables manualmente.
4. Ejecutar los notebooks de `code/03_modeling/` para obtener las métricas del estudio general.
5. **Data Preparation iterativa — selección de variables:** ejecutar `code/04_iterative_data_preparation/feature_selection_study.ipynb`. Este notebook prueba distintos subconjuntos de variables con MiniBatch K-Means y genera los resultados de `results/feature_selection/`.
6. **Modeling y validación multimodelo:** ejecutar `code/05_modeling_validation/multimodel_subset_validation.ipynb`. Este notebook vuelve a entrenar los siete algoritmos con los subconjuntos seleccionados y compara su estabilidad y rendimiento.
7. **Evaluation:** ejecutar los notebooks de `code/06_evaluation/`:
   - `01_simple_vs_extended_clustering_comparison.ipynb`, que compara las métricas del modelado general y puede ejecutarse después de `03_modeling`.
   - `02_best_clustering_interpretation.ipynb`, que interpreta las configuraciones finales después de la validación multimodelo.
   - `final_alternatives_analysis.py`, que compara las alternativas finales, recalcula la sensibilidad del ranking y genera perfiles y concordancias sin sobrescribir resultados anteriores.

La secuencia debe interpretarse según el ciclo iterativo de CRISP-DM:

```text
02_data_preparation
    └── 03_modeling
            ├── 06_evaluation/01_simple_vs_extended_clustering_comparison.ipynb
            │       └── comparación general del primer modelado
            └── 04_iterative_data_preparation/feature_selection_study.ipynb
                    └── 05_modeling_validation/multimodel_subset_validation.ipynb
                            └── 06_evaluation/02_best_clustering_interpretation.ipynb
```

La comparación general puede ejecutarse después del primer modelado. Si se realiza selección de variables, el flujo vuelve iterativamente a preparación, pasa por `05_modeling_validation/` para reentrenar los algoritmos y termina en `06_evaluation/`. Por tanto, la estructura distingue explícitamente la preparación, el modelado, la validación y la evaluación.

Los notebooks deben ejecutarse conservando la estructura de carpetas del proyecto. La documentación técnica ampliada (`PROJECT_DOCUMENTATION.md`) se mantiene como archivo local y está excluida del repositorio.

## Fuentes y uso de los datos

Este proyecto utiliza los siguientes conjuntos de datos exclusivamente con fines académicos:

- **H&M Personalized Fashion Recommendations**: datos proporcionados por H&M a través de la [competición de Kaggle](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations). Las reglas de la competición establecen el uso no comercial y académico de los datos y prohíben publicar, redistribuir o poner a disposición de terceros los archivos de datos. Por este motivo, este repositorio no incluye los CSV; deben descargarse directamente desde la fuente y usarse conforme a sus [reglas vigentes](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/rules).
- **Instacart Market Basket Analysis**: conjunto de datos de pedidos de Instacart publicado originalmente para la [competición de Kaggle](https://www.kaggle.com/competitions/instacart-market-basket-analysis). También puede consultarse la [referencia de datos de Instacart](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d35d4c5d9). Los archivos no se redistribuyen en este repositorio; quien los descargue debe aceptar y respetar las condiciones de la fuente utilizada.

Las transformaciones, datasets intermedios y resultados CSV generados localmente también están excluidos mediante `.gitignore`. Este repositorio contiene únicamente el código de análisis y la documentación mínima necesaria para reproducirlo con copias de los datos obtenidas legítimamente. La inclusión de estas referencias no sustituye la revisión de las condiciones de uso aplicables en la fecha de descarga.

## Alcance

Este repositorio contiene la parte experimental del TFM. Las métricas internas no sustituyen una validación de negocio y los datasets de H&M e Instacart deben obtenerse y citarse conforme a sus respectivas condiciones de uso.
