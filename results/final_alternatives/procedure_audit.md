# Auditoría previa del procedimiento

- `multi_model_feature_validation.csv`: 350 filas, 175 por conjunto de datos (cinco subsets × siete algoritmos × cinco valores de k).
- `multi_model_feature_validation_detail.csv`: 2.310 filas y 2.310 claves únicas (`dataset`, `subset`, `model`, `k`, `seed`); contiene 1.155 filas por conjunto. El script no lo sobrescribe.
- El notebook agrupa el detalle por configuración y promedia las tres semillas. La estabilidad es la media de los tres ARI por pares entre semillas.
- Las etiquetas de Fuzzy C-Means se obtienen mediante `np.argmax(u, axis=0)`.
- El ranking original calcula percentiles por dataset con `rank(pct=True)`; silhouette y Calinski se maximizan, Davies-Bouldin se minimiza y estabilidad se maximiza. Los empates reciben el rango medio. La ordenación posterior usa score, estabilidad y silhouette, aunque no fija explícitamente un método adicional de desempate para filas completamente idénticas.
- Las cuatro métricas del ranking provienen de la muestra común de hasta 30.000 clientes. Los perfiles actuales y los nuevos perfiles alternativos deben tratarse como resultados del ajuste completo.
- La comparación nueva usa las variables solicitadas literalmente y documenta el ajuste completo, la semilla, el número de clientes, el tiempo y la regla de etiquetado.
