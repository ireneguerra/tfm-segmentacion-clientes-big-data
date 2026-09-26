import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    Birch,
    BisectingKMeans
)

from sklearn.mixture import GaussianMixture

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

import skfuzzy as fuzz


RANDOM_STATE = 42
K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE = 10000


def _get_clara():
    """Importa CLARA solo cuando se necesita el algoritmo."""
    try:
        from sklearn_extra.cluster import CLARA
    except (ImportError, ValueError) as error:
        raise ImportError(
            "No se puede cargar CLARA desde 'scikit-learn-extra'. "
            "Si aparece 'numpy.dtype size changed', reinstala NumPy, "
            "scikit-learn y scikit-learn-extra en versiones compatibles "
            "y reinicia el kernel."
        ) from error

    return CLARA


def calculate_metrics(X, labels):
    """
    Calcula las métricas internas comunes a todos los algoritmos.
    """

    X_array = np.asarray(X)

    # Por seguridad
    n_clusters = len(np.unique(labels))

    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan
        }

    return {
        "silhouette": silhouette_score(
            X_array,
            labels,
            sample_size=min(SILHOUETTE_SAMPLE, len(X_array)),
            random_state=RANDOM_STATE
        ),

        "calinski_harabasz":
            calinski_harabasz_score(X_array, labels),

        "davies_bouldin":
            davies_bouldin_score(X_array, labels)
    }


def save_clustering_metrics(
    resultados,
    model_name,
    dataset_name,
    dataset_version,
    output_path=None
):
    """
    Guarda las métricas de un notebook en una tabla CSV consolidada.

    Si el mismo modelo, dataset y versión ya existen, sustituye sus filas
    para que ejecutar nuevamente un notebook no genere duplicados.
    """

    if output_path is None:
        output_path = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "clustering"
            / "clustering_metrics.csv"
        )
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nuevos_resultados = resultados.copy()
    nuevos_resultados.insert(0, "dataset_version", dataset_version)
    nuevos_resultados.insert(0, "dataset", dataset_name)
    nuevos_resultados.insert(0, "model", model_name)

    key_columns = ["model", "dataset", "dataset_version"]

    if output_path.exists():
        resultados_guardados = pd.read_csv(output_path)

        for column in nuevos_resultados.columns:
            if column not in resultados_guardados.columns:
                resultados_guardados[column] = np.nan

        for column in resultados_guardados.columns:
            if column not in nuevos_resultados.columns:
                nuevos_resultados[column] = np.nan

        misma_ejecucion = np.logical_and.reduce([
            resultados_guardados[column].astype(str)
            == str(nuevos_resultados[column].iloc[0])
            for column in key_columns
        ])

        resultados_guardados = resultados_guardados.loc[~misma_ejecucion]
        tabla_completa = pd.concat(
            [resultados_guardados, nuevos_resultados],
            ignore_index=True
        )
    else:
        tabla_completa = nuevos_resultados

    tabla_completa = tabla_completa.sort_values(
        key_columns + ["k"]
    ).reset_index(drop=True)

    tabla_completa.to_csv(output_path, index=False)

    return tabla_completa


def kmeans_metrics(X):

    resultados = []

    for k in K_RANGE:

        inicio = time.time()

        modelo = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X)

        tiempo = time.time() - inicio

        metricas = calculate_metrics(X, labels)

        resultados.append({
            "k": k,
            **metricas,
            "inertia": modelo.inertia_,
            "tiempo_segundos": tiempo
        })

    return pd.DataFrame(resultados)


def minibatch_kmeans_metrics(X):

    resultados = []

    for k in K_RANGE:

        inicio = time.time()

        modelo = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            batch_size=4096,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X)

        tiempo = time.time() - inicio

        metricas = calculate_metrics(X, labels)

        resultados.append({
            "k": k,
            **metricas,
            "inertia": modelo.inertia_,
            "tiempo_segundos": tiempo
        })

    return pd.DataFrame(resultados)


def birch_metrics(
    X,
    threshold=0.5,
    branching_factor=50
):

    resultados = []

    # BIRCH crea primero un conjunto reducido de subclusters. Usar
    # n_clusters=k directamente hace que sklearn aplique clustering
    # aglomerativo a todos esos subclusters, lo que requiere una matriz
    # cuadrática y puede consumir cientos de GB en datasets grandes.
    inicio_birch = time.time()

    birch = Birch(
        n_clusters=None,
        threshold=threshold,
        branching_factor=branching_factor,
        compute_labels=False
    )

    birch.fit(X)
    subcluster_centers = birch.subcluster_centers_
    tiempo_birch = time.time() - inicio_birch

    for k in K_RANGE:

        inicio = time.time()

        n_clusters = min(k, len(subcluster_centers))

        modelo_global = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            batch_size=4096,
            random_state=RANDOM_STATE
        )

        modelo_global.fit(subcluster_centers)
        labels = modelo_global.predict(X)

        tiempo = tiempo_birch + (time.time() - inicio)

        metricas = calculate_metrics(X, labels)

        resultados.append({
            "k": k,
            **metricas,
            "n_clusters_found": len(np.unique(labels)),
            "tiempo_segundos": tiempo
        })

    return pd.DataFrame(resultados)


def clara_metrics(X):
    CLARA = _get_clara()

    resultados = []

    X_array = np.asarray(X)

    for k in K_RANGE:

        inicio = time.time()

        modelo = CLARA(
            n_clusters=k,
            metric="euclidean",
            init="build",
            n_sampling_iter=5,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X_array)

        tiempo = time.time() - inicio

        metricas = calculate_metrics(X_array, labels)

        resultados.append({
            "k": k,
            **metricas,
            "tiempo_segundos": tiempo
        })

    return pd.DataFrame(resultados)


def fuzzy_cmeans_metrics(
    X,
    m=2,
    error=0.005,
    maxiter=300
):

    resultados = []

    X_array = np.asarray(X, dtype=np.float64)

    # skfuzzy necesita:
    # variables x observaciones
    X_fuzzy = X_array.T

    for k in K_RANGE:

        inicio = time.time()

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=X_fuzzy,
            c=k,
            m=m,
            error=error,
            maxiter=maxiter,
            seed=RANDOM_STATE
        )

        # Para poder comparar con los demás algoritmos
        # convertimos la pertenencia fuzzy en una etiqueta final
        labels = np.argmax(u, axis=0)

        tiempo = time.time() - inicio

        metricas = calculate_metrics(X_array, labels)

        resultados.append({
            "k": k,
            **metricas,
            "fpc": fpc,
            "objective_function": jm[-1],
            "iterations": p,
            "tiempo_segundos": tiempo
        })

    return pd.DataFrame(resultados)


def gmm_metrics(
    X,
    covariance_type="diag"
):

    resultados = []

    X_array = np.asarray(X)

    for k in K_RANGE:

        inicio = time.time()

        modelo = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            n_init=1,
            max_iter=200,
            random_state=RANDOM_STATE
        )

        modelo.fit(X_array)

        labels = modelo.predict(X_array)

        tiempo = time.time() - inicio

        metricas = calculate_metrics(X_array, labels)

        resultados.append({
            "k": k,
            **metricas,
            "bic": modelo.bic(X_array),
            "aic": modelo.aic(X_array),
            "tiempo_segundos": tiempo
        })

    return pd.DataFrame(resultados)


def bisecting_kmeans_metrics(X):

    resultados = []

    for k in K_RANGE:

        inicio = time.time()

        modelo = BisectingKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X)

        tiempo = time.time() - inicio

        metricas = calculate_metrics(X, labels)

        resultados.append({
            "k": k,
            **metricas,
            "inertia": modelo.inertia_,
            "tiempo_segundos": tiempo
        })

    return pd.DataFrame(resultados)


def plot_clustering_metrics(
    resultados,
    fourth_metric="tiempo_segundos",
    fourth_title="Tiempo de ejecución"
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9)
    )

    # Silhouette
    axes[0, 0].plot(
        resultados["k"],
        resultados["silhouette"],
        marker="o"
    )

    axes[0, 0].set_title("Silhouette")
    axes[0, 0].set_xlabel("Número de clusters")


    # Calinski-Harabasz
    axes[0, 1].plot(
        resultados["k"],
        resultados["calinski_harabasz"],
        marker="o"
    )

    axes[0, 1].set_title("Calinski-Harabasz")
    axes[0, 1].set_xlabel("Número de clusters")


    # Davies-Bouldin
    axes[1, 0].plot(
        resultados["k"],
        resultados["davies_bouldin"],
        marker="o"
    )

    axes[1, 0].set_title("Davies-Bouldin")
    axes[1, 0].set_xlabel("Número de clusters")


    # Métrica específica del modelo
    axes[1, 1].plot(
        resultados["k"],
        resultados[fourth_metric],
        marker="o"
    )

    axes[1, 1].set_title(fourth_title)
    axes[1, 1].set_xlabel("Número de clusters")


    plt.tight_layout()
    plt.show()



def fit_clustering_model(
    model_name,
    k,
    X,
    birch_threshold=0.5,
    birch_branching_factor=50,
    fuzzy_m=2,
    gmm_covariance_type="diag"
):
    """
    Entrena cualquiera de los 7 algoritmos para un k concreto.

    Devuelve:
        modelo
        labels

    model_name:
        "kmeans"
        "minibatch"
        "birch"
        "clara"
        "fuzzy"
        "gmm"
        "bisecting"
    """

    X_array = np.asarray(X)

    if model_name == "kmeans":

        modelo = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X)

        return modelo, labels


    elif model_name == "minibatch":

        modelo = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            batch_size=4096,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X)

        return modelo, labels

    elif model_name == "birch":

        birch = Birch(
            n_clusters=None,
            threshold=birch_threshold,
            branching_factor=birch_branching_factor,
            compute_labels=False
        )

        birch.fit(X)

        n_clusters = min(k, len(birch.subcluster_centers_))
        modelo_global = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            batch_size=4096,
            random_state=RANDOM_STATE
        )

        modelo_global.fit(birch.subcluster_centers_)
        labels = modelo_global.predict(X)

        modelo = {
            "birch": birch,
            "global_clusterer": modelo_global,
            "n_subclusters": len(birch.subcluster_centers_)
        }

        return modelo, labels


    elif model_name == "clara":
        CLARA = _get_clara()

        modelo = CLARA(
            n_clusters=k,
            metric="euclidean",
            init="build",
            n_sampling_iter=5,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X_array)

        return modelo, labels


    elif model_name == "fuzzy":

        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=X_array.T,
            c=k,
            m=fuzzy_m,
            error=0.005,
            maxiter=300,
            seed=RANDOM_STATE
        )

        labels = np.argmax(u, axis=0)

        # Guardamos la información relevante en un diccionario
        modelo = {
            "centers": cntr,
            "membership": u,
            "fpc": fpc,
            "objective_function": jm,
            "iterations": p
        }

        return modelo, labels


    elif model_name == "gmm":

        modelo = GaussianMixture(
            n_components=k,
            covariance_type=gmm_covariance_type,
            n_init=1,
            max_iter=200,
            random_state=RANDOM_STATE
        )

        modelo.fit(X_array)

        labels = modelo.predict(X_array)

        return modelo, labels


    elif model_name == "bisecting":

        modelo = BisectingKMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE
        )

        labels = modelo.fit_predict(X)

        return modelo, labels


    else:

        raise ValueError(
            "Modelo no reconocido. Usa: "
            "'kmeans', 'minibatch', 'birch', "
            "'clara', 'fuzzy', 'gmm' o 'bisecting'."
        )


def analyze_clusters(
    labels,
    df_scaled,
    df_original,
    id_col="user_id"
):
    """
    Relaciona las etiquetas del clustering con el identificador
    del cliente y calcula tamaños, porcentajes y perfiles usando
    las variables originales.
    """


    asignacion_clusters = df_scaled[[id_col]].copy()

    asignacion_clusters["cluster"] = labels


    df_clusterizado = df_original.merge(
        asignacion_clusters,
        on=id_col,
        how="left"
    )

    perfil_clusters = (
        df_clusterizado
        .drop(columns=[id_col])
        .groupby("cluster")
        .mean(numeric_only=True)
        .round(3)
    )


    cluster_sizes = (
        df_clusterizado["cluster"]
        .value_counts()
        .sort_index()
    )

    porcentajes = (
        df_clusterizado["cluster"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )


    return (
        cluster_sizes,
        porcentajes,
        perfil_clusters,
        df_clusterizado
    )

