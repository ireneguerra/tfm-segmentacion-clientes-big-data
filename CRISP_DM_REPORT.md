# Segmentación de clientes mediante técnicas de clustering

## Desarrollo metodológico, resultados y conclusiones según CRISP-DM

## 1. Introducción

La segmentación de clientes es una herramienta fundamental para comprender la heterogeneidad de una cartera comercial. Frente a una visión agregada, en la que todos los consumidores se estudian como una única población, la segmentación permite identificar grupos con patrones semejantes de compra, recurrencia, gasto, variedad o relación con el canal. Estos grupos pueden servir posteriormente como base para diseñar acciones diferenciadas de comunicación, fidelización, recomendación o reactivación.

El presente Trabajo de Fin de Máster aborda este problema mediante técnicas de aprendizaje no supervisado. El análisis se aplica a dos conjuntos de datos transaccionales de gran tamaño y pertenecientes a contextos comerciales diferentes:

- **H&M Personalized Fashion Recommendations**, representativo del comercio de moda.
- **Instacart Market Basket Analysis**, representativo de la compra recurrente de alimentación y productos del hogar.

La utilización de dos fuentes permite comprobar si las decisiones metodológicas y las conclusiones se mantienen ante comportamientos de compra distintos. H&M contiene información de precios, artículos, clientes y canales, pero no identifica explícitamente los pedidos. Instacart sí recoge pedidos y productos, aunque no incluye importes monetarios. Esta diferencia obliga a adaptar la representación del cliente a cada caso.

El proyecto se organiza de acuerdo con la metodología **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*). Sus fases no se entienden como una secuencia estrictamente lineal, sino como un proceso iterativo. En particular, los resultados del primer modelado motivan un regreso a la preparación de los datos para estudiar la contribución de las variables y construir subconjuntos más informativos.

## 2. Objetivos

### 2.1. Objetivo general

Desarrollar y evaluar un procedimiento reproducible de segmentación no supervisada capaz de identificar grupos de clientes con comportamientos de compra diferenciados en dos contextos comerciales.

### 2.2. Objetivos específicos

1. Analizar la estructura, calidad y contenido de los datos transaccionales de H&M e Instacart.
2. Construir una representación numérica de cada cliente a partir de su historial de compra.
3. Comparar una representación básica RFM o DFA con otra ampliada mediante variables de cesta, regularidad, variedad y canal.
4. Aplicar transformaciones que reduzcan la asimetría y permitan comparar variables medidas en escalas diferentes.
5. Evaluar siete algoritmos de clustering bajo un marco común de experimentación.
6. Comparar diferentes números de clusters mediante métricas internas.
7. Estudiar la aportación de las variables mediante un procedimiento de ablación y selección de características.
8. Validar los subconjuntos candidatos con varios algoritmos y semillas aleatorias.
9. Caracterizar los clusters finales en unidades originales y traducirlos a perfiles comprensibles.
10. Identificar las limitaciones del análisis y proponer líneas de mejora y posibles aplicaciones comerciales.

## 3. Diseño general de la investigación

El proyecto sigue las seis fases de CRISP-DM, pero su ejecución no es lineal. Primero se realiza un ciclo inicial completo hasta la evaluación de las representaciones y algoritmos. A continuación, los resultados de esa evaluación motivan un retorno a la fase de preparación de los datos para seleccionar variables. El subconjunto seleccionado se vuelve a modelar y se evalúa de nuevo antes de elaborar los perfiles finales.

```text
Fase I: comprensión del negocio
                  │
                  ▼
Fase II: comprensión de los datos
                  │
                  ▼
Fase III: preparación inicial
                  │
                  ▼
Fase IV: modelado inicial
                  │
                  ▼
Fase V: evaluación inicial ───────────────┐
                  │                         │
                  │                         ▼
                  │             Retorno a la preparación
                  │             (selección de variables)
                  │                         │
                  │                         ▼
                  │             Modelado y validación
                  │                         │
                  └─────────────────────────▼
                            Evaluación final
                                      │
                                      ▼
                         Fase VI: despliegue
```

La unidad de análisis es el **cliente**. Los registros transaccionales se agregan para obtener una fila por cliente y una serie de variables que resumen su comportamiento. Los identificadores se conservan para relacionar posteriormente los resultados con las tablas originales, pero se excluyen del espacio utilizado por los algoritmos. La selección de variables no sustituye al primer modelado: se realiza después de observar el comportamiento de las representaciones iniciales y da lugar a un segundo ciclo de modelado y evaluación.

---

# 4. Fase I de CRISP-DM: comprensión del negocio

## 4.1. Planteamiento del problema

El problema se formula como una tarea de clustering: no existe una etiqueta conocida que indique a qué segmento pertenece cada cliente y, por tanto, los grupos deben inferirse a partir de similitudes en su comportamiento.

Desde una perspectiva empresarial, la finalidad de la segmentación consiste en sustituir una estrategia homogénea por una comprensión diferenciada de la cartera. Entre las aplicaciones potenciales se encuentran:

- distinguir clientes activos, ocasionales o inactivos;
- identificar compradores frecuentes o de mayor valor;
- detectar diferencias en la regularidad de compra;
- reconocer preferencias de canal;
- estudiar el grado de variedad o repetición del consumo;
- orientar campañas de fidelización, reactivación o recomendación.

Estas aplicaciones son potenciales porque el proyecto evalúa la calidad interna de los clusters, pero no dispone de resultados de campañas, indicadores de rentabilidad ni etiquetas de negocio con los que demostrar su utilidad comercial externa.

## 4.2. Criterios de éxito

Al tratarse de aprendizaje no supervisado, el éxito técnico no puede medirse mediante exactitud frente a una variable objetivo. Se establecen los siguientes criterios:

- elevada cohesión dentro de cada cluster y separación entre clusters;
- estabilidad ante cambios de semilla aleatoria;
- ausencia de clusters degenerados o soluciones técnicamente inválidas;
- riqueza suficiente de la representación para producir perfiles interpretables;
- coste computacional razonable para conjuntos de datos de gran tamaño;
- coherencia entre las variables usadas por el modelo y la interpretación posterior.

El éxito empresarial requeriría una fase adicional de validación con expertos, campañas o indicadores externos. Por ello, una buena puntuación interna se considera evidencia favorable, pero no una demostración definitiva de valor comercial.

## 4.3. Alcance

El alcance incluye la exploración, preparación, modelado, evaluación y caracterización de segmentos. No incluye la implantación de un sistema productivo, la asignación en tiempo real de nuevos clientes ni la ejecución de campañas sobre los grupos obtenidos.

---

# 5. Fase II de CRISP-DM: comprensión de los datos

## 5.1. Datos de H&M

Los datos proceden de la competición **H&M Personalized Fashion Recommendations**, organizada por **H&M Group** en la plataforma Kaggle entre febrero y mayo de 2022. La empresa publicó un historial anonimizado de compras junto con metadatos de clientes y artículos —incluidas descripciones e imágenes de producto— para plantear un problema de recomendación: predecir hasta doce artículos que cada cliente compraría durante los siete días posteriores al periodo de entrenamiento. Por tanto, aunque en este trabajo los datos se reutilizan con un objetivo no supervisado de segmentación, su finalidad original era el desarrollo y la evaluación de sistemas de recomendación de moda. El conjunto y sus condiciones de uso están disponibles en la [página oficial de la competición en Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

El conjunto de H&M contiene tres tablas principales:

| Tabla | Contenido principal |
|---|---|
| `transactions_train.csv` | Fecha, cliente, artículo, precio y canal de venta |
| `customers.csv` | Información anonimizada de los clientes |
| `articles.csv` | Catálogo y jerarquía de los artículos |

Una limitación estructural relevante es la ausencia de un identificador de pedido. Las transacciones solo permiten conocer el cliente, el artículo y la fecha, por lo que no es posible determinar si todas las líneas registradas por una persona durante un mismo día pertenecen a una única compra real. Esta limitación deberá resolverse mediante una aproximación operativa durante la preparación.

## 5.2. Datos de Instacart

El origen de estos datos es **Instacart**, empresa estadounidense dedicada a la compra y entrega de productos de supermercado. En 2017, la compañía hizo público un conjunto anonimizado de más de tres millones de pedidos pertenecientes a más de 200.000 usuarios, conocido como **The Instacart Online Grocery Shopping Dataset 2017**. Posteriormente, estos datos se emplearon en la competición *Instacart Market Basket Analysis* de Kaggle, cuyo objetivo era predecir qué productos adquiridos previamente volverían a aparecer en el siguiente pedido de cada usuario. Para este trabajo se utilizó la copia del conjunto disponible en [Kaggle, publicada por psparks](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis). Así, Kaggle constituye la fuente de descarga empleada, mientras que Instacart es la entidad que generó y difundió originalmente la información.

El conjunto de Instacart relaciona seis tablas:

| Tabla | Contenido principal |
|---|---|
| `orders.csv` | Secuencia de pedidos, día, hora y días desde el pedido anterior |
| `order_products__prior.csv` | Productos incluidos en los pedidos históricos |
| `order_products__train.csv` | Productos del último pedido conocido de los clientes de entrenamiento |
| `products.csv` | Catálogo de productos |
| `aisles.csv` | Pasillos o categorías intermedias |
| `departments.csv` | Departamentos o categorías generales |

En este caso existe un identificador de pedido y un orden temporal por usuario, pero no se proporciona el importe económico. Esta ausencia impide calcular directamente la dimensión monetaria del modelo RFM y condiciona la representación que deberá construirse durante la preparación.

## 5.3. Exploración y calidad

La exploración se realiza en los notebooks de comprensión de datos. El procedimiento comprende:

1. inspección de dimensiones, tipos y primeras observaciones;
2. análisis de valores ausentes;
3. revisión de registros duplicados;
4. estudio de distribuciones y frecuencias;
5. análisis específico de clientes, artículos, pedidos y cestas.

En H&M se examinan, entre otros elementos, la edad, el estado de membresía, las comunicaciones comerciales, la actividad de los clientes y la composición del catálogo. En Instacart se estudian la distribución temporal de los pedidos, los días transcurridos entre compras, el tamaño de las cestas y el comportamiento de recompra.

## 5.4. Principales hallazgos del análisis exploratorio

### 5.4.1. Hallazgos en H&M

El conjunto original contiene **105.542 artículos**, **1.371.980 clientes** y **31.788.324 líneas de transacción**. El historial se extiende desde el 20 de septiembre de 2018 hasta el 22 de septiembre de 2020, por lo que ofrece aproximadamente dos años de comportamiento transaccional.

El estudio de calidad muestra que las transacciones no contienen valores nulos. En cambio, la tabla de clientes presenta ausencias importantes en `Active` (**66,15 %**) y `FN` (**65,24 %**), columnas en las que únicamente aparece el valor 1 cuando el indicador está registrado. La edad tiene un **1,16 %** de valores ausentes y una mediana de 32 años. También se observan ausencias en variables categóricas de clientes. En el catálogo, `detail_desc` es la única columna incompleta, con 416 ausencias (**0,39 %**). El tratamiento de estos problemas se define posteriormente, dentro de la fase de preparación.

Se detectan **2.974.905 filas completamente repetidas** en la tabla transaccional. No se eliminan automáticamente porque cada fila representa una unidad y varias coincidencias de cliente, artículo y fecha pueden corresponder a la compra real de varias unidades iguales. Esta observación refuerza la necesidad de tratar la transacción como una línea de artículo, no como un pedido completo.

La distribución de actividad por cliente es claramente asimétrica:

| Indicador por cliente | Media | Mediana | Percentil 75 | Máximo |
|---|---:|---:|---:|---:|
| Artículos comprados | 23,33 | 9 | 27 | 1.895 |
| Artículos distintos | 20,04 | 8 | 24 | 1.346 |
| Gasto acumulado | 0,65 | 0,25 | 0,70 | 57,68 |
| Días diferentes de compra | 6,67 | 3 | 8 | 427 |

La diferencia entre medias, medianas y máximos evidencia una mayoría de clientes con poca actividad y una minoría con historiales muy intensos. Esta asimetría justifica posteriormente el uso de transformaciones como `log1p`, raíz cuadrada y Yeo–Johnson antes de aplicar algoritmos basados en distancias.

> **Figura 1. Distribución del número de compras por cliente en H&M.**  
> *[Insertar aquí el histograma generado en `code/01_data_understanding/eda_hym.ipynb`, apartado «Clientes con más compras».]*  
> **Interpretación:** se espera una elevada concentración en valores bajos y una cola derecha prolongada. La figura permite mostrar visualmente que la media no representa al cliente típico y que el escalado, por sí solo, no corregiría la asimetría.

El precio por línea también presenta cola derecha: su mediana es 0,03, el percentil 99 alcanza 0,10 y el máximo es 0,59. Estas cantidades se mantienen en la escala anonimizada o normalizada del dataset y no deben interpretarse directamente como euros. Además, el canal 2 concentra **22.379.862 transacciones**, frente a **9.408.462** del canal 1, y presenta un precio medio ligeramente superior. Este desequilibrio respalda la incorporación de `channel_2_ratio` como variable de comportamiento individual.

> **Figura 2. Distribución del precio por artículo en H&M.**  
> *[Insertar aquí el histograma o boxplot de `price` generado en `code/01_data_understanding/eda_hym.ipynb`.]*  
> **Interpretación:** la concentración en precios reducidos y la presencia de una cola hasta 0,59 justifican una transformación robusta. La figura no debe rotular el eje como euros, ya que el dataset utiliza una escala normalizada.

El catálogo y las ventas muestran una concentración clara en moda femenina y prendas de uso general. `Ladieswear` es la línea comercial con más referencias; las prendas de la parte superior constituyen el grupo de producto más amplio y vendido; pantalones, vestidos, jerséis y camisetas se encuentran entre los tipos predominantes. El negro es, con diferencia, el color más comprado, seguido del blanco y el azul oscuro. Estos resultados muestran que la variedad de artículos y categorías podría diferenciar comportamientos, lo que motiva la creación de variables como artículos, tipos, grupos y secciones únicas por cliente.

> **Figura 3. Tipos de producto más comprados en H&M.**  
> *[Insertar aquí el gráfico de barras del apartado «Tipos de producto más comprados» de `code/01_data_understanding/eda_hym.ipynb`.]*  
> **Interpretación:** el predominio de unas pocas categorías muestra que contar solamente artículos distintos no recoge toda la diversidad. Dos clientes pueden comprar el mismo número de artículos y, sin embargo, concentrarse en uno o varios tipos de producto.

En términos temporales, el sábado registra el mayor número de líneas de compra, seguido del jueves y el miércoles. No obstante, el histórico de 2018 y 2020 no cubre años naturales completos, por lo que sus totales anuales no deben compararse directamente con 2019. La evolución temporal confirma la necesidad de calcular recencia respecto a una fecha de referencia común y de incorporar medidas de separación entre ocasiones de compra.

> **Figura 4. Evolución mensual de las transacciones de H&M.**  
> *[Insertar aquí la serie temporal mensual generada en `code/01_data_understanding/eda_hym.ipynb`.]*  
> **Interpretación:** la figura permite detectar cambios de volumen y posibles patrones estacionales. Los extremos del periodo deben leerse teniendo en cuenta su cobertura parcial, por lo que no es correcto atribuir cualquier descenso exclusivamente a una pérdida de demanda.

En conjunto, el EDA de H&M conduce a cuatro decisiones metodológicas: definir la ocasión de compra por cliente y día, agregar la actividad a nivel de cliente, transformar las variables con colas pronunciadas e incorporar dimensiones de regularidad, variedad y canal además de RFM.

### 5.4.2. Hallazgos en Instacart

Instacart contiene **3.421.083 pedidos** de **206.209 usuarios**, **49.688 productos**, 134 pasillos y 21 departamentos. La tabla histórica `order_products__prior.csv` reúne **32.434.489 líneas de producto**, mientras que `order_products__train.csv` contiene 1.384.617.

El único patrón de ausencia se encuentra en `days_since_prior_order`: hay **206.209 nulos**, exactamente uno por usuario, y todos corresponden a su primer pedido. No se trata, por tanto, de una pérdida aleatoria de información, sino de un nulo estructural: no puede existir un intervalo anterior para la primera compra. No se detectan filas completamente duplicadas ni identificadores duplicados en las tablas maestras. La codificación operativa del primer pedido se aborda en la fase de preparación.

Los clientes disponen de una media de **16,59 pedidos**, una mediana de 10 y un rango entre 4 y 100. La distancia entre pedidos tiene una mediana de 7 días, una media de 10,44 y un percentil 75 de 15 días. El máximo de 30 debe interpretarse con cautela porque el propio dataset limita esta variable a dicho valor. La recurrencia semanal y la variabilidad observada justifican la construcción de `avg_days_between_orders` y `std_days_between_orders`.

> **Figura 5. Distribución de los días desde el pedido anterior en Instacart.**  
> *[Insertar aquí el gráfico del apartado «Días desde el pedido anterior» de `code/01_data_understanding/eda_instacart.ipynb`.]*  
> **Interpretación:** la concentración en intervalos cortos refleja compra recurrente, mientras que la acumulación en 30 días debe leerse como censura superior de la variable y no necesariamente como un ciclo mensual exacto para todos esos pedidos.

Las cestas históricas contienen una media de **10,09 productos** y una mediana de 8; el percentil 75 es 14 y el máximo alcanza 145. La diferencia entre el centro de la distribución y sus valores extremos confirma una cola derecha y fundamenta la transformación logarítmica del tamaño medio y de su desviación.

> **Figura 6. Distribución del tamaño de cesta en Instacart.**  
> *[Insertar aquí el gráfico del apartado «Tamaño de las cestas» de `code/01_data_understanding/eda_instacart.ipynb`.]*  
> **Interpretación:** las cestas excepcionalmente grandes pueden dominar las distancias si se mantiene la escala original. La transformación logarítmica reduce su influencia sin eliminar estos comportamientos reales.

La tasa general de recompra es del **58,97 %**, lo que indica que repetir productos ya adquiridos constituye una parte central del comportamiento de los usuarios. Los productos más comprados están dominados por alimentos frescos: banana, bolsa de bananas orgánicas, fresas orgánicas, espinaca y aguacate aparecen en las primeras posiciones. Los departamentos con mayor volumen son `produce`, `dairy eggs`, `snacks`, `beverages` y `frozen`; los pasillos más utilizados corresponden a frutas frescas, verduras frescas, frutas y verduras envasadas, yogur y queso envasado.

También existen diferencias en la fidelidad por categoría. `dairy eggs` presenta una tasa media de recompra del 67 %, mientras que `beverages` y `produce` alcanzan aproximadamente el 65 %. En cambio, categorías como `pantry` y `personal care` presentan tasas notablemente menores. Esto respalda la creación de variables de recompra, variedad de productos y amplitud de categorías por cliente.

> **Figura 7. Tasa de recompra por departamento en Instacart.**  
> *[Insertar aquí el gráfico del apartado «Tasa de recompra por departamento» de `code/01_data_understanding/eda_instacart.ipynb`.]*  
> **Interpretación:** las diferencias entre departamentos muestran que la repetición no es homogénea. Un ratio de recompra individual puede distinguir hábitos rutinarios de patrones más exploratorios, aunque también está condicionado por la mezcla de categorías compradas.

En conjunto, el EDA de Instacart conduce a cuatro decisiones principales: distinguir los nulos estructurales del primer pedido, representar la recurrencia mediante intervalos, resumir el volumen mediante estadísticas de cesta e incorporar medidas de recompra y diversidad de producto.

### 5.4.3. Relación entre el EDA y las variables construidas

| Evidencia exploratoria | Decisión de preparación |
|---|---|
| Mayoría de clientes con poca actividad y minoría muy intensa | Transformar distribuciones asimétricas antes del escalado |
| Ausencia de identificador de pedido en H&M | Definir una ocasión de compra como cliente y fecha |
| Diferencias claras entre los dos canales de H&M | Crear `channel_2_ratio` |
| Catálogo amplio y jerárquico | Crear recuentos de artículos y categorías únicas |
| Intervalos recurrentes y variables entre pedidos | Crear medias y desviaciones de días entre compras |
| Elevada tasa de recompra en Instacart | Crear `reorder_ratio` |
| Cestas y diversidad de productos heterogéneas | Crear estadísticas de cesta y ratios de variedad |
| Nulos exclusivos del primer pedido de Instacart | Tratarlos como ausencia estructural, no como dato perdido aleatorio |

De este modo, el análisis exploratorio no constituye una etapa aislada. Sus resultados determinan la unidad de agregación, el tratamiento de ausencias, las transformaciones y buena parte de la ingeniería de características empleada posteriormente.

## 5.5. Diferencias entre los dominios

Las dos fuentes no deben compararse como si sus variables fueran equivalentes:

- H&M permite medir valor monetario y canal, pero aproxima el concepto de pedido mediante la fecha.
- Instacart ofrece pedidos explícitos, secuencia temporal y recompra, pero no contiene precios.
- La compra de moda puede ser más estacional y esporádica.
- La compra de alimentación suele mostrar una recurrencia mayor.

Estas diferencias justifican la construcción de representaciones adaptadas y aconsejan no comparar directamente los valores absolutos de las métricas entre datasets.

---

# 6. Fase III de CRISP-DM: preparación de los datos

## 6.1. Limpieza y tratamiento de valores ausentes

Una vez identificados los problemas de calidad durante la fase de comprensión, la preparación aplica los tratamientos necesarios para obtener tablas utilizables por los modelos. La limpieza no consiste en eliminar automáticamente cualquier registro incompleto o repetido: cada decisión atiende al significado de la variable y al proceso que pudo generar el dato.

En H&M se aplican los siguientes criterios:

- Los nulos de `FN` y `Active` se convierten en 0 porque estas variables funcionan como indicadores: el valor 1 señala explícitamente la presencia de la condición y la ausencia se interpreta como condición no registrada. El tratamiento conserva todas las filas y permite utilizar las columnas como variables binarias. Su limitación es que no puede distinguir entre un «no» real y una ausencia administrativa, por lo que deben interpretarse con prudencia.
- La edad se imputa con la mediana, igual a 32 años. Se elige la mediana porque es menos sensible que la media a edades extremas y mantiene un valor central plausible. Se añade `age_missing` para conservar la información de que el dato original estaba ausente.
- Los valores ausentes de `club_member_status`, `fashion_news_frequency` y `detail_desc` se sustituyen por `UNKNOWN`. La categoría explícita conserva los registros sin atribuirles una modalidad observada que en realidad se desconoce.
- Las filas coincidentes de `transactions_train.csv` se conservan. Eliminarlas reduciría artificialmente cantidades, cestas y gasto si representan varias unidades iguales adquiridas por el mismo cliente el mismo día. La ausencia de un identificador de pedido impide demostrar que sean errores de duplicación.

En Instacart, los nulos de `days_since_prior_order` se sustituyen por 0 únicamente cuando corresponden al primer pedido y se acompañan de `first_order`. El cero funciona como valor operativo y el indicador permite diferenciarlo de un intervalo real; no implica que hayan transcurrido cero días desde una compra anterior. El resto de las tablas no requiere imputación ni eliminación de duplicados.

Como comprobación final, se verifica que no queden nulos en los archivos destinados a las fases posteriores y que las claves maestras mantengan su unicidad. Las tablas tratadas necesarias se almacenan en `data_without_nulls/`.

## 6.2. Representación básica de H&M: RFM

La representación básica de H&M utiliza las dimensiones clásicas de recencia, frecuencia y valor monetario:

| Variable | Definición operacional |
|---|---|
| `recency` | Días entre la última compra del cliente y el final del periodo observado |
| `frequency` | Número de fechas distintas en las que el cliente compró |
| `monetary` | Suma de los precios de todos los artículos adquiridos |

El resultado contiene tres variables de modelado y el identificador `customer_id`.

Ante la ausencia de un identificador de pedido, se define operativamente una **ocasión de compra** como una fecha distinta en la que un cliente registra al menos una transacción. Esta aproximación permite calcular la frecuencia y el tamaño de cesta, aunque no garantiza que todas las transacciones del mismo día pertenezcan a una única compra real. Después de la agregación, el conjunto modelado contiene **1.362.281 clientes**.

Estas dimensiones se eligen porque describen aspectos complementarios. Una recencia baja indica actividad reciente; una frecuencia alta refleja repetición de compra; y un valor monetario alto aproxima la contribución económica acumulada. La fecha de referencia se fija al final del periodo para que todos los clientes se comparen desde el mismo instante. En frecuencia se cuentan días distintos y no líneas de transacción, pues varias prendas del mismo día deben entenderse como parte de una misma ocasión aproximada de compra.

## 6.3. Representación básica de Instacart: DFA

Al no disponer de importes, se utiliza una adaptación denominada DFA:

| Variable | Definición operacional |
|---|---|
| `days_since_previous_order` | Días desde el pedido anterior al último pedido histórico |
| `frequency` | Número de pedidos históricos |
| `avg_basket_size` | Número medio de productos por pedido |

El resultado contiene tres variables de modelado y el identificador `user_id`.

Después de la agregación a nivel de usuario, el conjunto modelado contiene **206.209 clientes**.

En Instacart, la frecuencia puede calcularse directamente a partir de pedidos identificados. `days_since_previous_order` representa el intervalo desde el pedido anterior al último pedido histórico y el tamaño medio de cesta aporta una dimensión de intensidad que sustituye, sin ser equivalente, al valor monetario ausente. Se denomina DFA para dejar claro que no se está utilizando el RFM clásico.

## 6.4. Representaciones ampliadas

El objetivo de las versiones extendidas es representar dimensiones que RFM y DFA no recogen directamente.

### 6.4.1. Variables ampliadas de H&M

| Dimensión | Variables |
|---|---|
| Cesta | `avg_basket_size`, `std_basket_size` |
| Precio | `avg_price`, `std_price` |
| Regularidad | `avg_days_between_purchases`, `std_days_between_purchases` |
| Variedad | `unique_articles`, `unique_product_types`, `unique_product_groups`, `unique_sections`, `unique_garment_groups` |
| Comportamiento relativo | `article_variety_ratio`, `channel_2_ratio` |

La versión ampliada contiene **16 variables** antes de la selección.

### 6.4.2. Variables ampliadas de Instacart

| Dimensión | Variables |
|---|---|
| Variabilidad de cesta | `std_basket_size` |
| Regularidad | `avg_days_between_orders`, `std_days_between_orders` |
| Variedad | `unique_products`, `unique_aisles`, `unique_departments` |
| Fidelidad y exploración | `reorder_ratio`, `product_variety_ratio` |

La versión ampliada contiene **11 variables**.

Las medias describen el nivel habitual de comportamiento, mientras que las desviaciones estándar representan su regularidad. Los recuentos únicos miden amplitud de consumo, y los ratios permiten comparar clientes con volúmenes totales diferentes. Por ejemplo, `channel_2_ratio` distingue preferencia de canal sin favorecer automáticamente a quien compra más, y `product_variety_ratio` relaciona exploración con el volumen total en lugar de utilizar únicamente un recuento bruto.

No obstante, algunas variables derivadas pueden compartir información. Frecuencia, artículos únicos, variedad y tamaño de cesta están relacionadas matemáticamente o por comportamiento. Esta posible redundancia explica por qué la versión extendida se considera un conjunto de candidatos que debe validarse, no una representación necesariamente superior.

## 6.5. Tratamiento de distribuciones

Las variables agregadas presentan escalas muy diferentes y, en numerosos casos, colas derechas pronunciadas. Esta situación puede hacer que las variables de mayor rango dominen las distancias. Para reducir este efecto se aplican transformaciones selectivas, evitando transformar indiscriminadamente todas las columnas.

En Instacart se aplica:

- `log1p` a `frequency`, `avg_basket_size`, `std_basket_size` y `unique_products`;
- raíz cuadrada a `unique_aisles`;
- conservación sin transformación del resto cuando no se considera necesario.

En H&M se aplica:

- `log1p` a las variables con colas derechas pronunciadas;
- raíz cuadrada a `recency`;
- Yeo–Johnson a `monetary`, `avg_price` y `std_price`;
- conservación binaria de `single_purchase`.

Durante el escalado de H&M se conservan las 16 variables del conjunto extendido, incluida `unique_garment_groups`, y se incorpora `single_purchase`, que identifica a los clientes con una única ocasión de compra. El conjunto escalado contiene así 17 variables de modelado, además de `customer_id`.

`single_purchase` resulta útil porque las variables de intervalo toman valores especiales en clientes que solo han comprado una vez: para ellos no existe una separación temporal observable. El indicador permite que el modelo distinga esta ausencia estructural de un cliente recurrente con intervalos pequeños. `unique_garment_groups` se transforma mediante `log1p`, de forma coherente con otras variables de variedad que presentan cola derecha, y se mantiene como candidata en el estudio de selección para que su aportación se determine empíricamente.

Las variables continuas se estandarizan mediante `StandardScaler`. La variable binaria `single_purchase` permanece codificada como 0 o 1. Los identificadores se reincorporan únicamente para guardar los archivos y conservar la trazabilidad.

La estandarización se ajusta después de transformar las distribuciones. Si se aplicara antes, las colas extremas seguirían condicionando la media y la desviación utilizadas por el escalador. `StandardScaler` no elimina la asimetría: únicamente centra y cambia la escala. Por ello, transformación y estandarización cumplen funciones distintas y se aplican en ese orden.

> **Figura 8. Comparación de distribuciones antes y después de las transformaciones.**  
> *[Insertar aquí una selección de histogramas de `code/02_data_preparation/03_variable_distributions.ipynb` y `code/02_data_preparation/04_scaling.ipynb`, preferentemente recencia, frecuencia y una variable de variedad de cada dataset.]*  
> **Interpretación:** la comparación debe mostrar que las transformaciones reducen las colas sin alterar el orden relativo de los clientes. No se busca obtener normalidad perfecta, sino evitar que unos pocos valores extremos dominen las distancias.

## 6.6. Prevención de errores de alineación

La asignación de etiquetas a los clientes se realiza utilizando `customer_id` o `user_id`, no la posición de las filas. Esta decisión evita errores silenciosos si un dataframe cambia de orden durante el muestreo, filtrado o transformación.

Con esta comprobación termina la preparación inicial de los datos. La selección de variables que se realiza posteriormente no forma parte de este primer recorrido: se documenta en la sección 8.4 porque se activa después del modelado y de la evaluación inicial.

---

# 7. Fase IV de CRISP-DM: modelado

## 7.1. Algoritmos evaluados

Se comparan siete algoritmos pertenecientes a familias diferentes:

| Algoritmo | Principio de funcionamiento | Interés en el estudio |
|---|---|---|
| K-Means | Minimización de distancias a centroides | Referencia clásica, eficiente y fácilmente interpretable |
| MiniBatch K-Means | Actualización de centroides mediante minilotes | Alternativa escalable para grandes volúmenes |
| Bisecting K-Means | División jerárquica mediante particiones sucesivas | Combina una estructura divisiva con K-Means |
| BIRCH | Construcción incremental de subclusters | Diseñado para grandes conjuntos de datos |
| CLARA | Aproximación de k-medoids mediante muestras | Utiliza observaciones representativas como medoides |
| Fuzzy C-Means | Asigna grados de pertenencia a varios grupos | Permite representar fronteras no completamente rígidas |
| Gaussian Mixture Model | Mezcla probabilística de distribuciones gaussianas | Modela clusters con formas y covarianzas probabilísticas |

En la implementación de BIRCH se generan primero subclusters con `n_clusters=None`. Posteriormente, sus centroides se agrupan mediante MiniBatch K-Means. Esta adaptación evita aplicar clustering aglomerativo a una cantidad excesiva de subclusters y reduce el consumo de memoria.

## 7.2. Diseño experimental inicial

Cada algoritmo se ejecuta sobre:

- H&M simple;
- H&M extendido;
- Instacart simple;
- Instacart extendido.

Esto produce **28 escenarios** de modelado. Para cada escenario se prueban nueve valores de `k`, desde 2 hasta 10, dando lugar a **252 configuraciones principales**.

Los resultados se consolidan en `results/clustering/clustering_metrics.csv`. El mecanismo de guardado permite reanudar las ejecuciones sin acumular filas duplicadas.

El rango `k=2..10` permite explorar desde una separación general de la cartera hasta una segmentación relativamente detallada. No se prueba `k=1` porque las métricas de separación requieren al menos dos grupos y una única agrupación no constituye segmentación. El límite superior controla el coste y evita generar una cantidad de perfiles difícil de interpretar comercialmente. No representa, sin embargo, la demostración de que nunca puedan existir más de diez segmentos.

La selección de familias algorítmicas responde a la necesidad de no hacer depender las conclusiones de un único supuesto geométrico. K-Means y sus variantes favorecen estructuras basadas en centroides; CLARA aporta robustez mediante medoides; BIRCH resume grandes volúmenes; GMM introduce una interpretación probabilística; y Fuzzy C-Means permite pertenencias graduales. La comparación se realiza bajo un preprocesamiento común para que las diferencias sean atribuibles, en la medida de lo posible, al modelo y no a escalas distintas.

Este primer modelado precede al estudio de selección de variables. Sus resultados no se consideran todavía la solución final, sino una referencia para comparar las representaciones simples y ampliadas y decidir si es necesario regresar a la preparación de los datos.

## 7.3. Métricas internas

La ausencia de etiquetas reales obliga a utilizar métricas internas:

| Métrica | Interpretación | Sentido óptimo |
|---|---|---|
| Silhouette | Compara cohesión interna y separación respecto a otros clusters | Mayor |
| Calinski–Harabasz | Relación entre dispersión entre grupos y dentro de los grupos | Mayor |
| Davies–Bouldin | Similitud media entre cada cluster y su vecino más próximo | Menor |
| Tiempo | Coste computacional de ajuste y evaluación | Menor, a igualdad de calidad |

El silhouette se calcula sobre una muestra máxima de **10.000 observaciones** para limitar su coste cuadrático.

Ninguna de las tres métricas principales es suficiente por separado. Silhouette puede favorecer soluciones con pocos grupos bien separados; Calinski–Harabasz puede crecer con estructuras compactas y está influida por el tamaño del conjunto; Davies–Bouldin resume semejanzas entre clusters, pero tampoco mide utilidad comercial. Su combinación reduce la dependencia respecto a un único criterio, aunque no elimina los supuestos geométricos compartidos por las métricas internas.

También se almacenan métricas específicas cuando corresponde:

- inercia para K-Means, MiniBatch K-Means y Bisecting K-Means;
- FPC, función objetivo e iteraciones para Fuzzy C-Means;
- BIC y AIC para Gaussian Mixture Model;
- número efectivo de clusters para BIRCH.

## 7.4. Construcción del ranking

Las escalas de las métricas internas son diferentes y Calinski–Harabasz depende, además, del tamaño muestral. Por ello, no se suman directamente sus valores originales. Las configuraciones se ordenan dentro de cada dataset y se combinan las posiciones normalizadas de silhouette, Calinski–Harabasz y Davies–Bouldin, invirtiendo el sentido de esta última.

La puntuación resultante facilita la comparación conjunta, pero no constituye una medida absoluta de calidad. Debe interpretarse como un ranking relativo entre las configuraciones incluidas en el experimento.

> **Figura 10. Evolución de las métricas según el número de clusters.**  
> *[Insertar aquí una figura representativa de los notebooks de `code/03_modeling/`, con silhouette, Calinski–Harabasz y Davies–Bouldin frente a `k`.]*  
> **Interpretación:** debe buscarse una configuración que muestre un compromiso entre las tres métricas, no necesariamente el óptimo aislado de una sola curva. Si las métricas discrepan, esa tensión debe mencionarse en lugar de ocultarse mediante la puntuación agregada.

---

# 8. Fase V de CRISP-DM: evaluación

La evaluación se desarrolla en dos momentos relacionados. En primer lugar, se comparan las representaciones simples y ampliadas y se identifican los subconjuntos candidatos. Esta evidencia provoca el retorno a la fase III para realizar la selección iterativa de variables. En segundo lugar, los subconjuntos preseleccionados se vuelven a modelar y se evalúan con varios algoritmos y semillas. La selección final y el perfilado se realizan únicamente después de este segundo ciclo.

## 8.1. Comparación entre representaciones simples y extendidas

La comparación inicial muestra que añadir todas las variables disponibles no mejora automáticamente la segmentación.

| Dataset | Versión | Puntuación media | Mejor puntuación | Peor puntuación |
|---|---|---:|---:|---:|
| H&M | Simple | 71,65 | 99,20 | 13,07 |
| H&M | Extendida | 28,35 | 68,00 | 0,27 |
| Instacart | Simple | 66,84 | 93,87 | 2,67 |
| Instacart | Extendida | 33,16 | 75,87 | 1,60 |

Las versiones simples obtienen una puntuación media muy superior en ambos datasets. Este resultado indica que la ampliación indiscriminada aumenta la dimensionalidad y puede introducir ruido, redundancia o variables con geometrías incompatibles con algunos algoritmos.

La conclusión no es que las variables adicionales carezcan de valor. Por el contrario, motiva la búsqueda de subconjuntos pequeños que combinen únicamente las dimensiones más informativas.

> **Figura 11. Comparación de la puntuación entre versiones simple y extendida.**  
> *[Insertar aquí el gráfico generado en `code/06_evaluation/simple_vs_extended_clustering_comparison.ipynb`.]*  
> **Interpretación:** la ventaja media de las versiones simples en los dos datasets evidencia que aumentar la dimensionalidad sin selección perjudica la separabilidad. La existencia posterior de buenos subconjuntos con variables adicionales confirma que el problema es su combinación indiscriminada y no su falta total de información.

## 8.2. Resultados generales por algoritmo

En el estudio inicial, K-Means presenta la mayor puntuación media en ambos datasets. La tabla siguiente recoge los siete algoritmos evaluados, no solo los tres métodos con mejor rendimiento:

| Dataset | Algoritmo | Puntuación media | Mejor puntuación |
|---|---|---:|---:|
| H&M | K-Means | 60,02 | 99,20 |
| H&M | MiniBatch K-Means | 57,94 | 99,20 |
| H&M | Bisecting K-Means | 51,29 | 98,40 |
| H&M | BIRCH | 50,49 | 95,73 |
| H&M | Fuzzy C-Means | 49,38 | 97,33 |
| H&M | CLARA | 47,38 | 94,67 |
| H&M | Gaussian Mixture Model | 33,51 | 96,27 |
| Instacart | K-Means | 65,75 | 93,87 |
| Instacart | MiniBatch K-Means | 61,22 | 92,27 |
| Instacart | Fuzzy C-Means | 52,30 | 90,13 |
| Instacart | BIRCH | 49,47 | 75,73 |
| Instacart | Bisecting K-Means | 47,94 | 88,13 |
| Instacart | CLARA | 47,54 | 90,13 |
| Instacart | Gaussian Mixture Model | 25,79 | 69,87 |

Este comportamiento refleja la robustez de los métodos basados en centroides sobre las representaciones analizadas. Sin embargo, la clasificación cambia al validar subconjuntos seleccionados, lo que demuestra que el rendimiento depende de la interacción entre algoritmo, variables y número de clusters.

## 8.3. Retorno a la fase III: selección iterativa de variables

La evaluación inicial muestra que la representación ampliada completa no mejora necesariamente a la representación básica. Este resultado motiva un retorno a la fase III de preparación de los datos para estudiar qué variables aportan información útil y qué combinaciones introducen redundancia o ruido.

La selección se plantea como un proceso experimental de ablación. No se presupone que los tres componentes de RFM o DFA deban mantenerse obligatoriamente en la solución final.

El notebook `code/04_iterative_data_preparation/feature_selection_study.ipynb` utiliza MiniBatch K-Means sobre una muestra reproducible de hasta **50.000 clientes**, valores de `k` entre 2 y 6 y semillas `42`, `123` y `2026`. Se evalúan:

- la base RFM/DFA;
- el conjunto extendido completo;
- cada variable individual;
- las parejas de variables básicas;
- la incorporación individual de variables adicionales a la base;
- la retirada individual de variables básicas del conjunto extendido;
- combinaciones de variables adicionales con cero, una o dos variables básicas;
- una acumulación condicionada que conserva RFM/DFA;
- una acumulación libre que parte de la variable individual mejor valorada.

Las configuraciones equivalentes se deduplican por sus columnas. Las soluciones de una o dos variables se conservan como diagnóstico, pero no pueden ser recomendaciones finales, ya que las métricas internas pueden favorecer artificialmente espacios de muy baja dimensión y producir perfiles poco ricos. Se exige un mínimo de tres variables para la recomendación.

La deduplicación es necesaria porque distintas rutas de acumulación pueden conducir al mismo conjunto de columnas. Contarlas varias veces introduciría una ponderación artificial en los rankings. Del mismo modo, la muestra común y las semillas compartidas reducen variación ajena al subconjunto evaluado: si cada alternativa utilizara clientes diferentes, parte de la diferencia podría proceder del muestreo y no de las variables.

> **Figura 9. Puntuación de los subconjuntos durante la selección de variables.**  
> *[Insertar aquí el gráfico comparativo generado en `code/04_iterative_data_preparation/feature_selection_study.ipynb` con los mejores subconjuntos de cada dataset.]*  
> **Interpretación:** la figura debe compararse con el número de variables de cada solución. El objetivo no es elegir mecánicamente la barra más alta, sino identificar un subconjunto compacto, reproducible y suficientemente rico para caracterizar clientes.

## 8.4. Resultados de la ablación y preselección de subconjuntos

El análisis individual y emparejado revela que una variable puede resultar informativa por sí sola y, al mismo tiempo, empeorar la base cuando se añade de forma directa. Esto puede deberse a redundancia o a interacciones con otras dimensiones.

Entre los resultados más destacados se encuentran:

- `std_days_between_purchases` mejora en **5,81 puntos** la base RFM de H&M;
- `unique_products` mejora en **2,48 puntos** la base DFA de Instacart;
- `unique_garment_groups` alcanza una puntuación individual media de **91,72**, pero al añadirse a RFM reduce su puntuación en **5,59 puntos** y aparece en el 100 % del tramo inferior de las configuraciones que la contienen;
- algunas variables obtienen puntuaciones individuales elevadas, pero presentan efectos negativos al añadirse a la representación básica;
- las diferencias medias por presencia son descriptivas y pueden estar condicionadas por la dimensionalidad y la ruta de selección.

Por este motivo, las conclusiones sobre contribución marginal deben basarse preferentemente en comparaciones emparejadas de adición o retirada, no solo en la puntuación media de todas las configuraciones que contienen una variable.

La primera fase recomienda:

| Dataset | Subconjunto | Número de variables | Puntuación inicial |
|---|---|---:|---:|
| H&M | `channel_2_ratio`, `std_days_between_purchases`, `avg_days_between_purchases` | 3 | 66,56 |
| Instacart | `days_since_previous_order`, `frequency`, `avg_days_between_orders` | 3 | 69,15 |

Estos resultados se obtienen con MiniBatch K-Means y funcionan como preselección, no como decisión definitiva. Representan el resultado del retorno desde la evaluación inicial a la preparación iterativa: todavía es necesario volver a ejecutar el modelado con varios algoritmos antes de seleccionar la configuración final.

## 8.5. Retorno a la fase IV: validación multimodelo y estabilidad

El notebook `code/05_modeling_validation/multimodel_subset_validation.ipynb` devuelve los cinco mejores subconjuntos de cada dataset a la fase de modelado. Estos subconjuntos se evalúan posteriormente con los siete algoritmos, valores de `k` entre 2 y 6 y las tres semillas definidas. Se utiliza una muestra común de hasta **30.000 observaciones** por dataset para que las comparaciones se realicen sobre los mismos clientes.

La estabilidad se mide mediante el **Adjusted Rand Index (ARI)** entre las particiones producidas con distintas semillas. Un valor próximo a uno indica que las asignaciones son prácticamente idénticas.

La puntuación final combina, con el mismo peso, los percentiles de:

- silhouette;
- Calinski–Harabasz;
- Davies–Bouldin en sentido inverso;
- estabilidad ARI.

Esta validación reduce el riesgo de seleccionar variables que solo funcionan bien con un algoritmo o una inicialización concreta.

Después de esta segunda etapa de modelado, la evaluación final selecciona las configuraciones que combinan mejor calidad interna, estabilidad y capacidad de interpretación. Solo entonces se generan los perfiles finales de clientes en la fase de evaluación.

El uso de las mismas semillas en todas las configuraciones garantiza una comparación emparejada. Promediar las métricas reduce el peso de una inicialización afortunada, mientras que el ARI comprueba si los clientes permanecen juntos al repetir el ajuste. Se registran también las semillas ejecutadas correctamente y los posibles errores, de modo que una configuración incompleta no se confunda con una validación completa.

> **Figura 12. Comparación multimodelo de los subconjuntos candidatos.**  
> *[Insertar aquí el gráfico de puntuaciones por modelo y subconjunto de `code/05_modeling_validation/multimodel_subset_validation.ipynb`.]*  
> **Interpretación:** un buen subconjunto debería mantener resultados competitivos en varios modelos. Una puntuación excepcional restringida a un único algoritmo sería una evidencia más débil de que las variables representan una estructura general de los datos.

## 8.6. Configuraciones finales

| Dataset | Variables seleccionadas | Modelo | `k` | Silhouette | Calinski–Harabasz | Davies–Bouldin | ARI | Puntuación |
|---|---|---|---:|---:|---:|---:|---:|---:|
| H&M | `channel_2_ratio`, `std_days_between_purchases`, `avg_days_between_purchases` | Fuzzy C-Means | 5 | 0,652 | 55.333,59 | 0,567 | 1,000 | 97,21 |
| Instacart | `days_since_previous_order`, `frequency`, `avg_days_between_orders`, `std_days_between_orders` | Fuzzy C-Means | 2 | 0,406 | 29.412,56 | 0,954 | 1,000 | 97,00 |

Fuzzy C-Means obtiene la mejor configuración individual en ambos datasets. El ARI igual a uno indica que estas soluciones son estables frente a las semillas probadas. Aun así, no debe confundirse estabilidad aleatoria con estabilidad temporal o muestral.

## 8.7. Rendimiento medio en la validación multimodelo

En H&M, K-Means alcanza la mayor puntuación media (**65,48**), seguido de Fuzzy C-Means (**63,57**). En Instacart, Fuzzy C-Means (**68,75**) y K-Means (**68,24**) quedan prácticamente igualados.

Esto permite distinguir dos conceptos:

- **mejor caso individual**: Fuzzy C-Means produce la configuración ganadora;
- **robustez media**: K-Means mantiene un rendimiento muy competitivo y lidera el promedio de H&M.

BIRCH, GMM y CLARA aparecen con más frecuencia en posiciones bajas del ranking multimodelo. Esta observación se limita a las variables, escalas e hiperparámetros estudiados y no implica que sean métodos inadecuados en otros problemas.

> **Figura 13. Rendimiento medio y estabilidad por algoritmo.**  
> *[Insertar aquí los gráficos de comparación algorítmica de `code/06_evaluation/simple_vs_extended_clustering_comparison.ipynb` o `code/06_evaluation/best_clustering_interpretation.ipynb`.]*  
> **Interpretación:** la figura debe separar rendimiento medio, mejor caso y estabilidad. Fuzzy C-Means gana las configuraciones finales, pero K-Means es especialmente competitivo en promedio; ambas afirmaciones son compatibles y describen criterios distintos.

## 8.8. Interpretación de la pertenencia difusa

Fuzzy C-Means asigna a cada cliente un grado de pertenencia a cada cluster. Para obtener perfiles y tamaños discretos puede utilizarse el cluster de pertenencia máxima, pero se conserva conceptualmente la posibilidad de que un cliente se encuentre cerca de la frontera entre varios segmentos.

La superioridad de Fuzzy C-Means en las dos mejores configuraciones sugiere que el comportamiento comercial no siempre forma grupos completamente rígidos. Esta es una interpretación coherente con la naturaleza gradual de la actividad, la frecuencia y la preferencia de canal, aunque requiere validación externa antes de convertirse en una conclusión de negocio.

## 8.9. Perfil final de H&M

La solución final contiene cinco clusters:

| Cluster | Clientes | Porcentaje | Recencia | Frecuencia | Valor monetario | Canal 2 | Interpretación descriptiva |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 444.658 | 32,64 % | 122,13 | 11,66 | 1,355 | 0,877 | Clientes activos y frecuentes, con fuerte preferencia por el canal 2 |
| 1 | 156.305 | 11,47 % | 442,49 | 1,21 | 0,096 | 0,011 | Clientes antiguos y casi inactivos, asociados principalmente al canal 1 |
| 2 | 350.004 | 25,69 % | 361,51 | 1,37 | 0,136 | 0,997 | Clientes esporádicos y poco recientes, casi exclusivos del canal 2 |
| 3 | 292.038 | 21,44 % | 139,19 | 10,22 | 0,665 | 0,188 | Clientes activos y frecuentes, con mayor peso del canal 1 |
| 4 | 119.276 | 8,76 % | 260,33 | 2,04 | 0,213 | 0,896 | Clientes ocasionales, irregulares y orientados al canal 2 |

Los nombres anteriores son etiquetas interpretativas propuestas a partir de las medias y no categorías originales del dataset.

La geometría del modelo se construye exclusivamente con `channel_2_ratio`, `avg_days_between_purchases` y `std_days_between_purchases`. La recencia, frecuencia, valor monetario, cesta, precio y variedad se utilizan después para enriquecer la descripción. Por tanto, las diferencias observadas en estas últimas variables caracterizan los segmentos, pero no causan directamente su formación.

La solución distingue especialmente dos ejes:

1. **canal de compra**, con grupos concentrados en el canal 1 o el canal 2;
2. **regularidad temporal**, que separa clientes frecuentes, esporádicos y con intervalos prolongados.

Resulta notable que ninguna variable RFM forme parte del subconjunto ganador. RFM sigue siendo una referencia útil para interpretar los grupos, pero no es imprescindible para obtener la mejor estructura interna encontrada en H&M.

> **Figura 14. Tamaño y perfil normalizado de los cinco clusters de H&M.**  
> *[Insertar aquí los gráficos de tamaños y mapa de calor generados en `code/06_evaluation/best_clustering_interpretation.ipynb`.]*  
> **Interpretación:** el gráfico de tamaños muestra que la solución no está dominada por un único grupo, aunque el cluster 4 representa solo el 8,76 %. El mapa de calor debe leerse por filas y variables: permite identificar diferencias relativas, pero sus valores estandarizados no sustituyen las medias originales de la tabla.

## 8.10. Perfil final de Instacart

La solución final contiene dos clusters:

| Cluster | Clientes | Porcentaje | Recencia | Frecuencia | Intervalo medio | Variabilidad temporal | Interpretación descriptiva |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 113.059 | 54,83 % | 23,57 | 7,50 | 17,21 | 11,88 | Clientes ocasionales, menos recientes y con compras menos regulares |
| 1 | 93.150 | 45,17 % | 9,17 | 25,41 | 8,84 | 6,56 | Clientes recientes, frecuentes y con ciclos de compra más regulares |

El segundo cluster también presenta, de forma descriptiva, mayor variedad de productos, mayor proporción de recompra y una cesta media ligeramente superior. Estas variables no construyen la solución final, pero ayudan a comprender las diferencias comerciales entre los grupos.

En Instacart se conservan `days_since_previous_order` y `frequency`, mientras que `avg_basket_size` se sustituye por el promedio y la desviación de los días entre pedidos. La regularidad aporta así más información para la estructura seleccionada que el tamaño medio de cesta.

> **Figura 15. Tamaño y perfil normalizado de los dos clusters de Instacart.**  
> *[Insertar aquí los gráficos de tamaños y mapa de calor generados en `code/06_evaluation/best_clustering_interpretation.ipynb`.]*  
> **Interpretación:** los tamaños son relativamente equilibrados, 54,83 % y 45,17 %. El contraste principal enfrenta clientes menos recientes y ocasionales con clientes recientes, frecuentes y regulares. Las diferencias en variedad y recompra son descriptivas, porque esas variables no participaron en el ajuste final.

## 8.11. Visualización mediante PCA

La interpretación final utiliza dos proyecciones PCA sobre los mismos clientes y etiquetas:

- una proyección en el espacio del subconjunto seleccionado;
- una proyección descriptiva sobre todas las variables escaladas.

La primera ayuda a visualizar la geometría aproximada utilizada por el modelo. La segunda permite estudiar cómo se distribuyen los clusters respecto al conjunto completo de características. Ninguna sustituye las métricas calculadas en el espacio original, y la segunda no implica que las variables externas hayan intervenido en la construcción de los grupos.

> **Figura 16. Proyecciones PCA de los clusters finales.**  
> *[Insertar aquí, para cada dataset, la PCA del subconjunto seleccionado y la PCA de todas las variables generadas en `code/06_evaluation/best_clustering_interpretation.ipynb`.]*  
> **Interpretación:** una separación visual clara respalda la legibilidad de la solución, pero el solapamiento bidimensional no demuestra necesariamente una mala segmentación, ya que la proyección descarta parte de la información. La PCA de todas las variables es exclusivamente descriptiva.

---

# 9. Fase VI de CRISP-DM: despliegue y transferencia

## 9.1. Situación actual

El proyecto no incluye un despliegue productivo. Su equivalente académico consiste en:

- conservar los datasets preparados;
- consolidar las métricas y rankings en archivos CSV;
- documentar las decisiones metodológicas;
- generar perfiles interpretables de los clusters;
- proponer usos y futuras validaciones de negocio.

## 9.2. Propuesta de utilización

Una transferencia a un entorno real podría seguir estas etapas:

1. acordar con responsables de negocio nombres y significado operativo de los segmentos;
2. definir campañas o tratamientos diferenciados para cada perfil;
3. guardar los transformadores y el modelo entrenado;
4. desarrollar un proceso para calcular las variables de nuevos clientes;
5. asignar pertenencias periódicamente y monitorizar sus cambios;
6. medir resultados mediante retención, conversión, frecuencia o margen;
7. reentrenar cuando se detecten cambios relevantes en las distribuciones.

En H&M podrían estudiarse estrategias diferenciadas por actividad y canal. En Instacart podrían plantearse acciones de reactivación para el segmento ocasional y programas de fidelización para los compradores recurrentes. Estas propuestas deben validarse mediante experimentos controlados antes de considerarse efectivas.

## 9.3. Monitorización recomendada

Un despliegue debería controlar:

- cambios en las distribuciones de entrada;
- proporción de clientes asignada a cada segmento;
- intensidad máxima y ambigüedad de las pertenencias difusas;
- estabilidad temporal de los perfiles;
- evolución de indicadores comerciales por cluster;
- calidad de los datos y disponibilidad de todas las variables.

---

# 10. Discusión

## 10.1. Dimensionalidad y calidad del clustering

El resultado más consistente del estudio es que una representación más extensa no garantiza una segmentación mejor. Las versiones completas obtienen puntuaciones medias inferiores a las versiones simples en los dos datasets. Sin embargo, la selección posterior identifica variables adicionales muy relevantes, especialmente las relacionadas con la regularidad temporal y el canal.

Por tanto, el problema no es la ingeniería de características en sí, sino la inclusión conjunta de variables redundantes o poco compatibles con la estructura buscada. La selección de subconjuntos actúa como una etapa necesaria entre la creación de variables y el modelado final.

## 10.2. Valor de RFM y DFA

RFM y DFA proporcionan bases compactas, conocidas e interpretables. Su buen rendimiento medio confirma su utilidad como punto de partida. No obstante, la ablación demuestra que sus componentes no deben imponerse de forma dogmática:

- H&M obtiene su mejor resultado sin ninguna variable RFM;
- Instacart mantiene recencia y frecuencia, pero descarta el tamaño medio de cesta;
- la regularidad temporal aparece en los dos subconjuntos finales;
- el canal resulta determinante en H&M.

La principal aportación metodológica consiste, por tanto, en conservar RFM/DFA como referencia teórica y permitir simultáneamente que la evidencia experimental proponga combinaciones alternativas.

## 10.3. Algoritmos rígidos y pertenencia gradual

K-Means ofrece un rendimiento medio sólido y un coste reducido, mientras que Fuzzy C-Means logra los mejores casos finales. Esta combinación sugiere dos recomendaciones:

- K-Means es una referencia eficiente y robusta para comparaciones generales;
- Fuzzy C-Means resulta especialmente interesante cuando se desea representar clientes con pertenencias intermedias.

La elección práctica dependerá de si el sistema consumidor puede utilizar grados de pertenencia. Si solo admite etiquetas únicas, la ventaja conceptual del enfoque difuso se reduce, aunque todavía puede conservarse la pertenencia máxima junto con una medida de confianza.

## 10.4. Generalización entre datasets

Los dos casos comparten la importancia de la regularidad temporal, pero difieren en el resto de la solución:

- H&M se organiza principalmente por canal y pautas temporales;
- Instacart se organiza por actividad, recencia y regularidad;
- el número final de grupos es cinco en H&M y dos en Instacart.

No existe, por tanto, una segmentación universal aplicable sin adaptación. El procedimiento general es transferible, pero las variables y el número de grupos dependen del dominio.

---

# 11. Limitaciones

1. **Ausencia de etiquetas reales.** Las métricas internas no prueban que los clusters sean comercialmente útiles.
2. **Falta de validación temporal.** La estabilidad entre semillas no garantiza que los segmentos persistan en otros periodos.
3. **Uso de muestras.** Parte de la selección y validación se realiza sobre muestras por restricciones computacionales.
4. **Dependencia de las decisiones experimentales.** Los resultados están condicionados por transformaciones, algoritmos, hiperparámetros y rangos de `k`.
5. **Aproximación de pedidos en H&M.** Agrupar por día puede unir compras diferentes o separar de forma imperfecta una ocasión real.
6. **Ausencia de importe en Instacart.** No es posible construir una dimensión monetaria equivalente a la de H&M.
7. **Métricas internas sensibles a la geometría.** Pueden favorecer clusters compactos o espacios de baja dimensión sin asegurar relevancia empresarial.
8. **Comparabilidad limitada.** Los valores absolutos de los dos datasets no representan exactamente los mismos conceptos.
9. **PCA únicamente descriptivo.** Una proyección bidimensional puede ocultar estructura o generar una percepción visual simplificada.
10. **Sin evaluación de impacto.** No se dispone de campañas, retención, margen u otros indicadores posteriores a la segmentación.

---

# 12. Conclusiones

El trabajo desarrolla un procedimiento completo de segmentación de clientes basado en CRISP-DM y lo aplica a más de 1,5 millones de clientes pertenecientes a dos contextos comerciales. La comparación de 252 configuraciones iniciales demuestra que las representaciones básicas superan, en promedio, a las versiones extendidas completas. Este resultado confirma que añadir variables sin selección puede deteriorar la estructura interna del clustering.

El análisis exploratorio desempeña un papel operativo en esta secuencia. La asimetría de la actividad y las cestas conduce a transformar variables; la falta de identificador de pedido en H&M obliga a definir ocasiones de compra por día; la recurrencia de Instacart motiva las medidas de intervalo y recompra; y las diferencias de canal y categoría justifican ampliar las representaciones básicas. Así, las decisiones de preparación no se introducen de forma arbitraria, sino como respuesta a propiedades observadas en los datos.

La iteración entre modelado y preparación permite superar esta limitación. El estudio de ablación identifica subconjuntos compactos y la validación multimodelo comprueba su comportamiento con siete algoritmos y varias semillas. Las configuraciones finales alcanzan puntuaciones relativas de **97,21** en H&M y **97,00** en Instacart, ambas mediante Fuzzy C-Means y con estabilidad ARI igual a uno en las semillas evaluadas.

En H&M, la mejor representación está compuesta por la preferencia de canal y dos medidas de regularidad temporal. Esto muestra que la estructura interna más clara no depende necesariamente de las variables RFM tradicionales. En Instacart, recencia y frecuencia permanecen, pero la dimensión de cesta es sustituida por información sobre los intervalos entre pedidos. La regularidad temporal aparece así como el elemento común más relevante entre ambos dominios.

Desde el punto de vista algorítmico, K-Means mantiene un rendimiento medio muy competitivo, mientras que Fuzzy C-Means obtiene las mejores soluciones individuales. La pertenencia difusa resulta conceptualmente apropiada para comportamientos comerciales con fronteras graduales, en los que un cliente puede compartir características con varios perfiles.

La principal conclusión metodológica es que la calidad de una segmentación no depende únicamente del algoritmo. Resulta de la interacción entre la representación del cliente, el preprocesamiento, la selección de variables, el número de clusters y el criterio de evaluación. Un enfoque iterativo, multimodelo y acompañado de análisis de estabilidad ofrece una base más sólida que la selección de una única configuración a partir de una sola métrica.

Las soluciones obtenidas son técnicamente coherentes e interpretables, pero su utilidad empresarial debe considerarse todavía una hipótesis. El siguiente paso necesario es validar la persistencia temporal de los segmentos y comprobar, mediante indicadores externos o experimentos comerciales, si permiten mejorar decisiones reales.

---

# 13. Líneas futuras

- incorporar validación temporal y analizar las transiciones entre segmentos;
- estudiar estabilidad mediante remuestreo o *bootstrap*;
- ajustar hiperparámetros específicos de BIRCH, CLARA, Fuzzy C-Means y GMM;
- explorar UMAP u otras técnicas no lineales como apoyo visual, no como sustituto de la evaluación;
- guardar transformadores y modelos para permitir inferencia reproducible;
- automatizar el flujo completo mediante un pipeline ejecutable;
- fijar versiones de dependencias y añadir pruebas para las funciones compartidas;
- estudiar la intensidad de pertenencia difusa y detectar clientes fronterizos;
- incorporar variables temporales, estacionales o de evolución reciente;
- contrastar los perfiles con expertos de negocio;
- ejecutar campañas controladas y medir conversión, recurrencia, retención o rentabilidad.

---

# 14. Correspondencia entre CRISP-DM y los archivos del proyecto

| Fase | Implementación principal |
|---|---|
| Comprensión del negocio | Planteamiento, objetivos, criterios y aplicaciones descritos en la documentación |
| Comprensión de los datos | `code/01_data_understanding/` |
| Preparación inicial de los datos | `code/02_data_preparation/` |
| Modelado inicial | `code/03_modeling/` |
| Preparación iterativa | `code/04_iterative_data_preparation/` |
| Modelado y validación | `code/05_modeling_validation/` |
| Evaluación inicial y final | `code/06_evaluation/` y `results/` |
| Despliegue | Propuesta académica de transferencia, todavía no implementada en producción |

## 14.1. Evidencias de resultados

| Archivo | Función |
|---|---|
| `results/clustering/clustering_metrics.csv` | Métricas de las 252 configuraciones iniciales |
| `results/feature_selection/feature_selection_results.csv` | Resultados del estudio de ablación |
| `results/feature_selection/recommended_features.csv` | Preselección de variables con MiniBatch K-Means |
| `results/multimodel/multi_model_feature_validation_detail.csv` | Resultados por modelo, `k` y semilla |
| `results/multimodel/multi_model_feature_validation.csv` | Agregación multimodelo y estabilidad |
| `results/multimodel/multi_model_recommended_features.csv` | Configuraciones finales recomendadas |
| `results/profiles/hym_best_clustering_cluster_profiles.csv` | Perfil completo de los clusters finales de H&M |
| `results/profiles/instacart_best_clustering_cluster_profiles.csv` | Perfil completo de los clusters finales de Instacart |

---

# 15. Nota sobre la interpretación de los resultados

Las puntuaciones generales utilizadas en este trabajo son rankings relativos construidos a partir de percentiles o posiciones normalizadas. No deben interpretarse como porcentajes de exactitud. Del mismo modo, los nombres descriptivos asignados a los clusters son propuestas de interpretación basadas en medias agregadas y no etiquetas proporcionadas por los datasets.

La caracterización mediante variables no usadas por el modelo permite enriquecer los perfiles, pero debe distinguirse siempre entre:

- **variables de construcción**, que determinan la geometría de los clusters;
- **variables de caracterización**, que describen diferencias posteriores entre los grupos.

Esta separación es esencial para evitar conclusiones causales que el diseño no permite sostener.

## Actualización: comparación de alternativas finales

La puntuación conjunta se conserva como referencia, pero no se interpreta como prueba de una superioridad clara. Diferencias mínimas de ARI pueden modificar el percentil de estabilidad y, por tanto, la posición relativa. Para estudiar esta limitación se añadió `code/06_evaluation/final_alternatives_analysis.py`, que compara seis configuraciones previamente verificadas: tres en H&M y tres en Instacart.

El análisis nuevo no reemplaza los CSV de `results/multimodel/` ni los perfiles de `results/profiles/`. Genera una carpeta separada, `results/final_alternatives/`, con una copia filtrada de los candidatos, la sensibilidad de la puntuación cuando el ARI pesa 0 %, 10 % o 25 %, y la comparación de las particiones ajustadas con todos los clientes.

La interpretación distingue entre métricas internas sobre la muestra común, estabilidad entre semillas, perfiles del ajuste completo y concordancia entre configuraciones diferentes. En Fuzzy C-Means, los perfiles discretos se construyen mediante la pertenencia máxima (`argmax`). La concordancia se calcula sobre los mismos identificadores y se acompaña de tablas de contingencia; no se presupone que el grupo 0 de un modelo equivalga al grupo 0 de otro.

En H&M se estudia si la solución de seis grupos subdivide perfiles de la solución de cinco. En Instacart se analiza el efecto de incorporar `std_days_between_orders` y de cambiar de algoritmo. Los pesos del ARI son escenarios de sensibilidad, no pesos óptimos ni un criterio para declarar automáticamente un ganador.

Por tanto, los resultados antiguos siguen siendo necesarios para documentar el estudio multimodelo completo, sus semillas y el contexto de todos los candidatos. `results/final_alternatives/` aporta una vista específica para las soluciones discutidas en la interpretación final.
