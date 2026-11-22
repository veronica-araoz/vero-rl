# Decision Transformer e IQL para recomendaciones — Reporte

DiploDatos 2026 · Aprendizaje por Refuerzos · Dataset: **Netflix** (752 películas, 8 grupos)

## 1. Problema y datos

Se recomiendan 20 películas a un usuario, de a una y sin repetir. Cada rating (1–5) se revela al mostrar la película y es la recompensa. El objetivo es maximizar la suma. Se aprende **sólo de logs** (RL offline).

**Datos.** Train: 16.000 usuarios con entre 25 y 200 ratings en el orden registrado. Test: 1.600 usuarios nuevos con rating para todo el catálogo. Los datos son sintéticos: el rating es μ[grupo, ítem] + ruido.

**Exploración** (notebook 01):
- El rating medio no cambia con la posición (≈ 3,22 en todas).
- Las apariciones de un ítem no se relacionan con su rating medio (correlación −0,04; cada ítem aparece entre 2.011 y 2.288 veces).
- Conclusión: los logs vienen de una **política uniforme**.
- Los grupos se parecen (correlación media de 0,58 entre los ratings medios por ítem), pero un mismo ítem difiere en promedio 2,5 puntos entre el mejor y el peor grupo. El grupo 5 es el más distinto.
- Conclusión: **personalizar por grupo es lo que hay para ganar**.

**Particiones.** `split_by_user` separa usuarios (90/10, estratificado por grupo, semilla 0); se verificó que train y validación no comparten usuarios. Validación se usa para elegir hiperparámetros. Test se usó una sola vez, para las tablas finales.

**Episodios y retorno.** Tramos de 20 interacciones consecutivas del mismo usuario (73.827 episodios de train). El retorno pendiente $\hat R_t = \sum_{k=t}^{19} r_k$ se calcula dentro del episodio. Su escala (26–97, media 64) coincide con la de una sesión evaluada.

**MDP.**
- Estado $s_t$: grupo (o "desconocido"), pares (ítem, rating) ya revelados y paso t. No incluye $r_t$.
- Acción: un ítem no mostrado.
- Recompensa: el rating.
- Horizonte: 20; el paso 19 es terminal.
- Descuento: el protocolo evalúa la suma sin descontar; γ se elige en IQL.

El rating no depende de lo mostrado antes. Lo único que una acción cambia del futuro es **lo que la política sabe** del usuario: inferir su grupo cuando está oculto.

## 2. Protocolo de evaluación

Se usa el protocolo común (`protocolo/`, sin modificar):
- Sesiones simuladas de 20 pasos sobre usuarios de test.
- Métrica principal: **retorno normalizado** (0 = azar, 1 = oráculo). Secundarias: hit rate y NDCG@10 del primer paso.
- Tres condiciones: grupo visible, grupo oculto (cold-start estricto) y warm-up de 5 ítems al azar (no es cold-start).
- 3 semillas de evaluación.

**Líneas base:** `random`, `popularity`, `item_mean` (sin personalización), `group_mean` (por segmento) y `group_posterior` (infiere el grupo por Bayes: personalización adaptativa). Sus números coinciden con la tabla de `PROTOCOLO.md`, lo que valida la instalación y los datos.

Para acelerar, los modelos se evalúan con `src/utils.evaluate_batched`. Es una reimplementación por lotes de `run_session` con las mismas semillas, warm-up, desempate y métricas. Un test verifica que da **exactamente** lo mismo que `protocolo.evaluar.evaluate` (`tests/test_modelos.py`).

## 3. Modelos

### 3.1 Decision Transformer (`src/dt.py`)

**Tokens.** Por paso: $(\hat R_t, s_t, a_t)$.
- $\hat R_t/100$ pasa por una capa lineal.
- $s_t$ = embedding del grupo (9 valores, incluido "desconocido") + embedding de $r_{t-1}$ ("ninguno" en t = 0).
- $a_t$ = embedding del ítem.
- A cada token se le suman un embedding de paso y uno de tipo.

**Arquitectura y predicción.**
- Transformer causal de 2 capas, d = 64, 4 cabezas, sobre 60 tokens.
- $a_t$ se predice **desde la salida de $s_t$**, que no ve $a_t$.
- Logits con pesos compartidos con los embeddings de ítems.
- Los ítems ya mostrados se enmascaran con −∞.
- No hace falta padding: todos los episodios miden 20. Al recomendar, la posición de $a_t$ lleva un token de relleno que $s_t$ no ve.

**Test de no fuga.** Al cambiar $a_{\ge t}$, $\hat R_{>t}$ y $s_{>t}$, las logits de t no cambian (diferencia 0,0).

**Entrenamiento.**
- 30 % de episodios con grupo "desconocido".
- AdamW con OneCycle, 6 épocas.
- Checkpoint con mínima CE en validación.

**Inicialización.** Con pesos compartidos, embeddings N(0, 1) daban logits enormes: la CE arrancaba en 8,1, por encima de la uniforme. Se corrigió con inicialización N(0; 0,02).

**CE de validación.** 6,589 / 6,601 / 6,601 según la semilla, frente a 6,609 de la uniforme sobre los ítems no mostrados. Con logs aleatorios el ítem siguiente es casi impredecible: la pérdida no mide la calidad.

**Al recomendar.** $\hat R_t$ = objetivo − ratings ya revelados, incluidos los del warm-up.

### 3.2 IQL (`src/iql.py`)

**Redes.** Q, V y π tienen **cada una su propio codificador**: una GRU sobre los pares (ítem, rating) previos, más el embedding del grupo y del paso. El codificador devuelve $h_0 … h_{20}$.
- **V:** regresión de expectil τ sobre $\bar Q(s,a)$ de la acción registrada.
- **Q:** objetivo TD $r_t + γ V(s_{t+1})$, con V sin gradiente. En t = 19 el objetivo es sólo $r_t$.
- **π:** AWR con pesos $\min(e^{β(\bar Q − V)}, 100)$, sin gradiente; los ítems mostrados se enmascaran.
- **$\bar Q$:** red objetivo con Polyak, λ = 0,005.

**Decisiones de entrenamiento.**
- Recompensa centrada: (r − 3)/2.
- 30 % de episodios con grupo "desconocido".
- Adam por red, 5 épocas, d = 64.
- Tests: expectil (τ = 0,5 → 4,0 y τ = 0,9 → 8,0 en el ejemplo de la guía), separación de gradientes y objetivo terminal.

**Selección en validación** (π, warm-up 0, visible y oculto). Se barrió τ ∈ {0,7; 0,9} × β ∈ {3; 10} × γ ∈ {0; 0,9}.
- Todas las configuraciones quedaron entre 0,595 y 0,600, una diferencia menor que el ruido.
- Elegida: **γ = 0,9, τ = 0,7, β = 10**. Para γ = 0, la mejor fue la misma τ y β.
- Con β = 10, alrededor del 5 % de los pesos queda truncado y el peso medio es ≈ 10. π es selectiva sin colapsar.
- Como era de esperar, $L_V$ y $L_π$ suben durante el entrenamiento mientras $L_Q$ baja.

## 4. Resultados en test (logs originales)

Normalizado. Celdas de los modelos: media de 3 semillas de entrenamiento ± desvío entre semillas. El IC 95 % por usuario es ≈ ±0,005–0,007 en todas las celdas (tablas completas en notebook 04).

| Política | Visible | Oculto | Warm-up 5, visible | Warm-up 5, oculto |
|---|---:|---:|---:|---:|
| `random` | 0,00 | 0,00 | 0,00 | 0,00 |
| `popularity` | −0,058 | −0,058 | −0,117 | −0,117 |
| `item_mean` | 0,689 | 0,689 | 0,705 | 0,705 |
| `group_mean` | **0,742** | 0,689 | **0,749** | 0,705 |
| `group_posterior` | **0,742** | **0,707** | **0,749** | **0,729** |
| DT (R̂ = 107) | 0,655 ± 0,020 | 0,657 ± 0,018 | 0,662 ± 0,027 | 0,667 ± 0,025 |
| IQL π (γ = 0,9) | 0,697 ± 0,002 | 0,631 ± 0,029 | 0,704 ± 0,001 | 0,666 ± 0,028 |
| IQL máx Q (γ = 0,9) | 0,705 ± 0,004 | 0,616 ± 0,008 | 0,713 ± 0,004 | 0,652 ± 0,008 |
| IQL π (γ = 0) | 0,700 ± 0,004 | 0,625 ± 0,024 | 0,701 ± 0,003 | 0,672 ± 0,017 |
| IQL máx Q (γ = 0) | 0,719 ± 0,003 | 0,637 ± 0,004 | 0,729 ± 0,002 | 0,671 ± 0,005 |

**Lectura.**
- **Ningún modelo supera a `group_mean` ni a `group_posterior`.** Con grupo visible, IQL supera a `item_mean` (máx Q γ = 0: +0,030, mayor que la dispersión), así que personaliza, pero no alcanza a la media por grupo.
- **La explicación es que la información útil es una tabla de 8 × 752 medias.** Las líneas base la estiman directamente promediando ~1.800 ratings por celda. Las redes la aproximan con ruido de entrenamiento y menos épocas.
- **Con grupo oculto, IQL queda por debajo de `item_mean`.** El diagnóstico por paso (notebook 03, §3.5) lo explica:
  - En el paso 0, IQL obtiene lo mismo que `item_mean` (4,57).
  - Después de ver un solo rating cae (4,39 frente a 4,60).
  - El codificador **sobrerreacciona** a cada rating, como si ya supiera el grupo. `group_posterior` hace esa inferencia de forma bayesiana y gana.
  - Aprender esa inferencia sólo con el 30 % de episodios sin grupo no alcanzó.
- **El DT rinde igual con el grupo visible que oculto (0,655 frente a 0,657), por debajo de `item_mean`.** Con logs uniformes, lo único que distingue a los episodios de retorno alto es que contienen ítems buenos *en general*. La interacción con el grupo es una señal de segundo orden que no aprendió: el desempeño por grupo es igual en ambas condiciones.
- **γ = 0 frente a γ = 0,9.** El máximo de Q con γ = 0 es mejor en las cuatro condiciones (+0,014 a +0,021, mayor que la dispersión); π queda igual dentro de la dispersión. El problema casi no tiene valor secuencial: el rating no depende del pasado. γ > 0 sólo agrega al objetivo de Q el ruido del bootstrapping ($L_Q$ 0,29 frente a 0,24). El único valor futuro posible, informativo con grupo oculto, no aparece: γ = 0,9 no mejora la condición oculta.
- **π frente al máximo de Q.** El máximo de Q es mejor que π con grupo visible (0,705 frente a 0,697 con γ = 0,9; 0,719 frente a 0,700 con γ = 0). Con logs uniformes cada ítem aparece unas 2.000 veces, así que Q está bien estimada en todo el catálogo y **no hay error de extrapolación** que evitar. π, en cambio, se entrena con una cross-entropy ponderada sobre acciones casi aleatorias, que es más ruidosa. La sección 6 muestra el caso contrario.

## 5. Return conditioning (DT, test, warm-up 0)

Los retornos pedidos van de 40 a 107 e incluyen los percentiles de train (soporte de train: 26–97). Se usan los 3 modelos (notebook 05).

| Pedido | 40 | 50 | 60 | 70 | 80 | 90 | 97 (máx. train) | 107 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Obtenido, visible | 44,8 | 51,6 | 60,7 | 70,0 | 80,2 | 86,2 | 87,1 | 87,1 |
| Obtenido, oculto | 46,5 | 53,4 | 61,5 | 70,3 | 79,1 | 86,0 | 87,2 | 87,2 |
| Normalizado (media) | −0,64 | −0,46 | −0,22 | 0,06 | 0,36 | 0,60 | 0,66 | 0,66 |

- **Entre 50 y 80, el obtenido sigue la diagonal casi exactamente.** El DT entendió el condicionamiento: pedirle poco hace que recomiende **mal a propósito** (normalizado negativo).
- **Por encima de ~85 se aplana en ≈ 87.** Los retornos de train mayores que 90 son muy raros (percentil 99 = 89): no hay ejemplos para aprender cómo lograrlos.
- **Fuera del soporte (107) no extrapola.** Se comporta como en el máximo. Por eso validación eligió 107, empatado con 97 (0,537 frente a 0,536): no es una mejora real.
- **El mejor retorno pedido no es "el más alto posible" (100), sino cualquiera en la meseta.**
- **Por grupo (grupo visible):** el retorno pedido está en escala absoluta, y la misma suma significa cosas distintas según el grupo.
  - Grupo 7 (el más exigente): pedir 40 es pedir un retorno "normal", así que queda en el nivel del azar (normalizado −0,01) y llega a su meseta (≈ 0,54) ya con 80.
  - Grupo 4: pedir 40 es pedir algo muy por debajo de lo habitual. El DT recomienda lo peor para ese grupo (−1,05 hasta pedidos de 60) y recién alcanza su meseta (0,67) con 97.
- **Incertidumbre:** el desvío entre semillas del obtenido es de 0,1 a 1,3 puntos. El IC 95 % por usuario del normalizado va de ±0,02 (pedidos bajos) a ±0,005 (meseta).

## 6. Calidad de los logs (`mixto`)

**Los logs `mixto`.** Los generó el sistema anterior con ε ∈ {0; 0,25; 0,5; 1} y τ = 0,25.
- Retorno medio de un episodio: 74,6, frente a 64,3 de los originales.
- Las apariciones de un ítem ahora correlacionan con su rating medio observado (0,75). Cada ítem aparece entre 807 y 8.622 veces.

**Qué se cambió.** DT e IQL se reentrenaron **sin cambiar nada**: mismas arquitecturas, mismos τ, β y γ. Sólo se volvió a elegir en validación el retorno pedido al DT: 99, el máximo de train. Las líneas base se ajustaron con los mismos logs.

| Política (test) | Visible | Oculto | WU5 visible | WU5 oculto |
|---|---:|---:|---:|---:|
| `popularity` | 0,688 | 0,688 | 0,704 | 0,704 |
| `item_mean` | 0,672 | 0,672 | 0,695 | 0,695 |
| `group_mean` | **0,741** | 0,672 | **0,750** | 0,695 |
| `group_posterior` | **0,741** | **0,705** | **0,750** | **0,727** |
| Sistema anterior (ε = 0) | 0,593 | 0,542 | 0,595 | 0,546 |
| DT (R̂ = 99) | 0,726 ± 0,002 | 0,684 ± 0,006 | 0,736 ± 0,001 | 0,666 ± 0,019 |
| IQL π | 0,737 ± 0,001 | 0,679 ± 0,005 | 0,745 ± 0,001 | 0,706 ± 0,002 |
| IQL máx Q | 0,696 ± 0,007 | 0,588 ± 0,011 | 0,711 ± 0,005 | 0,652 ± 0,011 |

**Qué aprende cada modelo.**
- **DT:** con logs uniformes, imitar era casi inútil. Ahora la política de registro elige según el grupo, y $P(a \mid g)$ ya contiene "qué le gusta a cada grupo".
  - La CE baja a 6,15, frente a 6,59 con los logs originales.
  - El DT pasa de 0,655 a 0,726 con grupo visible y ahora sí distingue visible de oculto.
  - Condicionar por un retorno alto filtra las sesiones explotadoras (ε = 0, retorno medio 82) frente a las aleatorias (64).
- **IQL π:** sube de 0,697 a 0,737 con grupo visible, prácticamente igual a `group_mean` (0,741). AWR imita las acciones registradas ponderadas por ventaja, y ahora esas acciones están concentradas en ítems buenos para cada grupo.
- **IQL máx Q:** **empeora** (de 0,705 a 0,696 visible y de 0,616 a 0,588 oculto). Es el **error de extrapolación** de la sección 5.1 de la guía. El sistema anterior casi nunca mostró a un grupo los ítems que ese grupo no valora, así que esos (grupo, ítem) tienen Q mal estimada. Si el error es optimista, el máximo lo elige. π, que se queda dentro del soporte, no sufre esto. Es exactamente la razón de ser de IQL.

**¿Superan al sistema que generó los datos?** Sí, por mucho: DT 0,726 e IQL π 0,737, frente a 0,593 del sistema anterior con ε = 0.
- El sistema anterior es estocástico: muestrea con softmax a τ = 0,25.
- Los modelos, en cambio, eligen el máximo de sus puntajes.
- Además combinan la información de muchas sesiones, incluidas las de exploración.
- Pedirle al DT un retorno alto o ponderar por ventaja en IQL selecciona lo mejor de los logs, no su promedio.

**Por qué cambian las líneas base.**
- **`popularity`** pasa de −0,06 a 0,69. Con logs uniformes, las apariciones no dicen nada; con los del sistema anterior, "lo que más se mostró" es "lo que el sistema creía bueno". Por la misma razón el protocolo reporta 0,04 en Goodreads con logs originales (no evaluado aquí).
- **`item_mean`** baja de 0,689 a 0,672 por **sesgo de selección**. Cada ítem se mostró sobre todo a los grupos que lo valoran, así que su media observada está inflada para ítems de nicho. El caso más claro: en el grupo 7, `item_mean` cae a 0,48.
- **`group_mean`** no cambia (0,742 frente a 0,741). Al condicionar por grupo se elimina ese sesgo.

## 7. Comparación DT frente a IQL

| | Decision Transformer | IQL |
|---|---|---|
| Qué aprende | qué acciones aparecían cuando faltaba obtener R̂ (supervisado) | cuánto vale cada acción (Q, V) y una política que prefiere las de ventaja positiva |
| Con logs uniformes | señal muy débil: ≈ `item_mean` − 0,03, no usa el grupo | Q bien estimada en todo el catálogo; el máximo de Q es su mejor versión |
| Con logs `mixto` | mejora mucho: imitar ya es recomendar bien, y usa el grupo | π mejora y alcanza a `group_mean`; el máximo de Q empeora por extrapolación |
| Control | el retorno pedido controla el resultado (diagonal 50–80) | τ y β casi no cambian nada en este problema |
| Grupo oculto | no se degrada, pero tampoco adapta | sobrerreacciona a los primeros ratings |

Ninguno de los dos aprovecha la estructura secuencial, porque casi no la hay: el rating no depende del orden, y el único valor secuencial es inferir el grupo, que `group_posterior` resuelve explícitamente. La diferencia de fondo es **qué datos necesitan**:
- El DT depende de que los logs contengan acciones distintas en episodios buenos y malos.
- IQL extrae valor incluso de logs aleatorios, pero su política sólo es tan buena como lo que los datos cubren.

## 8. Límites

- **Cómputo:** se entrenó en CPU, con modelos chicos (d = 64) y pocas épocas (DT 6, IQL 5). La CE del DT seguía bajando levemente en la época 6. Más capacidad y entrenamiento podrían cerrar parte de la distancia con `group_mean`.
- **Barrido de IQL:** se hizo con una sola semilla. Las diferencias entre configuraciones (≤ 0,005) son menores que el ruido, así que la elección es prácticamente indiferente.
- **Validación frente a test:** validación usa un catálogo restringido a los ítems de cada usuario; sus números no se comparan con los de test.
- **Alcance:** return conditioning sólo con los logs originales. Goodreads y el preset `experto` no se evaluaron.

## Reproducción

Ver `README.md`. Comando completo: `bash run_all.sh` (≈ 3 h en CPU). Semillas: partición 0; entrenamiento 0, 1 y 2; evaluación 0, 1 y 2. Las configuraciones están en `outputs/*/config.json` y la selección en `outputs/seleccion/`.
