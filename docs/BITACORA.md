# Bitácora de decisiones y contribuciones

## Decisiones verificadas

| # | Decisión | Verificación |
|---|---|---|
| 1 | Dataset Netflix | — |
| 2 | Protocolo oficial copiado sin modificar | 19 tests del protocolo pasan; las líneas base reproducen la tabla de `PROTOCOLO.md` (`item_mean` 0,689; `group_mean` 0,742/0,689; `group_posterior` 0,742/0,707) |
| 3 | Datos leídos del repo de la clase, sin duplicarlos | `config_dataset.py` + variable `VERO_RL_DATA` |
| 4 | Particiones por usuario (`split_by_user`, semilla 0) | 0 usuarios compartidos entre train y validación (notebook 01) |
| 5 | R̂ calculado dentro de cada episodio de 20 pasos | asserts `R̂_{t+1} = R̂_t − r_t` y `R̂_19 = r_19` (notebook 01) |
| 6 | DT: estado = grupo + rating anterior; a_t se predice desde s_t | test de no fuga: diferencia 0,0 al cambiar a_≥t, R̂_>t y s_>t |
| 7 | DT: inicialización N(0; 0,02) | con N(0, 1) la CE inicial era 8,1 (> uniforme 6,62); corregido: 6,62 |
| 8 | 30 % de episodios con grupo "desconocido" (DT e IQL) | sin esto el token "desconocido" nunca se ve al entrenar |
| 9 | IQL: tres codificadores separados; recompensa (r − 3)/2 | test: cada pérdida sólo actualiza su red; con γ = 0 el objetivo de Q es la recompensa |
| 10 | IQL: último paso terminal | `not_done = 0` en t = 19 (test) |
| 11 | Expectil | test con el ejemplo de la guía: τ = 0,5 → 4,0 y τ = 0,9 → 8,0 |
| 12 | Evaluación por lotes (velocidad) | test: resultados idénticos a `protocolo.evaluar.evaluate` |
| 13 | Retorno pedido al DT elegido en validación: 107 (random) y 99 (mixto) | barrido p10…máx + 10, 3 semillas (`outputs/seleccion/`) |
| 14 | IQL: γ = 0,9, τ = 0,7, β = 10 elegidos en validación | barrido de 8 configuraciones; diferencias ≤ 0,005 (indiferente) |
| 15 | Test usado una sola vez | `test_eval` y `rc_test` se corrieron después de fijar la selección |
| 16 | Logs `mixto`: mismos hiperparámetros, salvo el retorno pedido al DT (reelegido en validación) | consigna §5 |
| 17 | Modelos chicos (d = 64) por entrenar en CPU | limitación declarada en el reporte |

## Contribuciones

| Integrante | Aporte |
|---|---|
| _completar_ | |

## Uso de IA

Se utilizó Claude (Anthropic) como asistente para escribir el código, ejecutar los experimentos y redactar los textos. Se verificó:
- con tests automáticos: no fuga del DT, máscaras, expectil, separación de gradientes y equivalencia de la evaluación;
- que las líneas base coinciden con la tabla oficial del protocolo;
- que la selección de hiperparámetros se hizo sólo con validación.
