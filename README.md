# vero-rl — TP 2026: Decision Transformer e IQL para recomendaciones

Trabajo práctico de Aprendizaje por Refuerzos (DiploDatos 2026, FAMAF). Dataset: **Netflix**.
Reporte: [`docs/REPORTE.md`](docs/REPORTE.md) · Bitácora: [`docs/BITACORA.md`](docs/BITACORA.md)

## Estructura

| Ruta | Contenido |
|---|---|
| `config_dataset.py` | Dataset y ubicación de los datos |
| `protocolo/` | Protocolo oficial de evaluación (copia sin modificar del repo de la clase) |
| `src/dt.py` | Decision Transformer: modelo, pérdida, entrenamiento y política |
| `src/iql.py` | IQL: V (expectil), Q (TD), π (AWR), red objetivo y políticas π / máximo de Q |
| `src/utils.py` | Episodios, semillas y evaluación por lotes (idéntica a la oficial, verificada en los tests) |
| `src/experimentos.py` | Entrenamiento, selección en validación y evaluación en test |
| `src/reporte.py` | Tablas a partir de `outputs/` |
| `tests/` | No fuga del DT, máscaras, expectil, gradientes de IQL y equivalencia de la evaluación |
| `notebooks/01…06` | Una parte del TP por notebook, ya ejecutados |
| `outputs/` | Checkpoints, curvas, configuraciones, selección y resultados |
| `run_all.sh` | Reproduce todos los resultados |

## Datos

No se incluyen en el repo. Por defecto se leen de `../AprendizajePorRefuerzos/2026/tp/data`, es decir, del repo de la clase clonado en la misma carpeta que este. Para otra ubicación, definir la variable de entorno `VERO_RL_DATA`.

## Instalación (Windows, Anaconda Prompt)

```
cd /d E:\trabajo-final-famaf\vero-rl
conda create -n vero-rl python=3.12 -y
conda activate vero-rl
pip install -r requirements.txt
```

## Verificación rápida (≈ 2 min)

```
python -m pytest protocolo/tests tests
```

Deben pasar los 25 tests: 19 del protocolo y 6 propios.

## Ver los resultados

```
jupyter notebook notebooks
```

Los notebooks ya están ejecutados. Se pueden abrir sin correr nada, o volver a ejecutar con *Kernel → Restart & Run All*: leen los resultados de `outputs/`.

## Reentrenar todo desde cero (≈ 3 h en CPU)

```
bash run_all.sh
```

En Windows sin bash, ejecutar en orden los comandos de `run_all.sh`.

Semillas: partición 0; entrenamiento 0, 1 y 2; evaluación 0, 1 y 2.
