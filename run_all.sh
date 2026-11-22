#!/usr/bin/env bash
# Reproduce todos los resultados del TP (CPU, ~3 h). Ejecutar desde la raíz del repo.
# En Windows: correr cada comando en el mismo orden desde el Anaconda Prompt.
set -euo pipefail
E="python -m src.experimentos"
sel() { python -c "import json;d=json.load(open('outputs/seleccion/$1'));print($2)"; }

# 0. Líneas base (protocolo oficial)
python -m protocolo.evaluar --dataset netflix --output outputs/baselines/random_test
python -m protocolo.evaluar --dataset netflix --logs mixto --output outputs/baselines/mixto_test

# 1. Logs originales (random)
$E dt_train --logs random
$E dt_select --logs random
$E iql_sweep --logs random                       # 8 configuraciones, semilla 0, elige en val
for g in 0.0 0.9; do
  read t b < <(sel iql_random_elegido.json "' '.join(map(str, d['mejor_tau_beta_por_gamma']['$g']))")
  $E iql_train --logs random --gamma $g --tau $t --beta $b
done
$E test_eval --logs random
$E rc_test --logs random

# 2. Logs del sistema anterior (mixto): mismos hiperparámetros
$E dt_train --logs mixto
$E dt_select --logs mixto
read g t b < <(sel iql_random_elegido.json "d['gamma'], d['tau'], d['beta']" | tr -d '(),')
$E iql_train --logs mixto --gamma $g --tau $t --beta $b
$E test_eval --logs mixto
