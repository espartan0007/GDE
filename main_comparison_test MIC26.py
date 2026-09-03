import numpy as np
import matplotlib.pyplot as plt
import time

from gopt_algorithms import de_run

# ============================================
# 1. Función objetivo: Griewank
# ============================================
def griewank(x):

    dim = len(x)
    sum_sq = np.sum(x**2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1))))

    return 1.0 + sum_sq - prod_cos

# ============================================
# 2. Configuración del experimento
# ============================================
DIM = 50
BOUNDS = np.array([[-100, 100]] * DIM)

POP_SIZE = 50
GENS = 2000          # Total de generaciones
N_RUNS = 5           # Número de ejecuciones independientes
SEED_BASE = 42

# ============================================
# 3. Schedule  GDE
# ============================================

base_schedule = [("original", 120), ("smooth", 80)]
schedule = []
acc = 0
while acc < GENS:
    for phase, niters in base_schedule:
        if acc >= GENS:
            break
        take = min(niters, GENS - acc)
        schedule.append((phase, take))
        acc += take

print(f"Dimensiones: {DIM}")
print(f"Población: {POP_SIZE}")
print(f"Generaciones: {GENS}")
print(f"Runs independientes: {N_RUNS}")
print(f"Schedule GDE tiene {len(schedule)} fases (total {sum(n for _, n in schedule)} gen)\n")

# ============================================
# 4. Ejecución de los algoritmos
# ============================================
results_de = []
results_gde = []

for r in range(N_RUNS):
    seed = SEED_BASE + r
    print(f"Run {r+1}/{N_RUNS}...")

    # DE Original
    t0 = time.time()
    hist_de = de_run(
        obj_fun=griewank,
        dim=DIM,
        bounds=BOUNDS,
        pop_size=POP_SIZE,
        gens=GENS,
        mode="original",
        seed=seed
    )
    t_de = time.time() - t0
    results_de.append(hist_de)

    # GDE (DE_Hybrid_r2)
    t0 = time.time()
    hist_gde = de_run(
        obj_fun=griewank,
        dim=DIM,
        bounds=BOUNDS,
        pop_size=POP_SIZE,
        gens=GENS,
        mode="hybrid",
        schedule=schedule,
        seed=seed
    )
    t_gde = time.time() - t0
    results_gde.append(hist_gde)

    print(f"  DE   best final: {hist_de[-1]:.6e} | tiempo: {t_de:.2f}s")
    print(f"  GDE  best final: {hist_gde[-1]:.6e} | tiempo: {t_gde:.2f}s")

# Convertir a arrays para estadísticas
results_de = np.array(results_de)    # shape: (N_RUNS, GENS)
results_gde = np.array(results_gde)

# ============================================
# 5. Resultados estadísticos
# ============================================
print("\n" + "="*50)
print("RESULTADOS FINALES (última generación)")
print("="*50)

de_final = results_de[:, -1]
gde_final = results_gde[:, -1]

print(f"{'Metric':<20} {'DE Original':>15} {'GDE':>15}")
print("-" * 50)
print(f"{'Best':<20} {np.min(de_final):>15.6e} {np.min(gde_final):>15.6e}")
print(f"{'Mean':<20} {np.mean(de_final):>15.6e} {np.mean(gde_final):>15.6e}")
print(f"{'Std':<20} {np.std(de_final):>15.6e} {np.std(gde_final):>15.6e}")

# ============================================
# 6. Gráfica de convergencia
# ============================================
plt.figure(figsize=(10, 6))

# Media de las runs
mean_de = np.mean(results_de, axis=0)
mean_gde = np.mean(results_gde, axis=0)

# Mejor de las runs
best_de = np.min(results_de, axis=0)
best_gde = np.min(results_gde, axis=0)

gens_axis = np.arange(1, GENS + 1)

plt.semilogy(gens_axis, mean_de, 'b-', linewidth=2, label='DE Original (mean)')
plt.semilogy(gens_axis, mean_gde, 'r-', linewidth=2, label='GDE (mean)')
plt.fill_between(gens_axis, best_de, mean_de, color='blue', alpha=0.1)
plt.fill_between(gens_axis, best_gde, mean_gde, color='red', alpha=0.1)

plt.xlabel("Generación", fontsize=12)
plt.ylabel("Mejor fitness (log scale)", fontsize=12)
plt.title(f"Convergencia: DE vs GDE | Griewank {DIM}D | {N_RUNS} runs", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("convergence_de_vs_gde.png", dpi=150)
plt.show()

print("\nGráfica guardada como: convergence_de_vs_gde.png")
