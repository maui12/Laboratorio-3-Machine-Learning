import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any

def generate_all_plots(results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Genera todas las visualizaciones obligatorias del laboratorio."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_class_distributions(results_by_target, output_dir)
    _plot_metric_comparison(results_by_target, "f1_macro_mean", "F1 Macro", "comparacion_f1_macro.png", output_dir)
    _plot_metric_comparison(results_by_target, "balanced_accuracy_mean", "Balanced Accuracy", "comparacion_balanced_accuracy.png", output_dir)
    _plot_metric_comparison(results_by_target, "icn", "ICN", "comparacion_icn.png", output_dir)
    _plot_best_confusion_matrices(results_by_target, output_dir)
    _plot_normalized_metrics_heatmap(results_by_target, output_dir)


def _plot_class_distributions(results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Distribución de clases para cada columna objetivo."""
    for target, results in results_by_target.items():
        dist = results[0]["class_distribution"]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=list(dist.keys()), y=list(dist.values()), hue=list(dist.keys()), palette="viridis", legend=False)
        plt.xlabel("Clase")
        plt.ylabel("Frecuencia")
        plt.title(f"Distribución de clases - {target}")
        plt.tight_layout()
        plt.savefig(output_dir / f"distribucion_clases_{target}.png")
        plt.close()


def _plot_metric_comparison(results_by_target: dict[str, list[dict[str, Any]]], metric_key: str, ylabel: str, filename: str, output_dir: Path) -> None:
    """Comparación de F1 macro, Balanced accuracy e ICN entre modelos."""
    data = []
    for target, results in results_by_target.items():
        for res in results:
            if res["implemented"]:
                data.append({"Target": target, "Modelo": res["model_name"], ylabel: res[metric_key]})
    
    if not data:
        return

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Target", y=ylabel, hue="Modelo")
    plt.title(f"Comparación de {ylabel} entre modelos")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def _plot_best_confusion_matrices(results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Matriz de confusión del mejor modelo por objetivo, basado en el ICN."""
    for target, results in results_by_target.items():
        implemented = [r for r in results if r["implemented"]]
        if not implemented:
            continue

        # Seleccionar el mejor modelo según ICN
        best_model = max(implemented, key=lambda x: x.get("icn") or 0.0)
        cm = np.array(best_model["confusion_matrix"])
        labels = best_model["labels"]

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.title(f"Matriz de Confusión - Mejor Modelo ({best_model['model_name']})\nObjetivo: {target}")
        plt.tight_layout()
        plt.savefig(output_dir / f"matriz_confusion_mejor_{target}.png")
        plt.close()


def _plot_normalized_metrics_heatmap(results_by_target: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    """Tabla o mapa de calor con todas las métricas normalizadas."""
    metrics = ["f1_macro_mean", "balanced_accuracy_mean", "recall_macro_mean", "precision_macro_mean", "stability", "icn"]
    display_names = ["F1", "BA", "Recall", "Precision", "Estab.", "ICN"]

    for target, results in results_by_target.items():
        implemented = [r for r in results if r["implemented"]]
        if not implemented:
            continue

        df = pd.DataFrame(implemented)
        df.set_index("model_name", inplace=True)

        norm_df = pd.DataFrame(index=df.index)
        for col, d_name in zip(metrics, display_names):
            if col in ["stability", "icn"]:
                # Estas ya vienen normalizadas entre 0 y 1 desde evaluation.py
                norm_df[d_name] = df[col]
            else:
                # Normalización Min-Max para el resto
                c_min, c_max = df[col].min(), df[col].max()
                denom = c_max - c_min + 1e-12
                norm_df[d_name] = (df[col] - c_min) / denom

        plt.figure(figsize=(8, len(implemented) * 1.2 + 1))
        sns.heatmap(norm_df, annot=True, cmap="coolwarm", vmin=0, vmax=1, fmt=".2f")
        plt.title(f"Mapa de calor de métricas normalizadas - {target}")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_normalizado_{target}.png")
        plt.close()