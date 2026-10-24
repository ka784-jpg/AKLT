"""Plot a checked single-trajectory CSV; never substitutes paper data."""

import argparse
import csv
import math
from pathlib import Path
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_trace(directory, model):
    directory = Path(directory)
    metadata = tomllib.loads((directory / "metadata.toml").read_text(encoding="utf-8"))
    parameters = metadata["parameters"]
    if metadata.get("model", "nnn") != model:
        raise ValueError("Model does not match the saved metadata")
    required = ["sweep", "abs_mean_nn"]
    required += ["abs_mean_nnn"] if model == "nnn" else ["entropy_bits"]
    if model == "dimer":
        required += ["abs_dimer_order"]
    with (directory / "summary.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("Missing required CSV columns")
        rows = [{key: float(row[key]) for key in required} for row in reader]
    if not rows or not all(math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("Empty or non-finite trace")
    sweeps = [row["sweep"] for row in rows]
    if sweeps != sorted(set(sweeps)) or any(t <= 0 or not t.is_integer() for t in sweeps):
        raise ValueError("Sweep labels must be positive, distinct, increasing integers")
    return rows, parameters


def plot_trace(directory, model, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError("Choose a .png output")
    rows, config = load_trace(directory, model)
    x = [row["sweep"] for row in rows]
    titles = {"trivial": "NN AKLT / zero-trivial", "dimer": "NN AKLT / dimer", "nnn": "NN / NNN AKLT"}
    p = config["p_nnn"] if model == "nnn" else config["p_competing"]
    symbol = {"trivial": r"p_{\mathrm{TT}}", "dimer": r"p_{\mathrm{D}}", "nnn": r"p_{\mathrm{nnn}}"}[model]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
        fig.suptitle(f"{titles[model]}: single-trajectory demonstration\n"
                     f"L = {config['L']}, ${symbol} = {p:.2f}$, master seed = {config['seed']}", fontsize=13)
        axes[0].plot(x, [row["abs_mean_nn"] for row in rows], "o-", color="#246B8E", markersize=3.5,
                     label=r"$|\langle O_{\mathrm{nn}}^{\mathrm{str}}\rangle_{\mathrm{origins}}|$")
        axes[0].set_ylabel("String-order magnitude")
        if model == "dimer":
            axes[0].plot(x, [row["abs_dimer_order"] for row in rows], "s-", color="#C15F17", markersize=3.5,
                         label=r"$|O_{\mathrm{dimer}}|$")
            axes[0].set_ylabel("Order-parameter magnitude")
        if model == "nnn":
            axes[1].plot(x, [row["abs_mean_nnn"] for row in rows], "s-", color="#21835B", markersize=3.5,
                         label=r"$|\langle O_{\mathrm{nnn}}^{\mathrm{str}}\rangle_{\mathrm{origins}}|$")
            axes[1].set_ylabel("String-order magnitude")
        else:
            axes[1].plot(x, [row["entropy_bits"] for row in rows], "s-", color="#21835B", markersize=3.5,
                         label=r"$S_1(L/2)$")
            axes[1].set_ylabel("Half-chain entropy (bits)")
        for ax in axes:
            ax.set_xlabel("Sweeps (L local update slots each)")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.18)
            ax.legend(loc="best", fontsize=10)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("xb") as stream:
                fig.savefig(stream, format="png", dpi=170, metadata={"Software": "AKLT example plotting script"})
        finally:
            plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["trivial", "dimer", "nnn"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(plot_trace(args.directory, args.model, args.output))
