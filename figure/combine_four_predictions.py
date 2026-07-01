import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20


def _load_prediction_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required_cols = {"engine_id", "true_rul", "pred_rul"}
    missing = required_cols.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    return frame


def _plot_single_panel(ax, frame: pd.DataFrame, panel_label: str) -> None:
    engine_idx = frame["engine_id"].to_numpy(dtype=np.float32)
    true_rul = frame["true_rul"].to_numpy(dtype=np.float32)
    pred_rul = frame["pred_rul"].to_numpy(dtype=np.float32)
    error = pred_rul - true_rul

    ax.bar(engine_idx, error, color="#5fd6e8", width=0.8, alpha=0.9)
    ax.plot(
        engine_idx,
        true_rul,
        linestyle="--",
        color="#dc8a8a",
        marker="s",
        markerfacecolor="none",
        markersize=6,
        linewidth=1.0,
    )
    ax.plot(
        engine_idx,
        pred_rul,
        linestyle="--",
        color="#89d68c",
        marker="^",
        markerfacecolor="none",
        markersize=6,
        linewidth=1.0,
    )

    ax.axhline(0.0, color="#8c8c8c", linewidth=0.8)
    # Per-panel axis labels are removed in favor of shared figure-level labels.
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y", alpha=0.25)

    y_min = min(float(np.min(error)), float(np.min(true_rul)), float(np.min(pred_rul)))
    y_max = max(float(np.max(error)), float(np.max(true_rul)), float(np.max(pred_rul)))
    ax.set_ylim(min(-30.0, y_min - 5.0), y_max + 8.0)
    ax.set_xlim(0.5, len(engine_idx) + 0.5)

    ax.set_title(panel_label, fontsize=PANEL_LABEL_SIZE, pad=8)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    figure_dir = project_root / "figure"

    parser = argparse.ArgumentParser(
        description="Combine FD001-FD004 prediction plots into one SVG figure."
    )
    parser.add_argument("--fd001-csv", type=str, required=True)
    parser.add_argument("--fd002-csv", type=str, required=True)
    parser.add_argument("--fd003-csv", type=str, required=True)
    parser.add_argument("--fd004-csv", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=str(figure_dir / "FD001_FD004_combined.svg"),
        help="Output combined figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_paths = [
        Path(args.fd001_csv),
        Path(args.fd002_csv),
        Path(args.fd003_csv),
        Path(args.fd004_csv),
    ]
    panel_labels = [
        "(a) Result on FD001",
        "(b) Result on FD002",
        "(c) Result on FD003",
        "(d) Result on FD004",
    ]

    frames = [_load_prediction_csv(path) for path in csv_paths]

    # Create a 2x2 grid: TL FD001, TR FD002, BL FD003, BR FD004
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25, 10))
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for idx, (ax, frame, label) in enumerate(zip(axes_list, frames, panel_labels)):
        _plot_single_panel(ax, frame, label)
        # Restore horizontal reference line at y=-25 for panel 1 (FD001) and panel 4 (FD004)
        ax.axhline(-25.0, color="#8c8c8c", linewidth=0.8, linestyle="--")
            # Set major ticks every 25 units for these panels
        ax.yaxis.set_major_locator(MultipleLocator(25))

    # Hide redundant tick labels: show x tick labels only on bottom row,
    # and show y tick labels only on left column.
    for idx, ax in enumerate(axes_list):
        is_top_row = idx in (0, 1)
        is_left_col = idx in (0, 2)
        ax.tick_params(labelbottom=not is_top_row, labelleft=is_left_col)

    # Add shared axis labels for the whole figure
    fig.text(0.5, 0.02, "Engine Id", ha="center", fontsize=AXIS_LABEL_SIZE)
    fig.text(0.02, 0.5, "RUL(Cycle)", va="center", rotation="vertical", fontsize=AXIS_LABEL_SIZE)

    # Minimal padding to avoid whitespace
    fig.tight_layout(pad=1.0, h_pad=1.0, w_pad=0.4, rect=(0.02, 0.03, 0.98, 0.995))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)

    print(f"Saved combined figure to: {output_path}")


if __name__ == "__main__":
    main()
