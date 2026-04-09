"""
summarize_experiments.py

Loads all full_*.npz cache files, extracts bias², variance, and speedup metrics,
and shows an interactive TUI table (textual) with sortable columns and toggleable
column groups.  Also writes a CSV summary.

Usage:
    python summarize_experiments.py [--cache-dir cached_explanations]

TUI keybindings:
    Click column header  Sort by that column (toggle asc/desc)
    b                    Toggle base-info columns
    d                    Toggle per-dimension metric columns
    t                    Toggle tail (Speedup / VarRed%) columns
    q / Escape           Quit
"""

import csv
import json
import os
import argparse
import glob

import numpy as np
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header
from textual.reactive import reactive


# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------

def _load_meta(full_path: str) -> dict | None:
    """Read meta_* scalar fields from a full_*.npz and return a metadata dict.

    Returns None if the file does not contain metadata (old-format file).
    """
    try:
        data = np.load(full_path, allow_pickle=True)
    except Exception:
        return None

    required = {"meta_signal", "meta_snr", "meta_rho", "meta_n", "meta_R",
                "meta_K", "meta_L", "meta_centering", "meta_levels_up",
                "meta_variant", "meta_tag"}
    if not required.issubset(set(data.files)):
        return None

    return {
        "signal":     str(data["meta_signal"]),
        "snr":        float(data["meta_snr"]),
        "rho":        float(data["meta_rho"]),
        "n":          int(data["meta_n"]),
        "R":          int(data["meta_R"]),
        "model":      str(data["meta_model"]) if "meta_model" in data.files else "unknown",
        "K":          int(data["meta_K"]),
        "L":          int(data["meta_L"]),
        "centering":  str(data["meta_centering"]),
        "levels_up":  int(data["meta_levels_up"]),
        "variant":    str(data["meta_variant"]),
        "n_bootstrap": int(data["meta_n_bootstrap"]) if "meta_n_bootstrap" in data.files else 0,
        "tag":        str(data["meta_tag"]),
    }


# ---------------------------------------------------------------------------
# Tune JSON loader
# ---------------------------------------------------------------------------

def _load_tune(meta: dict, cache_dir: str) -> dict:
    """Load the tuning JSON for this experiment and return relevant fields."""
    model_name = meta.get("model", "")
    # derive DGP slug from meta fields (best effort; file may not exist)
    dgp_slug = f"{meta['signal']}_snr{meta['snr']:g}"  # simplified; actual slug in filename
    # Try to find by scanning for matching tune files
    pattern = os.path.join(cache_dir, f"tune_nn_*_snr{meta['snr']:g}*_n{meta['n']}_cv5_niter20.json")
    candidates = glob.glob(pattern)
    # pick the one whose signal name appears in the filename
    signal_name = meta["signal"]
    match = None
    for c in candidates:
        if signal_name in c:
            match = c
            break
    if match is None and candidates:
        match = candidates[0]
    if match is None:
        return {}
    with open(match) as f:
        t = json.load(f)
    layers = t.get("hidden_layer_sizes", [])
    return {
        "layers":   "x".join(str(s) for s in layers) if layers else "N/A",
        "act":      t.get("activation", "N/A"),
        "solver":   t.get("solver", "N/A"),
        "alpha":    t.get("alpha"),
        "cv_r2":    t.get("cv_r2"),
        "max_r2":   t.get("max_r2"),
    }


# ---------------------------------------------------------------------------
# Row loader
# ---------------------------------------------------------------------------

def _load_row(full_path: str, cache_dir: str):
    """Load one full_*.npz file and its companion ale_*.npz; return a result dict."""
    meta = _load_meta(full_path)
    if meta is None:
        print(f"  [skip] no metadata in: {os.path.basename(full_path)}")
        return None

    full_cache = np.load(full_path, allow_pickle=True)

    shap_times    = full_cache["shap_times"]          # (R,) sec/point
    shap_variance = np.atleast_1d(full_cache["shap_variance"])  # (d,)
    ale_variance  = np.atleast_1d(full_cache["ale_variance"])   # (d,)
    shap_bias2    = np.atleast_1d(full_cache["shap_bias2"])     # (d,)
    ale_bias2     = np.atleast_1d(full_cache["ale_bias2"])      # (d,)
    d = len(shap_variance)

    ale_fname = os.path.basename(full_path).replace("full_", "ale_", 1)
    ale_path  = os.path.join(cache_dir, ale_fname)

    ale_times = None
    if os.path.exists(ale_path):
        ale_cache = np.load(ale_path, allow_pickle=True)
        if "ale_times" in ale_cache.files:
            ale_times = ale_cache["ale_times"]  # (R,) sec/point

    speedup = (
        float(shap_times.mean()) / float(ale_times.mean())
        if ale_times is not None and ale_times.mean() > 0
        else np.nan
    )

    tune = _load_tune(meta, cache_dir)

    row = {
        **meta,
        **tune,
        "speedup":     speedup,
        "_cache_file": os.path.basename(full_path),
    }
    rvr_vals = []
    for i in range(d):
        sv = float(shap_variance[i])
        av = float(ale_variance[i])
        rvr = (sv - av) / sv if sv > 0 else np.nan
        row[f"ale_bias2_d{i+1}"]         = float(ale_bias2[i])
        row[f"shap_bias2_d{i+1}"]        = float(shap_bias2[i])
        row[f"ale_variance_d{i+1}"]      = av
        row[f"shap_variance_d{i+1}"]     = sv
        row[f"rel_var_reduction_d{i+1}"] = rvr
        rvr_vals.append(rvr)
    row["rel_var_reduction"] = float(np.nanmean(rvr_vals))
    row["dim"] = d
    return row


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_BASE_COLS = [
    # (header,       key)
    ("Signal",       "signal"),
    ("SNR",          "snr"),
    ("Rho",          "rho"),
    ("Model",        "model"),
    ("Layers",       "layers"),
    ("Act",          "act"),
    ("Solver",       "solver"),
    ("Alpha",        "alpha"),
    ("K",            "K"),
    ("L",            "L"),
    ("Variant",      "variant"),
    ("LevelsUp",     "levels_up"),
    ("CV R²",        "cv_r2"),
    ("Max R²",       "max_r2"),
]

_TAIL_COLS = [
    ("Speedup",      "speedup"),
    ("VarRed%",      "rel_var_reduction"),
]


def _dim_cols(d: int):
    cols = []
    for i in range(1, d + 1):
        cols += [
            (f"ALE Bias² D{i}",  f"ale_bias2_d{i}"),
            (f"SHAP Bias² D{i}", f"shap_bias2_d{i}"),
            (f"ALE Var D{i}",    f"ale_variance_d{i}"),
            (f"SHAP Var D{i}",   f"shap_variance_d{i}"),
            (f"VarRed% D{i}",    f"rel_var_reduction_d{i}"),
        ]
    return cols


def _fmt_cell(v, key):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if key.startswith(("ale_bias2", "shap_bias2", "ale_variance", "shap_variance")):
        return f"{v:.6f}"
    if key.startswith("rel_var_reduction"):
        return f"{v:+.1%}"
    if key == "speedup":
        return f"{v:.2f}x"
    if key == "rho":
        return f"{v:.2f}"
    if key == "snr":
        return f"{v:g}"
    if key in ("cv_r2", "max_r2"):
        return f"{v:.4f}"
    if key == "alpha":
        return f"{v:g}"
    return str(v)


def _sort_key(v):
    """Numeric sort key: NaN/None sorts last, strings sort lexicographically."""
    if v is None:
        return (1, 0, "")
    if isinstance(v, float) and np.isnan(v):
        return (1, 0, "")
    if isinstance(v, (int, float)):
        return (0, v, "")
    return (0, 0, str(v))


# ---------------------------------------------------------------------------
# Textual TUI
# ---------------------------------------------------------------------------

class SummaryApp(App):
    CSS = """
    DataTable { height: 1fr; }
    #status { height: 1; background: $panel; color: $text-muted; padding: 0 1; }
    """

    BINDINGS = [
        Binding("q,escape", "quit", "Quit"),
        Binding("b", "toggle_base", "Base cols"),
        Binding("d", "toggle_dim",  "Per-dim cols"),
        Binding("t", "toggle_tail", "Tail cols"),
    ]

    show_base: reactive[bool] = reactive(True)
    show_dim:  reactive[bool] = reactive(True)
    show_tail: reactive[bool] = reactive(True)

    def __init__(self, rows: list, max_dim: int):
        super().__init__()
        self._rows = rows
        self._max_dim = max_dim
        self._sort_col: str | None = None
        self._sort_reverse = False

    # ---- layout ------------------------------------------------------------

    def compose(self) -> ComposeResult:
        from textual.widgets import Static
        yield Header(show_clock=False)
        yield DataTable(cursor_type="row", zebra_stripes=True)
        yield Static(self._status_text(), id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild()

    # ---- helpers -----------------------------------------------------------

    def _visible_cols(self):
        cols = []
        if self.show_base:
            cols += _BASE_COLS
        if self.show_dim:
            cols += _dim_cols(self._max_dim)
        if self.show_tail:
            cols += _TAIL_COLS
        return cols

    def _status_text(self) -> str:
        groups = []
        if self.show_base:  groups.append("Base[b]")
        if self.show_dim:   groups.append("Per-dim[d]")
        if self.show_tail:  groups.append("Tail[t]")
        shown = " | ".join(groups) if groups else "(none)"
        sort_info = f"  •  sorted by {self._sort_col} {'↓' if self._sort_reverse else '↑'}" if self._sort_col else ""
        return f" Showing: {shown}{sort_info}  •  {len(self._rows)} rows"

    def _rebuild(self) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)

        cols = self._visible_cols()
        for header, key in cols:
            table.add_column(header, key=key)

        rows = self._rows
        if self._sort_col is not None:
            rows = sorted(
                rows,
                key=lambda r: _sort_key(r.get(self._sort_col)),
                reverse=self._sort_reverse,
            )

        for r in rows:
            table.add_row(*[_fmt_cell(r.get(key), key) for _, key in cols])

        # update status bar
        try:
            self.query_one("#status").update(self._status_text())
        except Exception:
            pass

    # ---- event handlers ----------------------------------------------------

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        key = str(event.column_key)
        if self._sort_col == key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = key
            self._sort_reverse = False
        self._rebuild()

    # ---- actions -----------------------------------------------------------

    def action_toggle_base(self) -> None:
        self.show_base = not self.show_base
        self._rebuild()

    def action_toggle_dim(self) -> None:
        self.show_dim = not self.show_dim
        self._rebuild()

    def action_toggle_tail(self) -> None:
        self.show_tail = not self.show_tail
        self._rebuild()


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(rows: list, path: str) -> None:
    d = max((r.get("dim", 1) for r in rows), default=1)
    cols = _BASE_COLS + _dim_cols(d) + _TAIL_COLS
    csv_keys = ["cache_file"] + [key for _, key in cols]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = {k: v for k, v in r.items() if not k.startswith("_")}
            flat["cache_file"] = r.get("_cache_file", "")
            for key in flat:
                if key.startswith(("ale_bias2", "shap_bias2", "ale_variance", "shap_variance")):
                    if flat[key] is not None:
                        flat[key] = f"{flat[key]:.6f}"
                elif key.startswith("rel_var_reduction") and key != "rel_var_reduction":
                    if flat[key] is not None:
                        flat[key] = f"{flat[key]:+.4f}"
            for key in ("cv_r2", "max_r2"):
                if key in flat and flat[key] is not None:
                    flat[key] = f"{flat[key]:.4f}"
            if "rel_var_reduction" in flat and flat["rel_var_reduction"] is not None:
                flat["rel_var_reduction"] = f"{flat['rel_var_reduction']:+.4f}"
            if "speedup" in flat and flat["speedup"] is not None:
                flat["speedup"] = f"{flat['speedup']:.4f}"
            writer.writerow(flat)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiment bias², variance, and speedup results."
    )
    parser.add_argument("--cache-dir", default="cached_explanations",
                        help="Directory containing full_*.npz and ale_*.npz files.")
    parser.add_argument("--csv-out", default="summary.csv",
                        help="Path for the output CSV file (default: summary.csv).")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "full_*.npz")))
    if not paths:
        print(f"No full_*.npz files found in {args.cache_dir!r}")
        return

    rows = []
    for path in paths:
        row = _load_row(path, args.cache_dir)
        if row is not None:
            rows.append(row)

    # Default sort: descending relative variance reduction
    rows.sort(
        key=lambda r: r["rel_var_reduction"]
        if not np.isnan(r["rel_var_reduction"])
        else -np.inf,
        reverse=True,
    )

    _write_csv(rows, args.csv_out)
    print(f"CSV written to {args.csv_out}")

    max_dim = max((r.get("dim", 1) for r in rows), default=1)
    SummaryApp(rows, max_dim).run()


if __name__ == "__main__":
    main()
