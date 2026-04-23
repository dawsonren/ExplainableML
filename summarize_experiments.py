"""
summarize_experiments.py

Walks the per-config cache directories produced by run_experiments.py, loads
each results_*.pkl, computes bias² and variance on the fly per (ALE tag, SHAP
tag) pair, and shows an interactive TUI table (textual) with sortable columns
and toggleable column groups.  Also writes a CSV summary.

Usage:
    python summarize_experiments.py [--cache-dir cached_explanations]

TUI keybindings:
    Click column header  Sort by that column (toggle asc/desc)
    b                    Toggle base-info columns
    d                    Toggle per-dimension metric columns
    t                    Toggle tail (time) columns
    q / Escape           Quit
"""

import csv
import json
import os
import argparse
import glob

import joblib
import numpy as np
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header
from textual.reactive import reactive

import models
from experiments import compute_bias_variance


# ---------------------------------------------------------------------------
# True-explanation lookup
# ---------------------------------------------------------------------------

def _true_explanation_fn(signal_name: str):
    """Return the *_explanation function for a signal name, if it exists and is rho-independent."""
    fn = getattr(models, f"{signal_name}_explanation", None)
    if fn is None:
        return None
    # rho-dependent explanations (e.g. signal_multiplicative) take two args; we
    # cannot evaluate them here without knowing rho. Skip them.
    try:
        import inspect
        nparams = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None
    if nparams != 1:
        return None
    return fn


# ---------------------------------------------------------------------------
# Tune JSON loader
# ---------------------------------------------------------------------------

def _load_tune(meta: dict, cache_dir: str) -> dict:
    """Load the tuning JSON for this experiment and return relevant fields."""
    pattern = os.path.join(cache_dir, f"tune_nn_*_snr{meta['snr']:g}*_n{meta['n']}_cv5_niter20.json")
    candidates = glob.glob(pattern)
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

def _rows_from_results(results: dict, config_name: str, subdir: str,
                       results_file: str) -> list[dict]:
    """Expand a results pickle into one row per (ale_tag, shap_tag) pair."""
    meta = results.get("experiment_meta", {})
    explain_grid = results.get("explain_grid")
    ale_store = results.get("ale", {}) or {}
    shap_store = results.get("shap", {}) or {}
    if not ale_store or not shap_store or explain_grid is None:
        return []

    true_fn = _true_explanation_fn(meta.get("signal", ""))
    true_exp = true_fn(explain_grid) if true_fn is not None else None

    tune = _load_tune(meta, subdir)

    # f(x) stddev across replications, averaged over grid points
    f_vals = results.get("f_vals")
    f_stddev = float(f_vals.std(axis=0).mean()) if f_vals is not None else np.nan

    rows = []
    for ale_tag, ale_entry in ale_store.items():
        ale_exps = ale_entry["exps"]
        ale_times = ale_entry.get("times")
        ec = ale_entry.get("config")
        ale_bv = compute_bias_variance(ale_exps, true_exp)

        for shap_tag, shap_entry in shap_store.items():
            shap_exps = shap_entry["exps"]
            shap_times = shap_entry.get("times")
            sc = shap_entry.get("config")
            shap_bv = compute_bias_variance(shap_exps, true_exp)

            d = ale_exps.shape[-1]
            row = {
                "config_name": config_name,
                "signal":      meta.get("signal", ""),
                "snr":         float(meta.get("snr", np.nan)),
                "rho":         float(meta.get("rho", np.nan)),
                "n":           int(meta.get("n", 0)),
                "R":           int(meta.get("replications", 0)),
                "model":       meta.get("model", ""),
                "K":           getattr(ec, "K", None),
                "L":           getattr(ec, "L", None),
                "centering":   getattr(ec, "centering", None),
                "levels_up":   getattr(ec, "levels_up", None),
                "variant":     getattr(ec, "variant", None),
                "n_bootstrap": getattr(ec, "n_bootstrap", 0),
                "local_method": getattr(ec, "local_method", "interpolate"),
                "ale_tag":     ale_tag,
                "shap_tag":    shap_tag,
                "shap_method":        getattr(sc, "method", None),
                "shap_sample_method": getattr(sc, "sample_method", None) or "none",
                "shap_sample_size":   getattr(sc, "sample_size", None),
                "ale_time_mean":   float(ale_times.mean()) if ale_times is not None else np.nan,
                "shap_time_mean":  float(shap_times.mean()) if shap_times is not None else np.nan,
                "dim":             d,
                "f_stddev":        f_stddev,
                "_results_file":   results_file,
                **tune,
            }
            rsr_vals = []
            for i in range(d):
                ss = float(shap_bv["stddev"][i])
                as_ = float(ale_bv["stddev"][i])
                rsr = (ss - as_) / ss if ss > 0 else np.nan
                row[f"ale_bias2_d{i+1}"]          = float(ale_bv["bias2"][i])
                row[f"shap_bias2_d{i+1}"]         = float(shap_bv["bias2"][i])
                row[f"ale_stddev_d{i+1}"]         = as_
                row[f"shap_stddev_d{i+1}"]        = ss
                row[f"rel_stddev_reduction_d{i+1}"] = rsr
                rsr_vals.append(rsr)
            row["rel_stddev_reduction"] = float(np.nanmean(rsr_vals)) if rsr_vals else np.nan
            rows.append(row)
    return rows


def _walk_cache(cache_dir: str) -> list[dict]:
    """Walk cached_explanations/{config_name}/results_*.pkl and return rows."""
    rows = []
    for subdir in sorted(glob.glob(os.path.join(cache_dir, "*"))):
        if not os.path.isdir(subdir):
            continue
        config_name = os.path.basename(subdir)
        for path in sorted(glob.glob(os.path.join(subdir, "results_*.pkl"))):
            try:
                results = joblib.load(path)
            except Exception as e:
                print(f"  [skip] could not load {path}: {e}")
                continue
            rows.extend(_rows_from_results(results, config_name, subdir,
                                           os.path.basename(path)))
    return rows


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

_BASE_COLS = [
    ("Config",       "config_name"),
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
    ("LocalMethod",  "local_method"),
    ("LevelsUp",     "levels_up"),
    ("ALE Tag",      "ale_tag"),
    ("SHAP Tag",     "shap_tag"),
    ("SHAP Method",  "shap_method"),
    ("SHAP Sample",  "shap_sample_method"),
    ("SHAP SampleN", "shap_sample_size"),
    ("CV R²",        "cv_r2"),
    ("Max R²",       "max_r2"),
]

_TAIL_COLS = [
    ("SHAP time/pt", "shap_time_mean"),
    ("ALE time/pt",  "ale_time_mean"),
    ("Std(f)",       "f_stddev"),
    ("StdRed%",      "rel_stddev_reduction"),
]


def _dim_cols(d: int):
    cols = []
    for i in range(1, d + 1):
        cols += [
            (f"ALE Bias² D{i}",  f"ale_bias2_d{i}"),
            (f"SHAP Bias² D{i}", f"shap_bias2_d{i}"),
            (f"ALE Std D{i}",    f"ale_stddev_d{i}"),
            (f"SHAP Std D{i}",   f"shap_stddev_d{i}"),
            (f"StdRed% D{i}",    f"rel_stddev_reduction_d{i}"),
        ]
    return cols


def _fmt_cell(v, key):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if key.startswith(("ale_bias2", "shap_bias2", "ale_stddev", "shap_stddev")):
        return f"{v:.6f}"
    if key.startswith("rel_stddev_reduction"):
        return f"{v:+.1%}"
    if key in ("ale_time_mean", "shap_time_mean"):
        return f"{v:.4f}s"
    if key == "rho":
        return f"{v:.2f}"
    if key == "snr":
        return f"{v:g}"
    if key in ("cv_r2", "max_r2"):
        return f"{v:.4f}"
    if key in ("alpha", "f_stddev"):
        return f"{v:g}"
    return str(v)


def _sort_key(v):
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

    def compose(self) -> ComposeResult:
        from textual.widgets import Static
        yield Header(show_clock=False)
        yield DataTable(cursor_type="row", zebra_stripes=True)
        yield Static(self._status_text(), id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild()

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
            rows = sorted(rows, key=lambda r: _sort_key(r.get(self._sort_col)),
                          reverse=self._sort_reverse)
        for r in rows:
            table.add_row(*[_fmt_cell(r.get(key), key) for _, key in cols])
        try:
            self.query_one("#status").update(self._status_text())
        except Exception:
            pass

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        key = str(event.column_key)
        if self._sort_col == key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = key
            self._sort_reverse = False
        self._rebuild()

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
    csv_keys = ["results_file"] + [key for _, key in cols]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = {k: v for k, v in r.items() if not k.startswith("_")}
            flat["results_file"] = r.get("_results_file", "")
            for key in flat:
                if key.startswith(("ale_bias2", "shap_bias2", "ale_stddev", "shap_stddev")):
                    if flat[key] is not None:
                        flat[key] = f"{flat[key]:.6f}"
                elif key.startswith("rel_stddev_reduction") and key != "rel_stddev_reduction":
                    if flat[key] is not None:
                        flat[key] = f"{flat[key]:+.4f}"
            for key in ("cv_r2", "max_r2"):
                if key in flat and flat[key] is not None:
                    flat[key] = f"{flat[key]:.4f}"
            if "rel_stddev_reduction" in flat and flat["rel_stddev_reduction"] is not None:
                flat["rel_stddev_reduction"] = f"{flat['rel_stddev_reduction']:+.4f}"
            for key in ("ale_time_mean", "shap_time_mean"):
                if key in flat and flat[key] is not None:
                    flat[key] = f"{flat[key]:.6f}"
            writer.writerow(flat)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Summarize experiment bias², variance, and time results."
    )
    parser.add_argument("--cache-dir", default="cached_explanations",
                        help="Root directory containing per-config sub-folders.")
    parser.add_argument("--csv-out", default="summary.csv",
                        help="Path for the output CSV file (default: summary.csv).")
    args = parser.parse_args()

    rows = _walk_cache(args.cache_dir)
    if not rows:
        print(f"No results_*.pkl files found under {args.cache_dir!r}")
        return

    rows.sort(
        key=lambda r: r["rel_stddev_reduction"]
        if not np.isnan(r["rel_stddev_reduction"])
        else -np.inf,
        reverse=True,
    )

    _write_csv(rows, args.csv_out)
    print(f"CSV written to {args.csv_out}")

    max_dim = max((r.get("dim", 1) for r in rows), default=1)
    SummaryApp(rows, max_dim).run()


if __name__ == "__main__":
    main()
