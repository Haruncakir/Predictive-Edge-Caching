"""Predictive Edge Caching — Tkinter GUI Dashboard.

Run:  cd ml/ && uv run python gui_app.py
"""
from __future__ import annotations
import logging, os, queue, sys, threading

# Fix Tcl/Tk path when running under uv's venv on Windows.
# sys.base_prefix always points to the real Python install, even inside a venv,
# so this works regardless of where Python is installed on the machine.
if sys.platform == "win32" and "TCL_LIBRARY" not in os.environ:
    _base = sys.base_prefix  # e.g. C:\Program Files\Python313, C:\Users\X\AppData\...
    _tcl = os.path.join(_base, "tcl", "tcl8.6")
    _tk = os.path.join(_base, "tcl", "tk8.6")
    if os.path.isdir(_tcl):
        os.environ["TCL_LIBRARY"] = _tcl
        if os.path.isdir(_tk):
            os.environ["TK_LIBRARY"] = _tk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pec.config import SimulationConfig, MLConfig, CacheConfig
from pec.simulator import run_comparison
from pec.metrics import MetricsReporter

DATA_DIR = Path(__file__).parent.parent / "data"
COLORS = {"ML-Driven": "#6ee7b7", "LRU": "#60a5fa", "LFU": "#f87171", "FIFO": "#7a8ba8"}
BG = "#0b0e14"; SURFACE = "#131820"; SURFACE2 = "#1a2030"; BORDER = "#232d3f"
TEXT = "#e0e6f0"; DIM = "#7a8ba8"; ACCENT = "#6ee7b7"; ACCENT2 = "#38bdf8"

class QueueHandler(logging.Handler):
    def __init__(self, q): super().__init__(); self.q = q
    def emit(self, record): self.q.put(self.format(record))

class PECDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Predictive Edge Caching — Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG)
        self.root.minsize(1000, 700)
        self.stop_event = threading.Event()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.results: dict[str, MetricsReporter] = {}
        self._setup_logging()
        self._build_styles()
        self._build_ui()
        self._poll_log()

    def _setup_logging(self):
        h = QueueHandler(self.log_queue)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logging.getLogger("pec").addHandler(h)
        logging.getLogger("pec").setLevel(logging.INFO)

    def _build_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, fieldbackground=SURFACE, borderwidth=0)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("TLabelframe", background=BG, foreground=ACCENT, font=("Segoe UI", 10, "bold"))
        s.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
        s.configure("TButton", background=SURFACE2, foreground=TEXT, font=("Segoe UI", 10, "bold"), padding=6)
        s.map("TButton", background=[("active", BORDER)])
        s.configure("Run.TButton", background="#1a3a2a", foreground=ACCENT)
        s.configure("Stop.TButton", background="#3a1a1a", foreground="#f87171")
        s.configure("TCheckbutton", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=SURFACE2)
        s.configure("Treeview", background=SURFACE, foreground=TEXT, fieldbackground=SURFACE,
                     rowheight=28, font=("Segoe UI", 10))
        s.configure("Treeview.Heading", background=SURFACE2, foreground=ACCENT,
                     font=("Segoe UI", 10, "bold"))
        s.map("Treeview", background=[("selected", BORDER)])
        s.configure("KPI.TLabel", font=("Segoe UI", 22, "bold"))
        s.configure("KPISub.TLabel", font=("Segoe UI", 9), foreground=DIM)
        s.configure("KPITitle.TLabel", font=("Segoe UI", 8, "bold"), foreground=DIM)
        s.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=ACCENT)
        s.configure("TScale", background=BG, troughcolor=SURFACE2)

    def _build_ui(self):
        # Header
        hdr = ttk.Frame(self.root); hdr.pack(fill="x", padx=12, pady=(10,0))
        ttk.Label(hdr, text="\u26a1 Predictive Edge Caching", style="Header.TLabel").pack(side="left")
        self.status_lbl = ttk.Label(hdr, text="\u25cf Idle", foreground=DIM); self.status_lbl.pack(side="right")

        # Top pane: config + architecture
        top = ttk.Frame(self.root); top.pack(fill="x", padx=12, pady=8)
        self._build_config(top)
        self._build_arch(top)

        # KPI row
        self.kpi_frame = ttk.Frame(self.root); self.kpi_frame.pack(fill="x", padx=12, pady=(0,4))
        self.kpi_labels = {}
        for pol in ["ML-Driven", "LRU", "LFU", "FIFO"]:
            f = tk.Frame(self.kpi_frame, bg=SURFACE, highlightbackground=COLORS[pol], highlightthickness=2, padx=14, pady=8)
            f.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Label(f, text=pol, style="KPITitle.TLabel", background=SURFACE).pack(anchor="w")
            v = ttk.Label(f, text="—", style="KPI.TLabel", foreground=COLORS[pol], background=SURFACE)
            v.pack(anchor="w")
            sub = ttk.Label(f, text="Hits: — · Misses: —", style="KPISub.TLabel", background=SURFACE)
            sub.pack(anchor="w")
            self.kpi_labels[pol] = (v, sub)

        # Charts
        self._build_charts()

        # Table + Log
        bot = ttk.Frame(self.root); bot.pack(fill="both", expand=True, padx=12, pady=(0,8))
        self._build_table(bot)
        self._build_log(bot)

    def _build_config(self, parent):
        cf = ttk.LabelFrame(parent, text="  Configuration  ", padding=10)
        cf.pack(side="left", fill="y", padx=(0,8))
        # Trace
        ttk.Label(cf, text="Trace File:").pack(anchor="w")
        tf = ttk.Frame(cf); tf.pack(fill="x", pady=(0,6))
        traces = sorted([f.name for f in DATA_DIR.glob("*.csv")]) if DATA_DIR.exists() else []
        self.trace_var = tk.StringVar(value=traces[0] if traces else "generated_trace.csv")
        cb = ttk.Combobox(tf, textvariable=self.trace_var, values=traces, width=20); cb.pack(side="left", fill="x", expand=True)
        ttk.Button(tf, text="...", width=3, command=self._browse_trace).pack(side="right", padx=(4,0))
        # Sliders
        self.cache_var = tk.IntVar(value=100)
        self._add_slider(cf, "Cache (MB):", self.cache_var, 10, 500)
        self.hist_var = tk.IntVar(value=256)
        self._add_slider(cf, "History Window:", self.hist_var, 64, 1024)
        self.retrain_var = tk.IntVar(value=256)
        self._add_slider(cf, "Retrain Interval:", self.retrain_var, 64, 1024)
        self.lr_var = tk.DoubleVar(value=0.05)
        self._add_slider(cf, "Learning Rate:", self.lr_var, 0.01, 0.5, resolution=0.01)
        # Policies
        ttk.Label(cf, text="Policies:").pack(anchor="w", pady=(6,2))
        self.pol_vars = {}
        for p in ["lru", "lfu", "fifo", "ml_driven"]:
            v = tk.BooleanVar(value=True); self.pol_vars[p] = v
            ttk.Checkbutton(cf, text=p.upper().replace("_","-"), variable=v).pack(anchor="w")
        # Buttons
        bf = ttk.Frame(cf); bf.pack(fill="x", pady=(10,0))
        self.run_btn = ttk.Button(bf, text="\u25b6 Run", style="Run.TButton", command=self._run)
        self.run_btn.pack(fill="x", pady=2)
        self.stop_btn = ttk.Button(bf, text="\u25a0 Stop", style="Stop.TButton", command=self._stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        ttk.Button(bf, text="\u21ba Reset", command=self._reset).pack(fill="x", pady=2)
        self.progress = ttk.Progressbar(cf, mode="determinate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=(8,0))
        self.prog_lbl = ttk.Label(cf, text="", foreground=DIM); self.prog_lbl.pack(anchor="w")

    def _add_slider(self, parent, label, var, lo, hi, resolution=1):
        f = ttk.Frame(parent); f.pack(fill="x", pady=2)
        ttk.Label(f, text=label).pack(side="left")
        vl = ttk.Label(f, text=str(var.get()), foreground=ACCENT, width=6)
        vl.pack(side="right")
        s = tk.Scale(f, from_=lo, to=hi, variable=var, orient="horizontal", resolution=resolution,
                     bg=BG, fg=TEXT, troughcolor=SURFACE2, highlightthickness=0, sliderrelief="flat",
                     showvalue=False, command=lambda v, l=vl, va=var: l.configure(text=f"{va.get():.2f}" if isinstance(va.get(), float) else str(va.get())))
        s.pack(side="left", fill="x", expand=True, padx=4)

    def _build_arch(self, parent):
        af = ttk.LabelFrame(parent, text="  System Architecture  ", padding=6)
        af.pack(side="left", fill="both", expand=True)
        self.arch_canvas = tk.Canvas(af, bg=SURFACE, highlightthickness=0, height=180)
        self.arch_canvas.pack(fill="both", expand=True)
        self.arch_canvas.bind("<Configure>", self._draw_arch)

    def _draw_arch(self, event=None):
        c = self.arch_canvas; c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50: return
        blocks = [("User\nEquipment", "#60a5fa"), ("Traffic\nEngine", "#a78bfa"),
                  ("Feature\nExtractor", "#fbbf24"), ("ML Predictor\n(XGBoost)", ACCENT),
                  ("Edge Cache\nController", "#f87171"), ("Cache\nStore", "#7a8ba8")]
        n = len(blocks); bw = min(120, (w-40)//n - 10); bh = 52; gap = (w - n*bw) / (n+1)
        for i, (txt, col) in enumerate(blocks):
            x = gap + i*(bw+gap); y = h//2 - bh//2
            c.create_rectangle(x, y, x+bw, y+bh, fill=SURFACE2, outline=col, width=2)
            c.create_text(x+bw//2, y+bh//2, text=txt, fill=TEXT, font=("Segoe UI", 8), justify="center")
            if i < n-1:
                ax = x+bw+2; ax2 = ax+gap-4
                c.create_line(ax, h//2, ax2, h//2, fill=col, arrow="last", width=2)
        # Feedback arrow
        fx1 = gap + (n-1)*(bw+gap) + bw//2; fx0 = gap + bw//2; fy = h//2 + bh//2 + 14
        c.create_line(fx1, h//2+bh//2, fx1, fy, fx0, fy, fx0, h//2+bh//2,
                      fill="#f87171", arrow="last", width=1, dash=(4,2))
        c.create_text((fx0+fx1)//2, fy+10, text="Feedback Loop", fill=DIM, font=("Segoe UI", 7))

    def _build_charts(self):
        cf = ttk.Frame(self.root); cf.pack(fill="both", expand=True, padx=12)
        plt.style.use("dark_background")
        self.fig = Figure(figsize=(14, 5), dpi=85, facecolor=BG)
        self.fig.subplots_adjust(hspace=0.45, wspace=0.3, left=0.05, right=0.97, top=0.92, bottom=0.12)
        self.axes = [self.fig.add_subplot(1, 4, i+1) for i in range(4)]
        titles = ["Hit Rate Over Time", "Final Comparison", "Cache Utilisation", "ML Training"]
        for ax, t in zip(self.axes, titles):
            ax.set_title(t, fontsize=9, color=DIM); ax.tick_params(labelsize=7)
            ax.set_facecolor(SURFACE)
        self.chart_canvas = FigureCanvasTkAgg(self.fig, cf)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_table(self, parent):
        tf = ttk.LabelFrame(parent, text="  Detailed Comparison  ", padding=6)
        tf.pack(side="left", fill="both", expand=True, padx=(0,4))
        cols = ("policy","requests","hits","misses","hit_rate","miss_rate","vs_lru")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=5)
        for c, h, w in zip(cols, ["Policy","Requests","Hits","Misses","Hit Rate","Miss Rate","vs LRU"],
                            [100, 90, 80, 80, 80, 80, 80]):
            self.tree.heading(c, text=h); self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)

    def _build_log(self, parent):
        lf = ttk.LabelFrame(parent, text="  Event Log  ", padding=6)
        lf.pack(side="right", fill="both", expand=True, padx=(4,0))
        self.log_text = tk.Text(lf, bg=SURFACE, fg=TEXT, font=("Consolas", 9), wrap="word",
                                height=5, insertbackground=TEXT, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(lf, command=self.log_text.yview); self.log_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.log_text.pack(fill="both", expand=True)

    def _browse_trace(self):
        p = filedialog.askopenfilename(initialdir=str(DATA_DIR), filetypes=[("CSV","*.csv")])
        if p: self.trace_var.set(Path(p).name)

    def _poll_log(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert("end", msg + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _run(self):
        pols = [p for p, v in self.pol_vars.items() if v.get()]
        if not pols:
            messagebox.showwarning("No Policies", "Select at least one policy."); return
        trace = DATA_DIR / self.trace_var.get()
        if not trace.exists():
            messagebox.showerror("Missing Trace", f"File not found:\n{trace}"); return
        self.run_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.status_lbl.config(text="\u25cf Running...", foreground=ACCENT)
        self.stop_event.clear()
        self.progress["value"] = 0
        cfg = SimulationConfig(
            trace_path=str(trace), history_window=self.hist_var.get(),
            ml=MLConfig(learning_rate=self.lr_var.get(), retrain_interval=self.retrain_var.get()),
            cache=CacheConfig(capacity_bytes=self.cache_var.get() * 1024 * 1024),
            policies_to_compare=pols,
        )
        self.log_text.delete("1.0", "end")
        self.log_queue.put(f"Starting simulation: {len(pols)} policies, trace={trace.name}")
        self._n_policies = len(pols); self._pol_idx = 0
        t = threading.Thread(target=self._worker, args=(cfg,), daemon=True)
        t.start()

    def _worker(self, cfg):
        try:
            results = run_comparison(cfg, self.stop_event, self._progress_cb)
            self.root.after(0, self._on_done, results)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _progress_cb(self, policy_name, idx, total):
        if self._last_policy != policy_name if hasattr(self, '_last_policy') else True:
            self._pol_idx = min(self._pol_idx + (1 if hasattr(self, '_last_policy') else 0), self._n_policies - 1)
            self._last_policy = policy_name
        pct = (self._pol_idx / self._n_policies + (idx / total) / self._n_policies) * 100
        self.root.after(0, self._update_progress, pct, policy_name, idx, total)

    def _update_progress(self, pct, name, idx, total):
        self.progress["value"] = pct
        self.prog_lbl.config(text=f"{name}: {idx:,}/{total:,}")

    def _on_error(self, msg):
        self.status_lbl.config(text="\u25cf Error", foreground="#f87171")
        self.run_btn.config(state="normal"); self.stop_btn.config(state="disabled")
        messagebox.showerror("Simulation Error", msg)

    def _on_done(self, results: dict[str, MetricsReporter]):
        self.results = results
        self.progress["value"] = 100
        self.status_lbl.config(text="\u25cf Done", foreground=ACCENT)
        self.run_btn.config(state="normal"); self.stop_btn.config(state="disabled")
        stopped = self.stop_event.is_set()
        self.log_queue.put("Simulation " + ("stopped by user." if stopped else "complete."))
        self._update_kpis()
        self._update_charts()
        self._update_table()

    def _stop(self):
        self.stop_event.set()
        self.log_queue.put("Stopping simulation...")

    def _reset(self):
        self.stop_event.set()
        self.results = {}
        for v, sub in self.kpi_labels.values():
            v.config(text="—"); sub.config(text="Hits: — · Misses: —")
        for ax in self.axes: ax.clear()
        self.chart_canvas.draw()
        for r in self.tree.get_children(): self.tree.delete(r)
        self.log_text.delete("1.0", "end")
        self.progress["value"] = 0; self.prog_lbl.config(text="")
        self.status_lbl.config(text="\u25cf Idle", foreground=DIM)
        self.run_btn.config(state="normal"); self.stop_btn.config(state="disabled")

    def _update_kpis(self):
        for pol, reporter in self.results.items():
            if pol not in self.kpi_labels: continue
            v_lbl, sub_lbl = self.kpi_labels[pol]
            snaps = reporter.snapshots
            if not snaps: continue
            last = snaps[-1]
            v_lbl.config(text=f"{last.cumulative_hit_rate*100:.2f}%")
            sub_lbl.config(text=f"Hits: {last.cumulative_hits:,} \u00b7 Misses: {last.cumulative_misses:,}")

    def _update_charts(self):
        for ax in self.axes: ax.clear()
        if not self.results: self.chart_canvas.draw(); return
        # Chart 1: Hit Rate Over Time
        ax0 = self.axes[0]
        for name, rep in self.results.items():
            snaps = rep.snapshots
            step = max(1, len(snaps)//500)
            xs = [s.request_idx for s in snaps[::step]]
            ys = [s.windowed_hit_rate*100 for s in snaps[::step]]
            ax0.plot(xs, ys, label=name, color=COLORS.get(name, "#888"), linewidth=1.2)
        ax0.set_title("Hit Rate Over Time", fontsize=9, color=DIM)
        ax0.set_xlabel("Request", fontsize=7); ax0.set_ylabel("%", fontsize=7)
        ax0.legend(fontsize=6, loc="lower right"); ax0.set_ylim(0, 100)
        # Chart 2: Bar comparison
        ax1 = self.axes[1]
        names = list(self.results.keys())
        rates = [r.final_hit_rate*100 for r in self.results.values()]
        cols = [COLORS.get(n, "#888") for n in names]
        bars = ax1.bar(names, rates, color=cols, edgecolor="white", linewidth=0.5)
        for b, r in zip(bars, rates): ax1.text(b.get_x()+b.get_width()/2, r+0.5, f"{r:.1f}%", ha="center", fontsize=7, color=TEXT)
        ax1.set_title("Final Hit Rate", fontsize=9, color=DIM); ax1.set_ylabel("%", fontsize=7)
        ax1.tick_params(axis='x', labelsize=7)
        # Chart 3: Cache Utilisation
        ax2 = self.axes[2]
        for name, rep in self.results.items():
            snaps = rep.snapshots; step = max(1, len(snaps)//500)
            ax2.plot([s.request_idx for s in snaps[::step]], [s.cache_utilisation*100 for s in snaps[::step]],
                     label=name, color=COLORS.get(name, "#888"), linewidth=1.2)
        ax2.set_title("Cache Utilisation", fontsize=9, color=DIM)
        ax2.set_xlabel("Request", fontsize=7); ax2.set_ylabel("%", fontsize=7)
        ax2.legend(fontsize=6); ax2.set_ylim(0, 105)
        # Chart 4: ML Training
        ax3 = self.axes[3]
        ml = self.results.get("ML-Driven")
        if ml and ml.snapshots:
            snaps = ml.snapshots; step = max(1, len(snaps)//500)
            ax3.plot([s.request_idx for s in snaps[::step]], [s.ml_training_steps for s in snaps[::step]],
                     color=ACCENT, linewidth=1.5)
            ax3.set_title("ML Training Steps", fontsize=9, color=DIM)
            ax3.set_xlabel("Request", fontsize=7); ax3.set_ylabel("Steps", fontsize=7)
        else:
            ax3.set_title("ML Training (N/A)", fontsize=9, color=DIM)
            ax3.text(0.5, 0.5, "No ML policy", transform=ax3.transAxes, ha="center", color=DIM)
        self.chart_canvas.draw()

    def _update_table(self):
        for r in self.tree.get_children(): self.tree.delete(r)
        lru_rate = self.results.get("LRU", None)
        lru_hr = lru_rate.final_hit_rate if lru_rate else 0
        best = max((r.final_hit_rate for r in self.results.values()), default=0)
        for name, rep in self.results.items():
            hr = rep.final_hit_rate; snaps = rep.snapshots
            if not snaps: continue
            last = snaps[-1]; h = last.cumulative_hits; m = last.cumulative_misses
            vs = "—" if name == "LRU" else f"{((hr-lru_hr)/lru_hr*100) if lru_hr else 0:+.1f}%"
            tag = "winner" if abs(hr - best) < 1e-6 else ""
            self.tree.insert("", "end", values=(
                ("\u2605 " if tag else "") + name, f"{h+m:,}", f"{h:,}", f"{m:,}",
                f"{hr*100:.2f}%", f"{(1-hr)*100:.2f}%", vs))


def main():
    root = tk.Tk()
    app = PECDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
