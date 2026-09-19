"""Render a 1–2 page PDF technical report from the experimental artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_reward(d: dict, key_mean="mean_reward", key_std="std_reward") -> str:
    if not d or d.get(key_mean) is None:
        return "n/a"
    if d.get(key_std) is not None:
        return f"{d[key_mean]:.1f} ± {d[key_std]:.1f}"
    return f"{d[key_mean]:.1f}"


def _fmt_minutes(seconds) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds / 60.0:.1f} min"


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, "Legged Locomotion via Reinforcement Learning  |  Technical report", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def _section(pdf: Report, title: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(26, 115, 232)
    pdf.set_line_width(0.4)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3)


def _body(pdf: Report, text: str) -> None:
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 5, text)
    pdf.ln(1)


def build_report(out_path: Path | None = None) -> Path:
    out_path = out_path or (ROOT / "report.pdf")
    ppo_eval = _load(RESULTS / "ppo_eval.json")
    sac_eval = _load(RESULTS / "sac_eval.json")
    ppo_stats = _load(RESULTS / "ppo_stats.json")
    sac_stats = _load(RESULTS / "sac_stats.json")
    robust = _load(RESULTS / "robustness.json")
    table = _load(RESULTS / "comparison_table.json")

    ppo_r = _fmt_reward(ppo_eval)
    sac_r = _fmt_reward(sac_eval)
    ppo_t = _fmt_minutes(ppo_stats.get("wall_clock_seconds"))
    sac_t = _fmt_minutes(sac_stats.get("wall_clock_seconds"))

    def se_for(algo: str) -> str:
        for row in table:
            if str(row.get("algo", "")).lower() == algo:
                v = row.get("steps_to_threshold")
                return f"{v:,} steps" if v else "not reached"
        return "n/a"

    drop = robust.get("percent_drop")
    drop_s = f"{drop:.1f}% relative drop" if drop is not None else "n/a"
    nom = robust.get("nominal") or {}
    rnd = robust.get("randomized") or {}
    rec = robust.get("recovered")

    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(15, 15, 15)
    pdf.multi_cell(0, 8, "Simulated Quadruped Locomotion with PPO, SAC, and Domain Randomization")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Gymnasium Ant-v4  |  Stable-Baselines3  |  MuJoCo", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _section(pdf, "1. Motivation")
    _body(
        pdf,
        "Physical quadruped platforms are expensive to iterate on, so this project uses "
        "simulation to go deeper on the control and learning side of legged locomotion. "
        "The task is the standard Ant-v4 environment: a four-legged MuJoCo robot that must "
        "walk forward. Two off-the-shelf RL algorithms (PPO and SAC) are trained with the "
        "same 1 million timestep budget, then the better-documented failure mode of "
        "sim-to-real — mismatch in contact friction and inertial parameters — is probed "
        "with domain randomization, without claiming a real-robot transfer result.",
    )

    _section(pdf, "2. Method")
    _body(
        pdf,
        "The agent observes Ant-v4's default state (joint angles, velocities, and torso "
        "pose-related features) and outputs continuous torques. PPO is trained with eight "
        "vectorized environments and otherwise Stable-Baselines3 defaults. SAC is trained "
        "with a single environment and SB3 defaults, matching the usual on-policy vs "
        "off-policy setup. Episode reward and length are logged to TensorBoard and Monitor "
        "CSVs. Domain randomization, applied at every reset, independently scales each "
        "geom's sliding friction by U(0.5, 1.5) and the torso mass (and matching inertia) "
        "by U(0.7, 1.3). The already-trained policy is evaluated on this distribution "
        "with no gradient updates; a short fine-tune with randomization on is optional.",
    )

    _section(pdf, "3. Results")
    _body(
        pdf,
        f"After 1M environment steps, PPO reached a mean evaluation reward of {ppo_r} "
        f"(wall-clock {ppo_t}). SAC reached {sac_r} (wall-clock {sac_t}). "
        f"Sample efficiency, defined as the first time the rolling-100 training reward "
        f"crosses 2000, was {se_for('ppo')} for PPO and {se_for('sac')} for SAC. "
        "On Ant-v4 a random policy typically stays near zero to low hundreds of reward "
        "and does not produce a stable gait; the trained policies produce a visible "
        "forward walking gait (see the GIF in the repository README).",
    )

    curve = RESULTS / "comparison_curves.png"
    if not curve.exists():
        curve = RESULTS / "ppo_reward_curve.png"
    if curve.exists():
        pdf.ln(1)
        # Keep the figure compact so the report stays at 1–2 pages.
        pdf.image(str(curve), w=180)
        pdf.ln(2)

    _section(pdf, "4. Robustness (simplified sim-to-real)")
    rec_s = ""
    if rec:
        rec_s = (
            f" Fine-tuning on the randomized environment recovered a mean reward of "
            f"{_fmt_reward(rec)} on held-out randomized episodes."
        )
    _body(
        pdf,
        f"The frozen baseline policy scored {_fmt_reward(nom)} on nominal physics and "
        f"{_fmt_reward(rnd)} when friction and torso mass were randomized at every "
        f"episode ({drop_s}). That gap is the point of the experiment: a gait that looks "
        f"solved in the default XML is not automatically stable under the kind of "
        f"parameter error that shows up when a controller leaves the simulator."
        f"{rec_s}",
    )

    bar = RESULTS / "robustness_bars.png"
    if bar.exists():
        pdf.image(str(bar), w=150)
        pdf.ln(2)

    _section(pdf, "5. Limitations")
    _body(
        pdf,
        "Ant-v4 is a convenient quadruped analogue, not a model of a specific robot, so "
        "this is not sim-to-real in the hardware sense. Randomizing only friction and "
        "torso mass leaves out latency, actuator dynamics, sensor noise, and terrain. "
        "A single seed per algorithm is reported; the table should be read as a paired "
        "comparison under a fixed budget, not as a claim about the algorithm class. "
        "Hyperparameters were not swept — SB3 defaults were used on purpose, so the "
        "comparison reflects typical practitioner settings rather than each method at "
        "its published best.",
    )

    pdf.output(str(out_path))
    return out_path


if __name__ == "__main__":
    path = build_report()
    print(f"Wrote {path}")
