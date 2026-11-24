#!/usr/bin/env python3
"""
Generates `docs/architecture.png` -- the pipeline figure referenced from
the top-level README.

Kept as a small, versioned script (rather than a hand-drawn image file
with no provenance) so the figure can be regenerated or tweaked the
same way the rest of this project is built: run it, read a diff, move
on. It has no runtime dependency on `lmsafe` itself -- it only draws
boxes and arrows describing the package's own data flow -- so it is
safe to run in any environment with matplotlib installed:

    python docs/make_architecture_diagram.py

Layout (top to bottom):
  1. Data sources (LANL / OpTC / synthetic)      -> a common `Event` stream
  2. Fair protocol + windowing                   -> train/test `WindowGraph`s
  3. The experiment loop: Detector <-> UpdatePolicy, one of five policies
  4. The boiling-frog conditioning attack feeding into that loop (RQ2/RQ3)
  5. Metrics -> the three research questions (RQ1/RQ2/RQ3)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

# -- palette: muted, print-safe, colorblind-conscious -----------------
INK = "#1f2430"
BG = "#ffffff"
BLUE = "#3B6EA5"      # data / sources
TEAL = "#2E8B8B"      # processing / windows
PURPLE = "#6B5B95"    # the core loop
AMBER = "#C97A2B"     # attack / adversarial
GREEN = "#4C8C4A"     # metrics / RQs
GREY = "#8A8F98"      # connective text / neutral boxes


def box(ax, xy, w, h, text, face, edge=INK, fontsize=10.5, fontweight="normal", text_color="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3,
        facecolor=face,
        edgecolor=edge,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, fontweight=fontweight, linespacing=1.35)
    return patch


def arrow(ax, p0, p1, color=INK, style="-|>", lw=1.6, connectionstyle="arc3,rad=0.0", ls="solid"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, connectionstyle=connectionstyle, linestyle=ls)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 15.5), dpi=200)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 15.5)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.text(6.5, 15.1, "lmsafe -- experimental pipeline", ha="center", va="center",
             fontsize=18, fontweight="bold", color=INK)
    ax.text(6.5, 14.65,
             "Reliable Graph Foundation Models for Lateral Movement Detection:\n"
             "Cross-Network Transfer under Drift and Contaminated Baselines",
             ha="center", va="center", fontsize=10.5, color=GREY, style="italic")

    # ---------------------------------------------------------------
    # Row 1: data sources -> lmsafe.data.schema.Event
    # ---------------------------------------------------------------
    y1 = 13.0
    box(ax, (0.4, y1), 3.1, 1.0, "LANL auth.txt.gz\n(streamed, never\ndownloaded whole)\nlmsafe.data.lanl", BLUE)
    box(ax, (4.2, y1), 3.1, 1.0, "OpTC\n(documented stub)\nlmsafe.data.optc", GREY, text_color="white")
    box(ax, (8.0, y1), 3.1, 1.0, "Synthetic generator\n(Zipf traffic, drift,\nLM episodes)\nlmsafe.synth", BLUE)

    box(ax, (3.5, y1 - 1.55), 5.6, 0.9, "Event  (src_user, dst_user, src_comp, dst_comp,\nt, logon_type, is_malicious*)\nlmsafe.data.schema", TEAL)
    arrow(ax, (1.95, y1), (5.3, y1 - 0.65))
    arrow(ax, (5.75, y1), (6.3, y1 - 0.65))
    arrow(ax, (9.55, y1), (7.5, y1 - 0.65))
    ax.text(12.6, y1 + 0.5, "*is_malicious:\nevaluation-only,\nnever read by a\nDetector/Policy", fontsize=8, color=GREY, ha="right", va="center", style="italic")

    legend_elems = [
        Line2D([0], [0], color=INK, lw=1.8, label="benign data flow"),
        Line2D([0], [0], color=AMBER, lw=1.8, linestyle="dashed", label="adversarial / gated-\ndefense path (RQ2, RQ3)"),
        Line2D([0], [0], color=GREY, lw=1.6, linestyle="dashed", label="drift-detector signal\n(DriftTriggered only)"),
    ]
    leg = ax.legend(handles=legend_elems, loc="center", bbox_to_anchor=(0.925, 0.665),
                     bbox_transform=fig.transFigure, ncol=1, frameon=True, framealpha=0.9,
                     fontsize=7.6, labelcolor=INK, title="Arrow key", title_fontsize=8.2,
                     edgecolor=GREY, handlelength=1.8, labelspacing=1.1)
    ax.add_artist(leg)

    # ---------------------------------------------------------------
    # Row 2: fair protocol + windowing -> train/test split
    # ---------------------------------------------------------------
    y2 = 10.1
    box(ax, (3.5, y2), 5.6, 0.9,
        "Fair evaluation protocol: keep all remote LogOn events,\n"
        "label-join on (t, dst_user, src_comp, dst_comp)\nlmsafe.data.lanl.apply_fair_protocol",
        TEAL)
    arrow(ax, (6.3, y1 - 0.65), (6.3, y2 + 0.9))

    y3 = 8.75
    box(ax, (3.5, y3), 5.6, 0.75, "iter_windows()  ->  WindowGraph per time bucket\nlmsafe.data.windows", TEAL)
    arrow(ax, (6.3, y2), (6.3, y3 + 0.75))

    y4 = 7.55
    box(ax, (1.2, y4), 3.6, 0.8, "Train windows\n( t < 41 h )", TEAL, edge=INK)
    box(ax, (6.6, y4), 2.9, 0.8, "Test windows\n( t >= 41 h )", TEAL, edge=INK)
    arrow(ax, (4.8, y3), (3.0, y4 + 0.8))
    arrow(ax, (7.5, y3), (8.0, y4 + 0.8))

    # ---------------------------------------------------------------
    # Row 3: the attack module feeding into the test stream
    # ---------------------------------------------------------------
    box(ax, (10.0, y4), 2.7, 0.8,
        '"Boiling-frog"\nconditioning attack\n(RQ2/RQ3 only)\nlmsafe.attack', AMBER, edge=INK, fontsize=9)
    arrow(ax, (10.0, y4 + 0.4), (9.5, y4 + 0.4), color=AMBER, ls="dashed")

    # ---------------------------------------------------------------
    # Row 4: the core experiment loop
    # ---------------------------------------------------------------
    y5 = 5.55
    box(ax, (0.4, y5), 2.6, 0.75, "fit_initial()\n(benign 41h fit)", GREY, text_color="white")
    arrow(ax, (3.0, y4), (1.7, y5 + 0.75))

    loop_x, loop_y, loop_w, loop_h = 0.4, 2.55, 12.2, 2.65
    ax.add_patch(FancyBboxPatch((loop_x, loop_y), loop_w, loop_h,
                                 boxstyle="round,pad=0.03,rounding_size=0.12",
                                 linewidth=1.6, facecolor="#F3F0FA", edgecolor=PURPLE))
    ax.text(loop_x + 0.15, loop_y + loop_h - 0.25,
            "per-window loop, one (detector, policy) cell  --  lmsafe.runner.run_experiment_cell",
            fontsize=9.5, color=PURPLE, fontweight="bold")

    box(ax, (0.9, loop_y + 1.25), 3.0, 0.8, "Detector.score(window)\n(CountBaseline / FeatureMLP)", PURPLE)
    box(ax, (5.1, loop_y + 1.25), 3.0, 0.8, "WindowScores\n(scores + labels + weight)", TEAL)
    box(ax, (9.3, loop_y + 1.25), 2.6, 0.8, "metrics\ncollector", GREEN)
    arrow(ax, (3.9, loop_y + 1.65), (5.1, loop_y + 1.65))
    arrow(ax, (8.1, loop_y + 1.65), (9.3, loop_y + 1.65))

    box(ax, (0.9, loop_y + 0.15), 3.0, 0.8,
        "UpdatePolicy.observe(window, scores, detector)\n"
        "Frozen | Periodic | SlidingReplay |\nDriftTriggered | Gated", PURPLE, fontsize=9)
    arrow(ax, (0.75, loop_y + 0.55), (0.75, loop_y + 1.65), color=PURPLE, lw=1.4,
          connectionstyle="arc3,rad=-0.9")
    ax.text(0.15, loop_y + 1.1, "updated\ndetector\nfor next\nwindow", fontsize=7.3, color=GREY, ha="right", va="center")

    box(ax, (5.1, loop_y + 0.15), 3.0, 0.8,
        "GatedPolicy only:\nscore-gated admission ->\nquarantine -> multi-host\ncorroboration -> commit", AMBER, fontsize=8.3)
    arrow(ax, (3.9, loop_y + 0.55), (5.1, loop_y + 0.55), color=AMBER, ls="dashed")

    box(ax, (9.3, loop_y + 0.15), 2.6, 0.8, "Drift detectors:\nPageHinkley /\nSimplifiedADWIN\n(DriftTriggered only)", GREY, fontsize=8.3)
    arrow(ax, (9.3, loop_y + 0.55), (8.1, loop_y + 0.55), color=GREY, lw=1.2, ls="dashed")

    # ---------------------------------------------------------------
    # Row 5: metrics -> research questions
    # ---------------------------------------------------------------
    y6 = 1.55
    box(ax, (0.6, y6), 4.4, 0.75,
        "weighted AUC/AP, Rec@B, episode\nclustering + delay, block/episode\nbootstrap CIs  --  lmsafe.metrics", GREEN, fontsize=9)
    arrow(ax, (10.6, loop_y), (2.8, y6 + 0.75))

    box(ax, (0.3, 0.1), 3.9, 1.0,
        "RQ1\ndrift cost per policy,\nno attack\n(run_rq1_drift_cost.py)", GREEN, edge=INK)
    box(ax, (4.55, 0.1), 3.9, 1.0,
        "RQ2\nupdate-poisoning via\nconditioning attack\n(run_rq2_poisoning.py)", AMBER, edge=INK)
    box(ax, (8.8, 0.1), 3.9, 1.0,
        "RQ3\ndoes Gated win on\nBOTH axes at once?\n(run_rq3_gated_defense.py)", PURPLE, edge=INK)

    arrow(ax, (1.8, y6), (2.2, 1.1))
    arrow(ax, (3.6, y6), (6.3, 1.1))
    arrow(ax, (4.6, y6), (10.5, 1.1))

    fig.tight_layout(pad=0.6)
    out_path = "docs/architecture.png"
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
