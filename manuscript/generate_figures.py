#!/usr/bin/env python3
"""
Generate figures for: The Subspace Penalty Kick
Uses matplotlib for diagrams and optionally Google's Imagen API for photorealistic images.

For Imagen API: requires GOOGLE_API_KEY environment variable.
Falls back to matplotlib-only diagrams if API is unavailable.

Usage:
    pip install matplotlib numpy requests Pillow
    python generate_figures.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# --- Configuration ---
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Colors matching the paper
KEEPER_BLUE = '#3B82F6'
STRIKER_RED = '#EF4444'
FIELD_GREEN = '#10B981'
WARN_ORANGE = '#F59E0B'
PURPLE = '#8B5CF6'
CYAN = '#06B6D4'
DARK_BG = '#111827'
CARD_BG = '#1A2236'
TEXT_COLOR = '#E8EAF0'
MUTED = '#6B7280'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#F8FAFC',
    'axes.edgecolor': '#CBD5E1',
    'axes.labelcolor': '#334155',
    'xtick.color': '#64748B',
    'ytick.color': '#64748B',
    'text.color': '#1E293B',
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#E2E8F0',
})


def fig1_task_mapping():
    """Figure 1: Mapping the monkey task onto the penalty kick."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Monkey task structure
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Monkey Categorisation Tasks\n(Tafazoli et al. 2026)', fontsize=11, fontweight='bold', pad=10)

    # Tasks as boxes
    tasks = [
        ('S1: Shape→Axis 1', 1.5, 7.5, WARN_ORANGE),
        ('C1: Colour→Axis 1', 5, 7.5, FIELD_GREEN),
        ('C2: Colour→Axis 2', 8.5, 7.5, KEEPER_BLUE),
    ]
    for label, x, y, color in tasks:
        box = FancyBboxPatch((x - 1.3, y - 0.6), 2.6, 1.2, boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color=color)

    # Shared components
    ax.annotate('', xy=(3.7, 7.5), xytext=(6.3, 7.5),
                arrowprops=dict(arrowstyle='<->', color=KEEPER_BLUE, lw=2))
    ax.text(5, 8.4, 'Shared: Colour', ha='center', fontsize=7, color=KEEPER_BLUE, style='italic')

    ax.annotate('', xy=(1.5, 6.6), xytext=(5, 6.6),
                arrowprops=dict(arrowstyle='<->', color=WARN_ORANGE, lw=2))
    ax.text(3.25, 6.1, 'Shared: Axis 1', ha='center', fontsize=7, color=WARN_ORANGE, style='italic')

    # Subspace labels
    ax.text(5, 4.5, 'Sensory Subspaces', ha='center', fontsize=9, fontweight='bold')
    ax.text(3, 3.7, 'Colour', ha='center', fontsize=8, color=KEEPER_BLUE,
            bbox=dict(boxstyle='round', facecolor=KEEPER_BLUE, alpha=0.15))
    ax.text(7, 3.7, 'Shape', ha='center', fontsize=8, color=WARN_ORANGE,
            bbox=dict(boxstyle='round', facecolor=WARN_ORANGE, alpha=0.15))

    ax.text(5, 1.8, 'Motor Subspaces', ha='center', fontsize=9, fontweight='bold')
    ax.text(3, 1.2, 'Axis 1 (UL/LR)', ha='center', fontsize=8, color=WARN_ORANGE,
            bbox=dict(boxstyle='round', facecolor=WARN_ORANGE, alpha=0.15))
    ax.text(7, 1.2, 'Axis 2 (UR/LL)', ha='center', fontsize=8, color=KEEPER_BLUE,
            bbox=dict(boxstyle='round', facecolor=KEEPER_BLUE, alpha=0.15))

    # Right: Penalty kick mapping
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Penalty Kick Tasks\n(Proposed Mapping)', fontsize=11, fontweight='bold', pad=10)

    # Goalkeeper tasks
    gk_tasks = [
        ('Dive Left', 1.5, 7.5, KEEPER_BLUE),
        ('Dive Right', 5, 7.5, CYAN),
        ('Stay/Low', 8.5, 7.5, PURPLE),
    ]
    for label, x, y, color in gk_tasks:
        box = FancyBboxPatch((x - 1.3, y - 0.6), 2.6, 1.2, boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color=color)

    ax.annotate('', xy=(1.5, 6.6), xytext=(5, 6.6),
                arrowprops=dict(arrowstyle='<->', color=STRIKER_RED, lw=2))
    ax.text(3.25, 6.1, 'Shared: Horizontal Axis', ha='center', fontsize=7, color=STRIKER_RED, style='italic')

    # Sensory subspaces
    ax.text(5, 4.5, 'Sensory (Anticipation Cues)', ha='center', fontsize=9, fontweight='bold')
    ax.text(2.5, 3.7, 'Hip Kinematics', ha='center', fontsize=8, color=KEEPER_BLUE,
            bbox=dict(boxstyle='round', facecolor=KEEPER_BLUE, alpha=0.15))
    ax.text(7.5, 3.7, 'Gaze Direction', ha='center', fontsize=8, color=MUTED,
            bbox=dict(boxstyle='round', facecolor=MUTED, alpha=0.15))
    ax.text(7.5, 3.1, '(suppressed by\ngain modulation)', ha='center', fontsize=6,
            color=MUTED, style='italic')

    # Motor subspaces
    ax.text(5, 1.8, 'Motor Subspaces', ha='center', fontsize=9, fontweight='bold')
    ax.text(3, 1.2, 'Horizontal Dive\n(Left/Right)', ha='center', fontsize=8, color=STRIKER_RED,
            bbox=dict(boxstyle='round', facecolor=STRIKER_RED, alpha=0.15))
    ax.text(7.5, 1.2, 'Vertical\n(Stay/Collapse)', ha='center', fontsize=8, color=PURPLE,
            bbox=dict(boxstyle='round', facecolor=PURPLE, alpha=0.15))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_task_mapping.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig1_task_mapping.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated: fig1_task_mapping")


def fig2_interference_predictions():
    """Figure 2: Within-axis vs cross-axis interference predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Within vs cross axis recovery time
    ax = axes[0]
    conditions = ['Within-Axis\n(L→R fake)', 'Cross-Axis\n(Low→High fake)']
    recovery_mean = [280, 150]
    recovery_err = [35, 25]
    bars = ax.bar(conditions, recovery_mean, yerr=recovery_err, capsize=5,
                  color=[STRIKER_RED, PURPLE], alpha=0.7, edgecolor='white', linewidth=1)
    ax.set_ylabel('Recovery Time (ms)', fontweight='bold')
    ax.set_title('A. Predicted Recovery\nAfter Deception', fontweight='bold')
    ax.set_ylim(0, 380)
    ax.axhline(y=200, color=MUTED, linestyle='--', alpha=0.5, label='Typical RT')
    ax.legend(fontsize=7, loc='upper right')

    # Panel B: Task history effect across kicks
    ax = axes[1]
    kicks = np.arange(1, 6)
    # Same-side ghost effect
    same_seq = [0.65, 0.58, 0.62, 0.55, 0.60]  # probability of diving same direction as kick n-2
    diff_seq = [0.50, 0.51, 0.49, 0.52, 0.48]   # control (no pattern)

    ax.plot(kicks, same_seq, 'o-', color=KEEPER_BLUE, lw=2, markersize=8, label='Kick n-2 = same side')
    ax.plot(kicks, diff_seq, 's--', color=MUTED, lw=1.5, markersize=7, label='Kick n-2 = diff side')
    ax.axhline(y=0.5, color=STRIKER_RED, linestyle=':', alpha=0.5, label='Chance')
    ax.fill_between(kicks, 0.48, 0.52, alpha=0.1, color=MUTED)
    ax.set_xlabel('Penalty Kick Number in Shootout', fontweight='bold')
    ax.set_ylabel('P(Same Dive Direction)', fontweight='bold')
    ax.set_title('B. Predicted Task History\nGhost (Kick n-2 Effect)', fontweight='bold')
    ax.set_ylim(0.35, 0.75)
    ax.legend(fontsize=7)

    # Panel C: Shot quality vs keeper commitment timing
    ax = axes[2]
    commit_time = np.linspace(-400, 0, 50)  # ms before ball strike (negative = early)
    # Power + placement combined quality
    np.random.seed(42)
    shot_quality = 0.85 - 0.25 * np.exp(-((commit_time + 100) ** 2) / (2 * 80 ** 2)) + np.random.normal(0, 0.03, 50)
    shot_quality = np.clip(shot_quality, 0.3, 1.0)

    ax.scatter(commit_time, shot_quality, c=commit_time, cmap='RdYlGn', s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.plot(commit_time, np.convolve(shot_quality, np.ones(8)/8, mode='same'), color=FIELD_GREEN, lw=2)
    ax.set_xlabel('Keeper Commitment Time\n(ms before ball strike)', fontweight='bold')
    ax.set_ylabel('Shot Quality Index', fontweight='bold')
    ax.set_title('C. Predicted: Late Keeper\nDegrades Shot Quality', fontweight='bold')
    ax.axvline(x=-150, color=WARN_ORANGE, linestyle='--', alpha=0.7, label='Typical commit')
    ax.legend(fontsize=7, loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_predictions.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig2_predictions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated: fig2_predictions")


def fig3_transformation_window():
    """Figure 3: The sensory-motor transformation window."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: Sensory and motor subspace activation over time
    ax = axes[0]
    time = np.linspace(-200, 500, 300)
    stim_onset = 0  # ball strike

    # Sensory categorisation
    sensory = np.zeros_like(time)
    mask = time > 60
    sensory[mask] = 0.9 * (1 - np.exp(-(time[mask] - 60) / 25)) * np.exp(-np.maximum(0, time[mask] - 300) / 500)
    sensory += np.random.normal(0, 0.02, len(time))
    sensory = np.clip(sensory, 0, 1)

    # Motor commitment
    motor = np.zeros_like(time)
    mask2 = time > 123
    motor[mask2] = 0.95 * (1 - np.exp(-(time[mask2] - 123) / 30))
    motor += np.random.normal(0, 0.02, len(time))
    motor = np.clip(motor, 0, 1)

    ax.plot(time, sensory, color=KEEPER_BLUE, lw=2.5, label='Sensory Categorisation\n("Going right")')
    ax.plot(time, motor, color=STRIKER_RED, lw=2.5, label='Motor Commitment\n(Dive right)')
    ax.axvline(x=0, color=MUTED, linestyle=':', alpha=0.5, label='Ball strike')

    # Highlight the gap
    ax.axvspan(60, 123, alpha=0.1, color=WARN_ORANGE)
    ax.annotate('~63ms\ntransformation\nwindow', xy=(91, 0.5), fontsize=8,
                ha='center', color=WARN_ORANGE, fontweight='bold')
    ax.set_xlabel('Time from ball strike (ms)', fontweight='bold')
    ax.set_ylabel('Subspace Activation', fontweight='bold')
    ax.set_title('A. Sequential Sensory→Motor\nTransformation (Goalkeeper)', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(-200, 500)
    ax.set_ylim(-0.05, 1.1)

    # Panel B: Stutter-step exploitation
    ax = axes[1]
    time2 = np.linspace(-200, 500, 300)

    # With stutter step: sensory loads "right" then switches to "left"
    sensory_fake = np.zeros_like(time2)
    mask_a = (time2 > 40) & (time2 < 150)
    sensory_fake[mask_a] = 0.7 * (1 - np.exp(-(time2[mask_a] - 40) / 20))
    mask_b = time2 >= 150
    sensory_fake[mask_b] = sensory_fake[mask_a][-1] * np.exp(-(time2[mask_b] - 150) / 30) - \
                           0.6 * (1 - np.exp(-(time2[mask_b] - 180) / 25))
    sensory_fake += np.random.normal(0, 0.02, len(time2))

    # Motor follows with delay (partially committed to wrong direction)
    motor_confused = np.zeros_like(time2)
    mask_c = time2 > 100
    motor_confused[mask_c] = 0.4 * (1 - np.exp(-(time2[mask_c] - 100) / 35))
    mask_d = time2 > 200
    motor_confused[mask_d] = motor_confused[mask_c][-1] * np.exp(-(time2[mask_d] - 200) / 40)
    motor_confused += np.random.normal(0, 0.02, len(time2))

    ax.plot(time2, sensory_fake, color=KEEPER_BLUE, lw=2.5, label='Sensory: initially "right"\nthen switches to "left"')
    ax.plot(time2, motor_confused, color=STRIKER_RED, lw=2.5, label='Motor: partially committed\nto "right" (stuck)')
    ax.axvline(x=0, color=MUTED, linestyle=':', alpha=0.5)
    ax.axvline(x=150, color=WARN_ORANGE, linestyle='--', alpha=0.7, label='Direction change')

    ax.annotate('Motor already\nloading "right"', xy=(130, 0.35), fontsize=7,
                ha='center', color=STRIKER_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=STRIKER_RED),
                xytext=(130, 0.55))

    ax.set_xlabel('Time from ball strike (ms)', fontweight='bold')
    ax.set_ylabel('Subspace Activation', fontweight='bold')
    ax.set_title('B. Stutter-Step Exploit:\nMid-Transformation Deception', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(-200, 500)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_transformation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig3_transformation.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated: fig3_transformation")


def fig4_gain_modulation():
    """Figure 4: Gain modulation and the expertise blind spot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: CPI by expertise
    ax = axes[0]
    expertise = ['Novice', 'Intermediate', 'Expert']
    hip_gain = [1.0, 1.5, 2.2]
    gaze_gain = [1.0, 0.7, 0.3]

    x = np.arange(len(expertise))
    w = 0.3
    ax.bar(x - w/2, hip_gain, w, label='Hip Kinematics (task-relevant)',
           color=KEEPER_BLUE, alpha=0.7, edgecolor='white')
    ax.bar(x + w/2, gaze_gain, w, label='Gaze Direction (suppressed)',
           color=MUTED, alpha=0.5, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(expertise)
    ax.set_ylabel('Relative Neural Gain', fontweight='bold')
    ax.set_title('A. Predicted Gain Modulation\nby Expertise Level', fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 2.8)

    # Panel B: Susceptibility to gaze deception
    ax = axes[1]
    # Counter-intuitive prediction: experts MORE susceptible to gaze fakes
    gaze_deception_rate = [0.55, 0.62, 0.72]  # fraction fooled by gaze fake
    gaze_err = [0.06, 0.05, 0.05]

    bars = ax.bar(expertise, gaze_deception_rate, yerr=gaze_err, capsize=5,
                  color=[FIELD_GREEN, WARN_ORANGE, STRIKER_RED], alpha=0.7,
                  edgecolor='white', linewidth=1)
    ax.axhline(y=0.5, color=MUTED, linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylabel('Susceptibility to\nGaze Deception', fontweight='bold')
    ax.set_title('B. Predicted: Expertise\nCreates Blind Spot', fontweight='bold')
    ax.set_ylim(0.3, 0.9)
    ax.legend(fontsize=7)

    # Add annotation
    ax.annotate('Paradoxical\nexpertise\ncost', xy=(2, 0.72), xytext=(1.5, 0.82),
                fontsize=8, color=STRIKER_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=STRIKER_RED, lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_gain_modulation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig4_gain_modulation.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated: fig4_gain_modulation")


def fig5_precommit_advantage():
    """Figure 5: Pre-commitment as interference reduction."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Simulate motor programme purity
    np.random.seed(42)
    n = 100

    # Pre-committed: single clean motor programme
    precommit_x = np.random.normal(0.8, 0.08, n)
    precommit_y = np.random.normal(0.85, 0.07, n)

    # Reactive: contaminated by competing programme
    reactive_x = np.random.normal(0.55, 0.18, n)
    reactive_y = np.random.normal(0.60, 0.16, n)

    ax.scatter(reactive_x, reactive_y, c=STRIKER_RED, alpha=0.4, s=25, label='Reactive (reads keeper)')
    ax.scatter(precommit_x, precommit_y, c=FIELD_GREEN, alpha=0.4, s=25, label='Pre-committed')

    # Ellipses for spread
    from matplotlib.patches import Ellipse
    ell1 = Ellipse((0.55, 0.60), 0.36, 0.32, angle=0, facecolor=STRIKER_RED, alpha=0.1, edgecolor=STRIKER_RED, lw=2)
    ell2 = Ellipse((0.8, 0.85), 0.16, 0.14, angle=0, facecolor=FIELD_GREEN, alpha=0.1, edgecolor=FIELD_GREEN, lw=2)
    ax.add_patch(ell1)
    ax.add_patch(ell2)

    ax.set_xlabel('Shot Placement Accuracy', fontweight='bold', fontsize=11)
    ax.set_ylabel('Shot Velocity (normalised)', fontweight='bold', fontsize=11)
    ax.set_title('Pre-Commitment Reduces Motor Interference\n(Simulated Shot Quality Distribution)', fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)

    # Annotations
    ax.annotate('Clean motor\nprogramme', xy=(0.8, 0.85), xytext=(0.45, 0.95),
                fontsize=9, color=FIELD_GREEN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=FIELD_GREEN, lw=1.5))
    ax.annotate('Shared-subspace\ninterference', xy=(0.55, 0.60), xytext=(0.15, 0.40),
                fontsize=9, color=STRIKER_RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=STRIKER_RED, lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_precommit.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'fig5_precommit.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Generated: fig5_precommit")


def try_imagen_generation():
    """
    Attempt to generate photorealistic images using Google's Imagen API.
    Requires GOOGLE_API_KEY environment variable.
    Falls back gracefully if unavailable.
    """
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("\nNote: GOOGLE_API_KEY not set. Skipping Imagen photorealistic generation.")
        print("To generate photorealistic images, set: export GOOGLE_API_KEY=your_key")
        print("Then re-run this script.\n")
        return

    try:
        import requests
        import base64
        from PIL import Image
        from io import BytesIO
    except ImportError:
        print("Note: requests and/or Pillow not installed. Skipping Imagen generation.")
        return

    prompts = [
        {
            "filename": "photo_keeper_dive.png",
            "prompt": (
                "Professional soccer goalkeeper mid-dive during a penalty kick, "
                "dramatic stadium lighting, photorealistic. "
                "Overlaid semi-transparent colored arrows showing two competing motor pathways: "
                "a blue arrow pointing left (the committed dive direction) and "
                "a red dashed arrow pointing right (the suppressed alternative), "
                "illustrating shared motor subspace interference. "
                "The blue pathway is bold and solid, the red is faded and broken, "
                "showing the brain's gain modulation suppressing the wrong response. "
                "Sports photography style, shallow depth of field, motion blur on gloves."
            )
        },
        {
            "filename": "photo_striker_approach.png",
            "prompt": (
                "Professional soccer player approaching a penalty kick from behind, "
                "photorealistic, stadium lighting. "
                "Overlaid transparent neural pathway diagram showing: "
                "sensory input (hip angle, gaze direction) as glowing blue nodes, "
                "flowing into a central 'task belief' node in purple, "
                "which selects between two motor output pathways (green arrows): "
                "'power shot' and 'placement shot' that share a common motor subspace "
                "(highlighted in orange where they overlap). "
                "The diagram is subtle and artistic, like a scientific figure overlay "
                "on a sports photograph. Clean infographic style."
            )
        },
        {
            "filename": "photo_transformation_window.png",
            "prompt": (
                "Split-screen photorealistic image of a penalty kick moment. "
                "Left side: goalkeeper's face and upper body, frozen in the moment "
                "of reading the striker's body cues, eyes focused, muscles tensing. "
                "Right side: the same goalkeeper 63 milliseconds later, body beginning to commit "
                "to a dive direction. Between the two frames, a glowing gradient bar "
                "showing the sensory-to-motor transformation window in orange-yellow. "
                "Text overlay: '63ms transformation gap'. "
                "Dramatic lighting, high contrast, sports documentary photography style."
            )
        }
    ]

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={api_key}"

    for item in prompts:
        print(f"Generating: {item['filename']}...")
        try:
            payload = {
                "instances": [{"prompt": item["prompt"]}],
                "parameters": {
                    "sampleCount": 1,
                    "aspectRatio": "16:9",
                    "safetyFilterLevel": "BLOCK_ONLY_HIGH"
                }
            }
            response = requests.post(endpoint, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                if 'predictions' in result and len(result['predictions']) > 0:
                    img_bytes = base64.b64decode(result['predictions'][0]['bytesBase64Encoded'])
                    img = Image.open(BytesIO(img_bytes))
                    filepath = os.path.join(FIG_DIR, item['filename'])
                    img.save(filepath)
                    print(f"  Saved: {filepath}")
                else:
                    print(f"  No image returned for {item['filename']}")
            else:
                print(f"  API error ({response.status_code}): {response.text[:200]}")
        except Exception as e:
            print(f"  Error generating {item['filename']}: {e}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating figures for: The Subspace Penalty Kick")
    print("=" * 50)

    fig1_task_mapping()
    fig2_interference_predictions()
    fig3_transformation_window()
    fig4_gain_modulation()
    fig5_precommit_advantage()

    print("\nAll matplotlib figures generated in:", FIG_DIR)
    print()

    # Attempt photorealistic generation
    try_imagen_generation()

    print("\nDone. Run compile.sh to build the PDF.")
