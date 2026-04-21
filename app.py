import io
import base64
from flask import Flask, request, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import math

# ============================================================
# DEINE ORIGINAL-FUNKTIONEN
# ============================================================

BASE_FERTILITY_PROFILE: Dict[int, float] = {
    -6: 0.02,
    -5: 0.10,
    -4: 0.16,
    -3: 0.14,
    -2: 0.27,
    -1: 0.31,
     0: 0.33,
     1: 0.08,
     2: 0.02,
}

SIGMA_MIN: float = 0.3
SIGMA_MAX: float = 2.5
CYCLES_PER_YEAR: int = 13
RISK_HIGH_THRESHOLD: float = 0.20
RISK_MEDIUM_THRESHOLD: float = 0.08
RISK_LOW_THRESHOLD: float = 0.01

@dataclass
class CycleParameters:
    cycle_length: int
    menstruation_start: int
    menstruation_end: int
    ovulation_day: int
    variability_factor: float

@dataclass
class DayResult:
    day: int
    day_offset: int
    phase: str
    probability: float
    risk_level: str

@dataclass
class PearlIndexResult:
    assumption: str
    cycles_per_year: int
    intercourse_days_count: int
    p_pregnancy_per_cycle: float
    p_no_pregnancy_per_cycle: float
    p_pregnancy_annual: float
    pearl_index: float
    p_fertile_window_only: float

def _classify_phase(day: int, params: CycleParameters) -> str:
    ov = params.ovulation_day
    if params.menstruation_start <= day <= params.menstruation_end:
        return "Menstruation"
    if day < ov - 5:
        return "Follicular (pre-fertile)"
    if ov - 5 <= day < ov:
        return "Fertile Window (pre-ovulation)"
    if day == ov:
        return "Ovulation"
    if ov < day <= ov + 2:
        return "Post-Ovulation (early luteal)"
    return "Luteal Phase"

def _classify_risk(probability: float) -> str:
    if probability >= RISK_HIGH_THRESHOLD:
        return "HIGH"
    if probability >= RISK_MEDIUM_THRESHOLD:
        return "MEDIUM"
    if probability >= RISK_LOW_THRESHOLD:
        return "LOW"
    return "MINIMAL"

def _validate_parameters(params: CycleParameters) -> None:
    if not (20 <= params.cycle_length <= 45):
        raise ValueError(f"cycle_length={params.cycle_length} is outside [20, 45].")
    if params.menstruation_start < 1:
        raise ValueError("menstruation_start must be ≥ 1.")
    if params.menstruation_end >= params.ovulation_day:
        raise ValueError(f"menstruation_end must be before ovulation_day")
    if params.ovulation_day >= params.cycle_length - 1:
        raise ValueError("ovulation_day must leave at least 2 days of luteal phase")
    if not (0.0 <= params.variability_factor <= 3.0):
        raise ValueError(f"variability_factor={params.variability_factor} must be in [0.0, 3.0].")

def _build_gaussian_kernel(sigma: float) -> np.ndarray:
    radius = int(math.ceil(4 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel

def build_fertility_curve(params: CycleParameters, base_profile: Dict[int, float] = None) -> np.ndarray:
    if base_profile is None:
        base_profile = BASE_FERTILITY_PROFILE
    n = params.cycle_length
    ov_idx = params.ovulation_day - 1
    raw = np.zeros(n)
    for offset, prob in base_profile.items():
        idx = ov_idx + offset
        if 0 <= idx < n:
            raw[idx] = prob
    sigma = SIGMA_MIN + (params.variability_factor / 3.0) * (SIGMA_MAX - SIGMA_MIN)
    kernel = _build_gaussian_kernel(sigma)
    smoothed = np.convolve(raw, kernel, mode='same')
    original_total = sum(base_profile.values())
    smoothed_total = smoothed.sum()
    if smoothed_total > 0:
        smoothed *= original_total / smoothed_total
    peak_cap = max(base_profile.values()) * 1.10
    smoothed = np.clip(smoothed, 0.0, peak_cap)
    return smoothed

def calculate_cycle_probabilities(params: CycleParameters, base_profile: Dict[int, float] = None) -> List[DayResult]:
    _validate_parameters(params)
    curve = build_fertility_curve(params, base_profile)
    results: List[DayResult] = []
    for i in range(params.cycle_length):
        day = i + 1
        prob = float(curve[i])
        phase = _classify_phase(day, params)
        risk = _classify_risk(prob)
        offset = day - params.ovulation_day
        results.append(DayResult(day=day, day_offset=offset, phase=phase, probability=prob, risk_level=risk))
    return results

def calculate_pearl_index(results: List[DayResult], params: CycleParameters, intercourse_days: Optional[List[int]] = None, cycles_per_year: int = CYCLES_PER_YEAR) -> PearlIndexResult:
    probs_array = np.array([r.probability for r in results])
    if intercourse_days is None:
        intercourse_days = list(range(1, params.cycle_length + 1))
        assumption = "daily intercourse (every day of the cycle)"
    else:
        assumption = f"intercourse on specified days: {intercourse_days}"
    intercourse_idx = [d - 1 for d in intercourse_days if 1 <= d <= params.cycle_length]
    selected_probs = probs_array[intercourse_idx]
    p_no_pregnancy = float(np.prod(1.0 - selected_probs))
    p_pregnancy = 1.0 - p_no_pregnancy
    p_annual = 1.0 - (1.0 - p_pregnancy) ** cycles_per_year
    pearl_index = p_pregnancy * cycles_per_year * 100.0
    ov_idx = params.ovulation_day - 1
    fertile_idx = [ov_idx + k for k in range(-5, 2) if 0 <= ov_idx + k < params.cycle_length]
    fertile_probs = probs_array[fertile_idx]
    p_fertile_only = 1.0 - float(np.prod(1.0 - fertile_probs))
    return PearlIndexResult(
        assumption=assumption,
        cycles_per_year=cycles_per_year,
        intercourse_days_count=len(intercourse_idx),
        p_pregnancy_per_cycle=p_pregnancy,
        p_no_pregnancy_per_cycle=p_no_pregnancy,
        p_pregnancy_annual=p_annual,
        pearl_index=pearl_index,
        p_fertile_window_only=p_fertile_only,
    )

# ============================================================
# FLASK WEB-APP
# ============================================================

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Wehe du lässt mich an den sicheren Tagen rausziehen</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f2f5;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #e74c3c;
            margin-top: 0;
            font-size: 1.8rem;
            text-align: center;
        }
        .love-note {
            background: #ffe8f0;
            border-left: 4px solid #e74c3c;
            padding: 12px;
            margin: 20px 0;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: #c0392b;
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 6px;
            color: #333;
        }
        input, select {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover {
            background: #c0392b;
        }
        .result {
            margin-top: 30px;
            border-top: 2px solid #eee;
            padding-top: 20px;
        }
        .stats {
            background: #f8f9fa;
            padding: 16px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        }
        .stats h3 {
            margin: 0 0 8px 0;
            color: #e74c3c;
            font-size: 2rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        th {
            background: #f2f2f2;
            position: sticky;
            top: 0;
        }
        .HIGH { background-color: #ffdddd; }
        .MEDIUM { background-color: #fff3cd; }
        .LOW { background-color: #d4edda; }
        .MINIMAL { background-color: #e2e3e5; }
        .table-container {
            overflow-x: auto;
            max-height: 400px;
        }
        img {
            max-width: 100%;
            height: auto;
            margin-top: 20px;
            border-radius: 8px;
        }
        footer {
            text-align: center;
            margin-top: 30px;
            font-size: 12px;
            color: #999;
        }
        .single-day-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        }
        .probability-big {
            font-size: 56px;
            font-weight: bold;
            color: #e74c3c;
            margin: 15px 0;
        }
        @media (max-width: 600px) {
            .container { padding: 16px; }
            th, td { font-size: 11px; padding: 4px; }
            .probability-big { font-size: 40px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>💢 Wehe du lässt mich rausziehen rechner 💢</h1>
        
        <div class="love-note">
            ❤️ PS: Ich liebe dich über alles! ❤️
        </div>
        
        <form method="POST">
            <div class="form-group">
                <label>Zykluslänge (Tage, 20-45):</label>
                <input type="number" name="cycle_length" value="28" min="20" max="45" required>
            </div>
            <div class="form-group">
                <label>Menstruation beginnt (Tag 1 = erster Blutungstag):</label>
                <input type="number" name="menstruation_start" value="1" min="1" required>
            </div>
            <div class="form-group">
                <label>Menstruation endet (Tag):</label>
                <input type="number" name="menstruation_end" value="5" required>
            </div>
            <div class="form-group">
                <label>Eisprung (Tag, üblich: Zykluslänge - 14):</label>
                <input type="number" name="ovulation_day" value="14" required>
            </div>
            <div class="form-group">
                <label>Zyklus-Variabilität (0 = sehr regelmäßig, 3 = sehr unregelmäßig):</label>
                <input type="number" step="0.5" name="variability_factor" value="1.0" min="0" max="3" required>
            </div>
            <div class="form-group">
                <label>📅 Wann Creampie 😏</label>
                <input type="number" name="intercourse_day" value="" placeholder="z.B. 14 - leer lassen für Tabelle">
                <small>Tag 1 = erster Tag der Periode</small>
            </div>
            <button type="submit">🔥 Berechnen 🔥</button>
        </form>

        {% if result %}
        <div class="result">
            
            {% if result.single_day_percent is not none %}
            <div class="single-day-box">
                <h2>🎯 Ergebnis für Creampie an Tag {{ result.single_day }}</h2>
                <div class="probability-big">
                    {{ "%.1f"|format(result.single_day_percent) }}%
                </div>
                <p style="font-size: 18px;">
                    Wahrscheinlichkeit, dass <strong>dieser eine Akt</strong> zu einer Schwangerschaft führt
                </p>
                
                {% if result.single_day_percent > 20 %}
                    <p style="color: #e74c3c; font-size: 18px;">🔴 <strong>HOCHES Risiko</strong> - Du lässt mich besser rausziehen!!</p>
                {% elif result.single_day_percent > 8 %}
                    <p style="color: #e67e22; font-size: 18px;">🟡 <strong>MITTELES Risiko</strong> - Lieber rausziehen :d</p>
                {% elif result.single_day_percent > 1 %}
                    <p style="color: #27ae60; font-size: 18px;">🟢 <strong>LOW Risk!</strong> - No Risk no Fun oder...</p>
                {% else %}
                    <p style="color: #7f8c8d; font-size: 18px;">⚪ <strong>NO Risk!</strong> - Beine verschränken!! 😏</p>
                {% endif %}
            </div>
            {% else %}
            <div class="stats">
                <h3>📊 Gib einen Tag oben ein!</h3>
                <p>Trage oben einen Zyklustag ein, um die Wahrscheinlichkeit für einen bestimmten Tag zu berechnen.</p>
                <p>Die Tabelle unten zeigt alle Tage mit ihren Wahrscheinlichkeiten.</p>
            </div>
            {% endif %}

            <!-- Die fruchtbarsten Tage -->
            <div style="margin-top: 20px; padding: 12px; background: #f8f9fa; border-radius: 8px;">
                <h4>📅 Die gefährlichsten Tage (am besten lässt du mich hier rausziehen :p):</h4>
                <ul>
                    {% for day in result.table %}
                        {% if day.probability * 100 > 15 %}
                            <li><strong>Tag {{ day.day }}</strong> ({{ day.phase }}) → {{ "%.1f"|format(day.probability * 100) }}% Wahrscheinlichkeit</li>
                        {% endif %}
                    {% endfor %}
                </ul>
            </div>
            
            <h3>📅 Alle Tage im Überblick</h3>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Tag</th><th>Offset</th><th>Wahrscheinlichkeit</th><th>Risiko</th><th>Phase</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for day in result.table %}
                        <tr class="{{ day.risk_level }}">
                            <td>{{ day.day }}</td>
                            <td>{{ day.offset }}</td>
                            <td>{{ "%.2f"|format(day.probability*100) }}%</td>
                            <td>{{ day.risk_level }}</td>
                            <td>{{ day.phase }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <h3>📊 Grafische Darstellung</h3>
            <img src="data:image/png;base64,{{ result.plot_url }}" alt="Wahrscheinlichkeitsgrafik">
        </div>
        {% endif %}
        
        <footer>
            Made with ❤️ and HUGE BONER by Laris — Basierend auf Wilcox et al. (1995) NEJM — Für die Frau welche ich über alles liebe ❤️
        </footer>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            params = CycleParameters(
                cycle_length=int(request.form["cycle_length"]),
                menstruation_start=int(request.form["menstruation_start"]),
                menstruation_end=int(request.form["menstruation_end"]),
                ovulation_day=int(request.form["ovulation_day"]),
                variability_factor=float(request.form["variability_factor"])
            )
            
            results = calculate_cycle_probabilities(params)
            
            # ──────────────────────────────────────────────────────────
            # Einzelner Tag für Creampie-Berechnung
            # ──────────────────────────────────────────────────────────
            intercourse_day = request.form.get("intercourse_day")
            single_day_percent = None
            single_day_value = None
            
            if intercourse_day and intercourse_day.strip():
                try:
                    day = int(intercourse_day)
                    if 1 <= day <= params.cycle_length:
                        day_result = next((r for r in results if r.day == day), None)
                        if day_result:
                            single_day_percent = day_result.probability * 100
                            single_day_value = day
                except ValueError:
                    pass
            
            # Tabelle vorbereiten
            table_data = []
            for r in results:
                offset_str = f"Ov{r.day_offset:+d}" if r.day_offset != 0 else "Ov"
                table_data.append({
                    "day": r.day,
                    "offset": offset_str,
                    "probability": r.probability,
                    "risk_level": r.risk_level,
                    "phase": r.phase
                })
            
            # Grafik generieren
            fig, ax = plt.subplots(figsize=(12, 5))
            days = [r.day for r in results]
            probs = [r.probability * 100 for r in results]
            
            risk_colors = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#27ae60", "MINIMAL": "#95a5a6"}
            colors = [risk_colors[r.risk_level] for r in results]
            
            ax.bar(days, probs, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
            
            # Werte über die Balken schreiben (nur wenn >5%)
            for bar, prob in zip(ax.patches, probs):
                if prob >= 5:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{prob:.0f}%", 
                           ha="center", va="bottom", fontsize=8)
            
            ax.axvline(params.ovulation_day, color="#8e44ad", linestyle="--", linewidth=2, 
                      label=f"Eisprung (Tag {params.ovulation_day})")
            ax.axvspan(params.menstruation_start - 0.5, params.menstruation_end + 0.5, 
                      alpha=0.15, color="#e74c3c", label="Menstruation")
            
            ax.set_xlabel("Zyklustag", fontsize=11)
            ax.set_ylabel("Schwangerschaftswahrscheinlichkeit (%)", fontsize=11)
            ax.set_title(f"Tägliche Wahrscheinlichkeit — {params.cycle_length}-Tag Zyklus", fontsize=12)
            ax.set_xlim(0.5, params.cycle_length + 0.5)
            ax.set_ylim(0, max(probs) * 1.2 if max(probs) > 0 else 5)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches="tight")
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)
            
            return render_template_string(HTML_TEMPLATE, result={
                "table": table_data,
                "plot_url": plot_url,
                "single_day_percent": single_day_percent,
                "single_day": single_day_value
            })
        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=str(e))
    
    return render_template_string(HTML_TEMPLATE, result=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)