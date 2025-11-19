"""Acoustic drum synthesis engine — physically-informed modal modeling.

This module replaces simplified subtractive synthesis (sine-sweep + noise-burst)
with a multi-component modal architecture grounded in circular-membrane physics:

  * 20-30 membrane modes derived from Bessel-function zeros
  * Air-cavity resonance (low-frequency sub-harmonic reinforcement)
  * Shell/body resonance (mid-range woody coloration)
  * Impact transient (critical first 5-10 ms of the attack)
  * Snare-wire buzz model (filtered noise + chaotic flutter)
  * Velocity-dependent timbre (not just volume)
  * Per-mode micro-randomisation for round-robin variation
  * Soft saturation and nonlinear elements

Public API
----------
render_instrument(recipe, *, sample_rate, duration, velocity,
                  round_robin_index, params) -> List[float]

Compatible with the same JSON config / generate_drums.py pipeline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

TAU = 2 * math.pi
Signal = List[float]


# ============================================================================
# Public entry point
# ============================================================================

def render_instrument(
    recipe: str,
    *,
    sample_rate: int,
    duration: float,
    velocity: int,
    round_robin_index: int,
    params: Dict[str, Any] | None = None,
) -> Signal:
    """Render a single drum hit and return floating-point PCM samples."""
    factory = _RECIPES.get(recipe)
    if factory is None:
        raise ValueError(f"Unknown recipe '{recipe}'")
    seed = hash((recipe, velocity, round_robin_index)) & 0xFFFFFFFF
    samples = factory(
        sample_rate=sample_rate,
        duration=duration,
        velocity=velocity,
        rr_index=round_robin_index,
        params=params or {},
        seed=seed,
    )
    return samples


# ============================================================================
# Bessel function zeros and membrane mode table
# ============================================================================

# Zeros of J_m(x), m = 0..5, first 5 positive roots each.
# f_{m,n} = z_{m,n} * c / (2*pi*a)  for a circular membrane.
_BESSEL_ZEROS: Dict[int, List[float]] = {
    0: [2.4048, 5.5201, 8.6537, 11.7915, 14.9309],
    1: [3.8317, 7.0156, 10.1735, 13.3237, 16.4706],
    2: [5.1356, 8.4172, 11.6198, 14.7960, 17.9598],
    3: [6.3802, 9.7610, 13.0152, 16.2235, 19.4094],
    4: [7.5884, 10.7143, 13.8534, 17.0080, 20.1229],
    5: [8.7715, 12.3386, 15.7002, 18.9801, 22.2178],
}

_ALPHA_01 = _BESSEL_ZEROS[0][0]  # 2.4048 — fundamental

# Build sorted mode table: (ratio, angular_order, radial_order)
_MEMBRANE_MODES: List[Tuple[float, int, int]] = []
for _m, _zeros in _BESSEL_ZEROS.items():
    for _ni, _z in enumerate(_zeros):
        _MEMBRANE_MODES.append((_z / _ALPHA_01, _m, _ni + 1))
_MEMBRANE_MODES.sort(key=lambda x: x[0])
# Result: 30 modes from ratio 1.000 up to ~9.24


# ============================================================================
# DSP utility helpers
# ============================================================================

def _vel_gain(velocity: int, curve: float = 1.0, floor: float = 0.05) -> float:
    return max(floor, (velocity / 127.0) ** curve)


def _normalize(samples: Signal, target: float = 0.96) -> Signal:
    peak = max((abs(s) for s in samples), default=0.0)
    if peak < 1e-12:
        return samples
    return [s * (target / peak) for s in samples]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _tanh_softclip(samples: Signal, drive: float) -> Signal:
    """Soft saturation via tanh with drive compensation."""
    if drive <= 1.0:
        return samples
    inv = 1.0 / math.tanh(drive)
    return [math.tanh(s * drive) * inv for s in samples]


def _apply_fade(samples: Signal, sr: int, fade_s: float) -> None:
    n = min(len(samples), int(sr * fade_s))
    if n <= 0:
        return
    for i in range(1, n + 1):
        samples[-i] *= i / n


def _dc_block(samples: Signal, coeff: float = 0.995) -> Signal:
    """Single-pole DC blocker."""
    if not samples:
        return samples
    out: Signal = []
    xprev = 0.0
    yprev = 0.0
    for x in samples:
        y = x - xprev + coeff * yprev
        xprev = x
        yprev = y
        out.append(y)
    return out


# ---- Biquad filter --------------------------------------------------------

@dataclass
class _Biquad:
    b0: float; b1: float; b2: float
    a1: float; a2: float
    z1: float = 0.0; z2: float = 0.0

    def process(self, x: float) -> float:
        y = x * self.b0 + self.z1
        self.z1 = x * self.b1 + self.z2 - self.a1 * y
        self.z2 = x * self.b2 - self.a2 * y
        return y

    def reset(self) -> None:
        self.z1 = self.z2 = 0.0


def _bq(sr: int, mode: str, freq: float, q: float = 0.707,
        gain_db: float = 0.0) -> _Biquad:
    freq = max(1.0, min(sr * 0.45, freq))
    w = TAU * freq / sr
    sn, cs = math.sin(w), math.cos(w)
    alpha = sn / (2 * max(0.001, q))
    A = 10 ** (gain_db / 40)

    if mode == "lp":
        b0 = (1 - cs) / 2; b1 = 1 - cs; b2 = b0
        a0 = 1 + alpha; a1 = -2 * cs; a2 = 1 - alpha
    elif mode == "hp":
        b0 = (1 + cs) / 2; b1 = -(1 + cs); b2 = b0
        a0 = 1 + alpha; a1 = -2 * cs; a2 = 1 - alpha
    elif mode == "bp":
        b0 = sn / 2; b1 = 0.0; b2 = -sn / 2
        a0 = 1 + alpha; a1 = -2 * cs; a2 = 1 - alpha
    elif mode == "notch":
        b0 = 1.0; b1 = -2 * cs; b2 = 1.0
        a0 = 1 + alpha; a1 = -2 * cs; a2 = 1 - alpha
    elif mode == "peak":
        b0 = 1 + alpha * A; b1 = -2 * cs; b2 = 1 - alpha * A
        a0 = 1 + alpha / A; a1 = -2 * cs; a2 = 1 - alpha / A
    elif mode == "loshelf":
        sq = 2 * math.sqrt(max(A, 0.001)) * alpha
        b0 = A * ((A + 1) - (A - 1) * cs + sq)
        b1 = 2 * A * ((A - 1) - (A + 1) * cs)
        b2 = A * ((A + 1) - (A - 1) * cs - sq)
        a0 = (A + 1) + (A - 1) * cs + sq
        a1 = -2 * ((A - 1) + (A + 1) * cs)
        a2 = (A + 1) + (A - 1) * cs - sq
    elif mode == "hishelf":
        sq = 2 * math.sqrt(max(A, 0.001)) * alpha
        b0 = A * ((A + 1) + (A - 1) * cs + sq)
        b1 = -2 * A * ((A - 1) + (A + 1) * cs)
        b2 = A * ((A + 1) + (A - 1) * cs - sq)
        a0 = (A + 1) - (A - 1) * cs + sq
        a1 = 2 * ((A - 1) - (A + 1) * cs)
        a2 = (A + 1) - (A - 1) * cs - sq
    else:
        raise ValueError(f"Unknown filter mode '{mode}'")

    inv = 1.0 / a0
    return _Biquad(b0 * inv, b1 * inv, b2 * inv, a1 * inv, a2 * inv)


def _chain(samples: Signal, sr: int,
           stages: Sequence[Tuple[str, float, float, float]] | None) -> Signal:
    if not stages:
        return samples
    data = samples[:]
    filts = [_bq(sr, m, f, q, g) for m, f, q, g in stages]
    for i, s in enumerate(data):
        v = s
        for flt in filts:
            v = flt.process(v)
        data[i] = v
    return data


# ---- Noise generators -----------------------------------------------------

def _white(n: int, rng: random.Random) -> Signal:
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def _colored(n: int, rng: random.Random, color: float = 0.3) -> Signal:
    alpha = max(0.0, min(0.999, color))
    prev = 0.0
    out: Signal = []
    for _ in range(n):
        w = rng.uniform(-1.0, 1.0)
        prev = alpha * prev + (1 - alpha) * w
        out.append(prev)
    return out


def _bp_noise(n: int, sr: int, rng: random.Random,
              center: float, q: float = 2.0) -> Signal:
    flt = _bq(sr, "bp", center, q)
    return [flt.process(rng.uniform(-1.0, 1.0)) for _ in range(n)]


def _hp_noise(signal: Signal, coeff: float = 0.5) -> Signal:
    if not signal:
        return signal
    o: Signal = []
    prev = signal[0]
    for s in signal:
        o.append(s - prev)
        prev = s * (1 - coeff) + prev * coeff
    return o


# ---- Mix helpers -----------------------------------------------------------

def _mix_into(dest: Signal, src: Signal, level: float, offset: int = 0) -> None:
    """Add src * level into dest starting at offset, in place."""
    end = min(len(dest), offset + len(src))
    for i in range(offset, end):
        dest[i] += src[i - offset] * level


def _zeros(n: int) -> Signal:
    return [0.0] * n


# ============================================================================
# Modal synthesis core
# ============================================================================

@dataclass
class _Mode:
    """State for a single resonant mode."""
    freq: float            # Hz
    amp: float             # current amplitude weight
    decay_per_sample: float  # multiplicative decay per sample
    phase: float           # current phase in radians
    phase_inc: float       # radians per sample
    env: float = 1.0       # current envelope value

    # optional per-mode tremolo for chaotic decay
    trem_phase: float = 0.0
    trem_inc: float = 0.0
    trem_depth: float = 0.0


def _build_membrane_modes(
    fundamental: float,
    sr: int,
    velocity: int,
    rng: random.Random,
    *,
    num_modes: int = 26,
    decay_base: float = 6.0,
    decay_slope: float = 0.6,
    amp_falloff: float = 0.5,
    vel_timbre: float = 0.04,
    freq_jitter: float = 0.003,
    amp_jitter: float = 0.08,
    decay_jitter: float = 0.06,
    pitch_shift_vel: float = 0.003,
) -> List[_Mode]:
    """Create modal bank for a circular membrane.

    Parameters
    ----------
    fundamental : Hz of the (0,1) mode.
    num_modes   : how many Bessel modes to use (max 30).
    decay_base  : base exponential decay rate (1/s) for fundamental.
    decay_slope : how much faster upper modes decay (exponent on ratio).
    amp_falloff : controls how fast amplitudes decrease with angular order.
    vel_timbre  : velocity-dependent brightness scaling per mode index.
    freq_jitter, amp_jitter, decay_jitter : per-hit randomisation ranges.
    pitch_shift_vel : velocity shifts pitch upward by this fraction at v=127.
    """
    vel_norm = velocity / 127.0
    # Slight upward pitch shift under hard hits
    pitch_mult = 1.0 + pitch_shift_vel * vel_norm

    modes: List[_Mode] = []
    n = min(num_modes, len(_MEMBRANE_MODES))
    for idx in range(n):
        ratio, m_order, n_order = _MEMBRANE_MODES[idx]

        # Frequency with jitter and velocity pitch shift
        freq = fundamental * ratio * pitch_mult
        freq *= 1.0 + rng.uniform(-freq_jitter, freq_jitter)

        # Amplitude: decreases with angular and radial order
        # Velocity makes upper modes relatively louder
        base_amp = 1.0 / ((1.0 + amp_falloff * m_order) *
                          (1.0 + 0.3 * (n_order - 1)))
        vel_scale = vel_norm ** (1.0 + vel_timbre * idx)
        amp = base_amp * vel_scale
        amp *= 1.0 + rng.uniform(-amp_jitter, amp_jitter)

        # Decay: higher modes decay faster, with jitter
        mode_decay = decay_base * (ratio ** decay_slope)
        mode_decay *= 1.0 + rng.uniform(-decay_jitter, decay_jitter)
        decay_per_sample = math.exp(-mode_decay / sr)

        phase = rng.uniform(0, TAU)
        phase_inc = TAU * freq / sr

        # Per-mode tremolo for chaotic decay (subtle slow modulation)
        trem_rate = rng.uniform(2.0, 8.0)
        trem_depth = rng.uniform(0.01, 0.04)

        modes.append(_Mode(
            freq=freq, amp=amp,
            decay_per_sample=decay_per_sample,
            phase=phase, phase_inc=phase_inc,
            trem_phase=rng.uniform(0, TAU),
            trem_inc=TAU * trem_rate / sr,
            trem_depth=trem_depth,
        ))
    return modes


def _render_modes(modes: List[_Mode], total: int) -> Signal:
    """Render a bank of modes to a signal buffer."""
    out = _zeros(total)
    for mode in modes:
        ph = mode.phase
        inc = mode.phase_inc
        amp = mode.amp
        env = 1.0
        decay = mode.decay_per_sample
        t_ph = mode.trem_phase
        t_inc = mode.trem_inc
        t_dep = mode.trem_depth
        for i in range(total):
            trem = 1.0 + t_dep * math.sin(t_ph)
            out[i] += math.sin(ph) * amp * env * trem
            ph += inc
            t_ph += t_inc
            env *= decay
        mode.phase = ph
        mode.env = env
    return out


def _render_air_cavity(
    fundamental: float,
    sr: int,
    total: int,
    velocity: int,
    rng: random.Random,
    *,
    num_resonances: int = 3,
    q_range: Tuple[float, float] = (1.5, 4.0),
    decay_rate: float = 3.0,
    level: float = 0.3,
) -> Signal:
    """Low-frequency air-cavity resonances that reinforce the fundamental.

    The enclosed air volume creates Helmholtz-type resonances below and near
    the membrane fundamental, adding depth and 'boom'.
    """
    vel_norm = velocity / 127.0
    out = _zeros(total)
    resonances = []
    for k in range(num_resonances):
        # Air modes cluster near and below the fundamental
        freq = fundamental * (0.5 + 0.4 * k) * (1.0 + rng.uniform(-0.05, 0.05))
        q = rng.uniform(*q_range)
        amp = level * (1.0 / (1 + 0.5 * k)) * vel_norm
        phase = rng.uniform(0, TAU)
        decay = decay_rate * (1 + 0.3 * k)
        resonances.append((freq, amp, decay, phase))

    for freq, amp, decay, phase in resonances:
        ph = phase
        inc = TAU * freq / sr
        decay_ps = math.exp(-decay / sr)
        env = 1.0
        for i in range(total):
            out[i] += math.sin(ph) * amp * env
            ph += inc
            env *= decay_ps
    return out


def _render_shell(
    fundamental: float,
    sr: int,
    total: int,
    velocity: int,
    rng: random.Random,
    *,
    shell_freqs: Sequence[float] | None = None,
    q_range: Tuple[float, float] = (1.0, 3.0),
    decay_rate: float = 5.0,
    level: float = 0.15,
    material: str = "wood",
) -> Signal:
    """Shell/body resonances — broad mid-frequency coloration.

    Wood shells add warmth at 200-800 Hz; metal shells are brighter.
    """
    vel_norm = velocity / 127.0
    if shell_freqs is None:
        if material == "metal":
            shell_freqs = [
                fundamental * 3.2, fundamental * 5.1, fundamental * 7.8,
                fundamental * 11.0,
            ]
        else:  # wood
            shell_freqs = [
                fundamental * 2.4, fundamental * 3.8, fundamental * 5.5,
                fundamental * 7.2, fundamental * 9.0,
            ]

    # Use filtered noise resonated through bandpass filters for shell
    noise_src = _colored(total, rng, 0.5)
    out = _zeros(total)

    for k, sf in enumerate(shell_freqs):
        sf *= 1.0 + rng.uniform(-0.02, 0.02)
        q = rng.uniform(*q_range)
        flt = _bq(sr, "bp", sf, q)
        weight = level * (1.0 / (1 + 0.4 * k)) * vel_norm
        drate = decay_rate * (1 + 0.3 * k)
        dps = math.exp(-drate / sr)
        env = 1.0
        for i in range(total):
            out[i] += flt.process(noise_src[i]) * weight * env
            env *= dps
    return out


def _render_impact(
    sr: int,
    total: int,
    velocity: int,
    rng: random.Random,
    *,
    brightness: float = 0.6,
    click_freq: float = 3000.0,
    click_level: float = 0.35,
    noise_level: float = 0.45,
    duration_ms: float = 6.0,
) -> Signal:
    """Stick/beater impact transient — the critical first few milliseconds.

    Three components:
      1. Sub-millisecond broadband click
      2. High-passed noise burst (3-8 ms)
      3. Brief tonal 'stick resonance'
    """
    vel_norm = velocity / 127.0
    out = _zeros(total)

    # 1) Broadband click (< 1 ms)
    click_samples = max(1, int(sr * 0.0008))
    for i in range(min(click_samples, total)):
        env = 1.0 - i / click_samples
        out[i] += rng.uniform(-1.0, 1.0) * env * click_level * vel_norm

    # 2) HP noise burst
    attack_dur = duration_ms / 1000.0 * (0.7 + 0.6 * brightness * vel_norm)
    burst_n = min(total, max(1, int(sr * attack_dur)))
    hp_freq = 2000 + 6000 * brightness * vel_norm
    flt = _bq(sr, "hp", hp_freq, 0.7)
    decay_rate = 1.0 / max(0.0001, attack_dur)
    for i in range(burst_n):
        t = i / sr
        env = math.exp(-t * decay_rate * 4)
        out[i] += flt.process(rng.uniform(-1.0, 1.0)) * env * noise_level * vel_norm

    # 3) Brief tonal click resonance (stick material)
    click_freq_actual = click_freq * (1.0 + rng.uniform(-0.1, 0.1))
    click_decay = 300.0
    ph = rng.uniform(0, TAU)
    inc = TAU * click_freq_actual / sr
    tone_n = min(total, int(sr * 0.004))
    for i in range(tone_n):
        t = i / sr
        out[i] += math.sin(ph) * math.exp(-t * click_decay) * 0.2 * vel_norm
        ph += inc

    return out


def _render_snare_wires(
    membrane_signal: Signal,
    sr: int,
    velocity: int,
    rng: random.Random,
    *,
    wire_freq_lo: float = 2000.0,
    wire_freq_hi: float = 8000.0,
    wire_level: float = 0.55,
    wire_decay: float = 12.0,
    buzz_amount: float = 0.15,
) -> Signal:
    """Snare-wire noise model.

    The wires are excited by membrane velocity and produce a filtered-noise
    signal with chaotic flutter. A small feedback-delay network creates the
    characteristic 'buzz'.
    """
    total = len(membrane_signal)
    vel_norm = velocity / 127.0
    out = _zeros(total)

    # Bandpass filter for wire frequency range
    bp1 = _bq(sr, "bp", (wire_freq_lo + wire_freq_hi) / 2,
              q=(wire_freq_hi - wire_freq_lo) / wire_freq_lo)
    bp2 = _bq(sr, "bp", wire_freq_hi * 0.8, q=3.0)

    # Envelope follower on membrane signal (simple abs + smooth)
    env_smooth = 0.0
    env_coeff = math.exp(-50.0 / sr)

    # Small delay lines for buzz (feedback delay network)
    dly_len1 = max(1, int(sr * rng.uniform(0.0001, 0.0008)))
    dly_len2 = max(1, int(sr * rng.uniform(0.0002, 0.001)))
    dly1 = [0.0] * dly_len1
    dly2 = [0.0] * dly_len2
    d1i = 0
    d2i = 0
    feedback = rng.uniform(0.1, 0.25) * buzz_amount / 0.15

    # Chaotic modulation LFO
    lfo_phase = rng.uniform(0, TAU)
    lfo_inc = TAU * rng.uniform(5.0, 15.0) / sr

    decay_ps = math.exp(-wire_decay / sr)
    env_val = 1.0

    for i in range(total):
        # Envelope follower
        env_smooth = env_coeff * env_smooth + (1 - env_coeff) * abs(membrane_signal[i])

        # Noise excited by envelope
        noise = rng.uniform(-1.0, 1.0) * env_smooth

        # Chaotic modulation
        lfo = 1.0 + 0.05 * math.sin(lfo_phase)
        lfo_phase += lfo_inc

        # Filter
        filtered = bp1.process(noise) + 0.3 * bp2.process(noise)

        # Feedback delay = buzz
        buzz = dly1[d1i] * feedback + dly2[d2i] * feedback * 0.6
        dly1[d1i] = filtered
        dly2[d2i] = filtered * 0.8 + buzz * 0.2
        d1i = (d1i + 1) % dly_len1
        d2i = (d2i + 1) % dly_len2

        sample = (filtered + buzz) * lfo * env_val * wire_level * vel_norm
        out[i] = sample
        env_val *= decay_ps

    return out


# ============================================================================
# Cymbal / hi-hat inharmonic mode generator
# ============================================================================

def _build_cymbal_modes(
    base_freq: float,
    sr: int,
    velocity: int,
    rng: random.Random,
    *,
    num_modes: int = 70,
    freq_power: float = 1.18,
    freq_spread: float = 0.04,
    decay_base: float = 2.5,
    decay_freq_power: float = 0.3,
    vel_timbre: float = 0.02,
    amp_jitter: float = 0.15,
) -> List[_Mode]:
    """Build inharmonic modal bank for cymbals and metallic percussion.

    Cymbal modes follow approximately f_i ~ f0 * i^power with significant
    irregularity. Higher modes decay faster. Many closely-spaced modes
    create natural beating patterns.
    """
    vel_norm = velocity / 127.0
    modes: List[_Mode] = []

    for idx in range(num_modes):
        i = idx + 1
        # Inharmonic frequency distribution
        freq = base_freq * (i ** freq_power)
        freq *= 1.0 + rng.uniform(-freq_spread, freq_spread)

        # Amplitude decreases with frequency, velocity brightens upper modes
        base_amp = 1.0 / (i ** 0.55)
        vel_scale = vel_norm ** (1.0 + vel_timbre * idx)
        amp = base_amp * vel_scale
        amp *= 1.0 + rng.uniform(-amp_jitter, amp_jitter)
        amp = max(0.0, amp)

        # Frequency-dependent decay — high modes ring shorter
        freq_ratio = freq / base_freq
        mode_decay = decay_base * (freq_ratio ** decay_freq_power)
        decay_per_sample = math.exp(-mode_decay / sr)

        phase = rng.uniform(0, TAU)
        phase_inc = TAU * freq / sr

        trem_rate = rng.uniform(1.0, 5.0)
        trem_depth = rng.uniform(0.005, 0.02)

        modes.append(_Mode(
            freq=freq, amp=amp,
            decay_per_sample=decay_per_sample,
            phase=phase, phase_inc=phase_inc,
            trem_phase=rng.uniform(0, TAU),
            trem_inc=TAU * trem_rate / sr,
            trem_depth=trem_depth,
        ))
    return modes


# ============================================================================
# Drum engines
# ============================================================================

# ---- Kick ----------------------------------------------------------------

def _kick_engine(profile: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    cfg = {
        "fundamental": 52.0,
        "pitch_start_ratio": 2.8,
        "pitch_sweep_rate": 28.0,
        "num_modes": 22,
        "membrane_decay": 5.5,
        "membrane_decay_slope": 0.55,
        "air_level": 0.35,
        "air_decay": 2.8,
        "shell_level": 0.10,
        "shell_decay": 6.0,
        "click_level": 0.30,
        "noise_level": 0.18,
        "attack_brightness": 0.4,
        "sub_freq": 38.0,
        "sub_level": 0.45,
        "sub_decay": 3.2,
        "drive": 1.35,
    }
    if profile == "tight":
        cfg.update({
            "fundamental": 48.0,
            "pitch_start_ratio": 3.2,
            "pitch_sweep_rate": 35.0,
            "membrane_decay": 7.0,
            "air_level": 0.25,
            "sub_level": 0.55,
            "sub_decay": 3.8,
            "drive": 1.5,
        })
    cfg.update(kw.get("params", {}))

    f0 = cfg["fundamental"]

    # --- Membrane modes with pitch sweep ---
    # Kick drums have a characteristic pitch sweep on the fundamental.
    # We handle this by rendering the first few modes with time-varying frequency.
    modes = _build_membrane_modes(
        f0, sr, vel, rng,
        num_modes=cfg["num_modes"],
        decay_base=cfg["membrane_decay"],
        decay_slope=cfg["membrane_decay_slope"],
        amp_falloff=0.6,
        vel_timbre=0.03,
    )

    # Apply pitch sweep to all modes proportionally
    membrane = _zeros(total)
    sweep_rate = cfg["pitch_sweep_rate"]
    start_ratio = cfg["pitch_start_ratio"]
    for mode in modes:
        ph = mode.phase
        base_inc = mode.phase_inc
        env = 1.0
        decay = mode.decay_per_sample
        amp = mode.amp
        t_ph = mode.trem_phase
        t_inc = mode.trem_inc
        t_dep = mode.trem_depth
        for i in range(total):
            t = i / sr
            # Pitch sweep: starts high, settles to base frequency
            sweep = 1.0 + (start_ratio - 1.0) * math.exp(-t * sweep_rate)
            trem = 1.0 + t_dep * math.sin(t_ph)
            membrane[i] += math.sin(ph) * amp * env * trem
            ph += base_inc * sweep
            t_ph += t_inc
            env *= decay

    # --- Sub bass reinforcement ---
    sub_phase = rng.uniform(0, TAU)
    sub_inc = TAU * cfg["sub_freq"] / sr
    sub_decay_ps = math.exp(-cfg["sub_decay"] / sr)
    sub_env = 1.0
    sub_signal = _zeros(total)
    for i in range(total):
        sub_signal[i] = math.sin(sub_phase) * cfg["sub_level"] * sub_env * vel_norm
        sub_phase += sub_inc
        sub_env *= sub_decay_ps

    # --- Air cavity ---
    air = _render_air_cavity(f0, sr, total, vel, rng,
                             num_resonances=3, decay_rate=cfg["air_decay"],
                             level=cfg["air_level"])

    # --- Shell ---
    shell = _render_shell(f0, sr, total, vel, rng,
                          decay_rate=cfg["shell_decay"], level=cfg["shell_level"])

    # --- Beater impact ---
    impact = _render_impact(sr, total, vel, rng,
                            brightness=cfg["attack_brightness"],
                            click_level=cfg["click_level"],
                            noise_level=cfg["noise_level"],
                            duration_ms=5.0)

    # --- Mix ---
    out = _zeros(total)
    for i in range(total):
        out[i] = (membrane[i] * 0.40
                  + sub_signal[i]
                  + air[i]
                  + shell[i]
                  + impact[i])

    out = _tanh_softclip(out, cfg["drive"])
    out = _chain(out, sr, [
        ("hp", 25.0, 0.5, 0.0),
        ("peak", f0 * 1.8, 1.2, 2.5),
        ("lp", 12000.0, 0.7, 0.0),
    ])
    out = _dc_block(out)
    _apply_fade(out, sr, 0.025)
    return _normalize(out)


# ---- Snare ---------------------------------------------------------------

def _snare_engine(flavor: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))

    cfg = {
        "fundamental": 185.0,
        "num_modes": 22,
        "membrane_decay": 10.0,
        "membrane_decay_slope": 0.5,
        "wire_level": 0.55,
        "wire_decay": 14.0,
        "wire_buzz": 0.15,
        "shell_level": 0.18,
        "shell_decay": 8.0,
        "click_level": 0.40,
        "noise_level": 0.30,
        "attack_brightness": 0.7,
        "drive": 1.3,
    }
    if flavor == "bright":
        cfg.update({
            "fundamental": 210.0,
            "membrane_decay": 8.0,
            "wire_level": 0.65,
            "wire_decay": 18.0,
            "attack_brightness": 0.85,
            "drive": 1.5,
        })
    cfg.update(kw.get("params", {}))

    f0 = cfg["fundamental"]

    # --- Membrane modes ---
    modes = _build_membrane_modes(
        f0, sr, vel, rng,
        num_modes=cfg["num_modes"],
        decay_base=cfg["membrane_decay"],
        decay_slope=cfg["membrane_decay_slope"],
        amp_falloff=0.45,
        vel_timbre=0.05,
    )
    membrane = _render_modes(modes, total)

    # --- Snare wires ---
    wires = _render_snare_wires(
        membrane, sr, vel, rng,
        wire_level=cfg["wire_level"],
        wire_decay=cfg["wire_decay"],
        buzz_amount=cfg["wire_buzz"],
    )

    # --- Shell body ---
    shell = _render_shell(f0, sr, total, vel, rng,
                          decay_rate=cfg["shell_decay"],
                          level=cfg["shell_level"],
                          material="metal" if flavor == "bright" else "wood")

    # --- Stick impact ---
    impact = _render_impact(sr, total, vel, rng,
                            brightness=cfg["attack_brightness"],
                            click_level=cfg["click_level"],
                            noise_level=cfg["noise_level"],
                            duration_ms=4.0)

    # --- Mix ---
    out = _zeros(total)
    for i in range(total):
        out[i] = membrane[i] * 0.30 + wires[i] + shell[i] + impact[i]

    out = _tanh_softclip(out, cfg["drive"])
    out = _chain(out, sr, [
        ("hp", 120.0, 0.7, 0.0),
        ("peak", f0 * 2.0, 1.5, 1.5),
    ])
    out = _dc_block(out)
    _apply_fade(out, sr, 0.02)
    return _normalize(out)


# ---- Side stick -----------------------------------------------------------

def _side_stick_engine(**kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    params = {"rim_freq": 1400.0, "shell_freq": 800.0, **kw.get("params", {})}

    out = _zeros(total)

    # Rim click: very short, bright
    rim_ph = rng.uniform(0, TAU)
    rim_inc = TAU * params["rim_freq"] * (1 + rng.uniform(-0.05, 0.05)) / sr
    rim_decay = 180.0

    # Shell resonance: brief lower tone
    shell_ph = rng.uniform(0, TAU)
    shell_inc = TAU * params["shell_freq"] * (1 + rng.uniform(-0.05, 0.05)) / sr
    shell_decay = 60.0

    # Broadband click
    click_n = min(total, int(sr * 0.001))

    for i in range(total):
        t = i / sr
        # Rim
        rim = math.sin(rim_ph) * math.exp(-t * rim_decay) * 0.6
        rim_ph += rim_inc
        # Shell
        shell = math.sin(shell_ph) * math.exp(-t * shell_decay) * 0.25
        shell_ph += shell_inc
        # Click
        click = 0.0
        if i < click_n:
            click = rng.uniform(-1.0, 1.0) * (1.0 - i / click_n) * 0.8
        out[i] = (rim + shell + click) * vel_norm

    out = _chain(out, sr, [("hp", 400.0, 0.8, 0.0)])
    _apply_fade(out, sr, 0.008)
    return _normalize(out)


# ---- Hand clap -----------------------------------------------------------

def _clap_engine(**kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    # Multiple slaps staggered ~10-25ms apart
    num_slaps = rng.randint(3, 5)
    delays = [0.0]
    for _ in range(num_slaps - 1):
        delays.append(delays[-1] + rng.uniform(0.008, 0.022))

    # Each slap = filtered noise burst
    out = _zeros(total)
    hp_flt = _bq(sr, "hp", 1200.0, 0.8)
    bp_flt = _bq(sr, "bp", 2800.0, 2.5)

    for delay in delays:
        offset = int(delay * sr)
        slap_dur = rng.uniform(0.012, 0.025)
        slap_n = int(slap_dur * sr)
        for i in range(offset, min(total, offset + slap_n)):
            t = (i - offset) / sr
            env = math.exp(-t * 45.0)
            noise = rng.uniform(-1.0, 1.0)
            filtered = hp_flt.process(noise) * 0.5 + bp_flt.process(noise) * 0.5
            out[i] += filtered * env * vel_norm

    # Tail: filtered noise decay
    tail_flt = _bq(sr, "bp", 1800.0, 1.5)
    tail_start = int(delays[-1] * sr)
    for i in range(tail_start, total):
        t = (i - tail_start) / sr
        env = math.exp(-t * 22.0)
        out[i] += tail_flt.process(rng.uniform(-1.0, 1.0)) * env * vel_norm * 0.35

    out = _chain(out, sr, [("hp", 500.0, 0.7, 0.0)])
    _apply_fade(out, sr, 0.04)
    return _normalize(out)


# ---- Tom -----------------------------------------------------------------

def _tom_engine(**kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))

    cfg = {
        "fundamental": 150.0,
        "num_modes": 24,
        "membrane_decay": 7.0,
        "membrane_decay_slope": 0.55,
        "pitch_sweep_ratio": 1.3,
        "pitch_sweep_rate": 18.0,
        "air_level": 0.25,
        "air_decay": 3.5,
        "shell_level": 0.20,
        "shell_decay": 5.5,
        "click_level": 0.30,
        "noise_level": 0.15,
        "attack_brightness": 0.55,
        "drive": 1.2,
    }
    cfg.update(kw.get("params", {}))
    f0 = cfg["fundamental"]

    # --- Membrane modes with slight pitch sweep ---
    modes = _build_membrane_modes(
        f0, sr, vel, rng,
        num_modes=cfg["num_modes"],
        decay_base=cfg["membrane_decay"],
        decay_slope=cfg["membrane_decay_slope"],
        amp_falloff=0.50,
        vel_timbre=0.04,
    )

    membrane = _zeros(total)
    sweep_rate = cfg["pitch_sweep_rate"]
    start_ratio = cfg["pitch_sweep_ratio"]
    for mode in modes:
        ph = mode.phase
        base_inc = mode.phase_inc
        env = 1.0
        decay = mode.decay_per_sample
        amp = mode.amp
        t_ph = mode.trem_phase
        t_inc = mode.trem_inc
        t_dep = mode.trem_depth
        for i in range(total):
            t = i / sr
            sweep = 1.0 + (start_ratio - 1.0) * math.exp(-t * sweep_rate)
            trem = 1.0 + t_dep * math.sin(t_ph)
            membrane[i] += math.sin(ph) * amp * env * trem
            ph += base_inc * sweep
            t_ph += t_inc
            env *= decay

    # --- Air cavity ---
    air = _render_air_cavity(f0, sr, total, vel, rng,
                             num_resonances=2, decay_rate=cfg["air_decay"],
                             level=cfg["air_level"])

    # --- Shell ---
    shell = _render_shell(f0, sr, total, vel, rng,
                          decay_rate=cfg["shell_decay"],
                          level=cfg["shell_level"])

    # --- Stick impact ---
    impact = _render_impact(sr, total, vel, rng,
                            brightness=cfg["attack_brightness"],
                            click_level=cfg["click_level"],
                            noise_level=cfg["noise_level"],
                            duration_ms=5.0)

    # --- Mix ---
    out = _zeros(total)
    for i in range(total):
        out[i] = membrane[i] * 0.45 + air[i] + shell[i] + impact[i]

    out = _tanh_softclip(out, cfg["drive"])
    out = _chain(out, sr, [
        ("hp", max(40.0, f0 * 0.4), 0.6, 0.0),
        ("peak", f0 * 1.5, 1.0, 1.5),
    ])
    out = _dc_block(out)
    _apply_fade(out, sr, 0.025)
    return _normalize(out)


# ---- Hi-Hat --------------------------------------------------------------

def _hat_engine(mode_type: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    cfg = {
        "base_freq": 320.0,
        "num_modes": 65,
        "freq_power": 1.22,
        "freq_spread": 0.05,
    }
    if mode_type == "closed":
        cfg.update({"decay_base": 35.0, "decay_freq_power": 0.15})
    elif mode_type == "pedal":
        cfg.update({"decay_base": 28.0, "decay_freq_power": 0.18})
    else:  # open
        cfg.update({"decay_base": 5.5, "decay_freq_power": 0.25})
    cfg.update(kw.get("params", {}))

    # --- Inharmonic cymbal modes ---
    modes = _build_cymbal_modes(
        cfg["base_freq"], sr, vel, rng,
        num_modes=cfg["num_modes"],
        freq_power=cfg["freq_power"],
        freq_spread=cfg["freq_spread"],
        decay_base=cfg["decay_base"],
        decay_freq_power=cfg["decay_freq_power"],
        vel_timbre=0.015,
    )
    body = _render_modes(modes, total)

    # --- Metallic attack transient ---
    attack = _render_impact(sr, total, vel, rng,
                            brightness=0.9,
                            click_level=0.15,
                            noise_level=0.40,
                            click_freq=6000.0,
                            duration_ms=3.0)

    # --- HP noise layer for 'sizzle' ---
    sizzle_flt = _bq(sr, "hp", 8000.0, 0.9)
    sizzle_decay = cfg["decay_base"] * 0.7
    sizzle_dps = math.exp(-sizzle_decay / sr)
    sizzle_env = 1.0
    sizzle = _zeros(total)
    for i in range(total):
        sizzle[i] = sizzle_flt.process(rng.uniform(-1.0, 1.0)) * sizzle_env * 0.15 * vel_norm
        sizzle_env *= sizzle_dps

    # --- Mix ---
    out = _zeros(total)
    for i in range(total):
        out[i] = body[i] * 0.55 + attack[i] + sizzle[i]

    out = _chain(out, sr, [("hp", 500.0, 0.7, 0.0)])
    out = _dc_block(out)
    fade = 0.012 if mode_type == "closed" else (0.015 if mode_type == "pedal" else 0.08)
    _apply_fade(out, sr, fade)
    return _normalize(out)


# ---- Cymbal --------------------------------------------------------------

def _cymbal_engine(profile: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    catalog = {
        "crash": {
            "base_freq": 280.0, "num_modes": 80, "freq_power": 1.15,
            "decay_base": 1.8, "decay_freq_power": 0.35,
        },
        "crash_bright": {
            "base_freq": 320.0, "num_modes": 85, "freq_power": 1.12,
            "decay_base": 1.6, "decay_freq_power": 0.32,
        },
        "china": {
            "base_freq": 260.0, "num_modes": 75, "freq_power": 1.25,
            "decay_base": 1.5, "decay_freq_power": 0.30,
        },
        "splash": {
            "base_freq": 350.0, "num_modes": 60, "freq_power": 1.10,
            "decay_base": 3.5, "decay_freq_power": 0.25,
        },
        "ride": {
            "base_freq": 300.0, "num_modes": 70, "freq_power": 1.18,
            "decay_base": 1.2, "decay_freq_power": 0.40,
            "bell_freq": 1200.0, "bell_amp": 0.30,
        },
        "ride_bright": {
            "base_freq": 340.0, "num_modes": 70, "freq_power": 1.15,
            "decay_base": 1.0, "decay_freq_power": 0.38,
            "bell_freq": 1350.0, "bell_amp": 0.25,
        },
        "ride_bell": {
            "base_freq": 420.0, "num_modes": 50, "freq_power": 1.08,
            "decay_base": 2.5, "decay_freq_power": 0.20,
            "bell_freq": 1100.0, "bell_amp": 0.55,
        },
    }
    cfg = {**catalog.get(profile, catalog["crash"]), **kw.get("params", {})}

    # --- Inharmonic cymbal modes ---
    modes = _build_cymbal_modes(
        cfg["base_freq"], sr, vel, rng,
        num_modes=cfg["num_modes"],
        freq_power=cfg["freq_power"],
        decay_base=cfg["decay_base"],
        decay_freq_power=cfg["decay_freq_power"],
        vel_timbre=0.015,
    )
    body = _render_modes(modes, total)

    # --- Bell component (for rides) ---
    bell_signal = _zeros(total)
    if "bell_freq" in cfg:
        bell_f = cfg["bell_freq"] * (1 + rng.uniform(-0.02, 0.02))
        bell_amp = cfg["bell_amp"] * vel_norm
        bell_ph = rng.uniform(0, TAU)
        bell_inc = TAU * bell_f / sr
        bell_decay = math.exp(-4.0 / sr)
        bell_env = 1.0
        # A few bell harmonics
        bell_ph2 = rng.uniform(0, TAU)
        bell_inc2 = TAU * bell_f * 2.12 / sr
        bell_ph3 = rng.uniform(0, TAU)
        bell_inc3 = TAU * bell_f * 3.48 / sr
        for i in range(total):
            bell_signal[i] = (
                math.sin(bell_ph) * 1.0 +
                math.sin(bell_ph2) * 0.35 +
                math.sin(bell_ph3) * 0.15
            ) * bell_amp * bell_env
            bell_ph += bell_inc
            bell_ph2 += bell_inc2
            bell_ph3 += bell_inc3
            bell_env *= bell_decay

    # --- Attack ---
    attack = _render_impact(sr, total, vel, rng,
                            brightness=0.8,
                            click_level=0.20,
                            noise_level=0.35,
                            click_freq=4500.0,
                            duration_ms=4.0)

    # --- Shimmer noise ---
    shimmer_flt = _bq(sr, "bp", 7000.0, 1.5)
    shimmer_decay_ps = math.exp(-cfg["decay_base"] * 0.5 / sr)
    shimmer_env = 1.0
    shimmer = _zeros(total)
    for i in range(total):
        shimmer[i] = shimmer_flt.process(rng.uniform(-1.0, 1.0)) * shimmer_env * 0.12 * vel_norm
        shimmer_env *= shimmer_decay_ps

    # --- Mix ---
    out = _zeros(total)
    for i in range(total):
        out[i] = body[i] * 0.50 + bell_signal[i] + attack[i] + shimmer[i]

    out = _chain(out, sr, [("hp", 200.0, 0.6, 0.0)])
    out = _dc_block(out)
    _apply_fade(out, sr, 0.15)
    return _normalize(out)


# ---- Cowbell -------------------------------------------------------------

def _cowbell_engine(**kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    params = {"fundamental": 540.0, **kw.get("params", {})}
    f0 = params["fundamental"]

    # Cowbell: a few strong inharmonic metallic modes
    mode_ratios = [
        (1.00, 1.0), (1.35, 0.8), (1.72, 0.5),
        (2.10, 0.35), (2.90, 0.25), (3.60, 0.15),
    ]
    modes: List[_Mode] = []
    for ratio, amp_base in mode_ratios:
        freq = f0 * ratio * (1 + rng.uniform(-0.008, 0.008))
        amp = amp_base * vel_norm
        decay = 7.0 * (ratio ** 0.4)
        decay_ps = math.exp(-decay / sr)
        modes.append(_Mode(
            freq=freq, amp=amp,
            decay_per_sample=decay_ps,
            phase=rng.uniform(0, TAU),
            phase_inc=TAU * freq / sr,
        ))

    body = _render_modes(modes, total)

    # Strike click
    impact = _render_impact(sr, total, vel, rng,
                            brightness=0.5, click_level=0.25,
                            noise_level=0.10, duration_ms=2.0)

    out = _zeros(total)
    for i in range(total):
        out[i] = body[i] + impact[i]

    out = _chain(out, sr, [("hp", 300.0, 0.7, 0.0)])
    _apply_fade(out, sr, 0.03)
    return _normalize(out)


# ---- Tambourine ----------------------------------------------------------

def _tambourine_engine(**kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    # Jingles: many small metallic discs
    jingle_modes = _build_cymbal_modes(
        600.0, sr, vel, rng,
        num_modes=30,
        freq_power=1.3,
        freq_spread=0.08,
        decay_base=8.0,
        decay_freq_power=0.2,
    )
    jingles = _render_modes(jingle_modes, total)

    # Shell thump (small frame drum)
    shell_modes = _build_membrane_modes(
        280.0, sr, vel, rng,
        num_modes=8, decay_base=18.0,
        decay_slope=0.4, amp_falloff=0.6,
    )
    shell = _render_modes(shell_modes, total)

    # Shake noise
    shake_flt = _bq(sr, "hp", 4000.0, 0.8)
    shake = _zeros(total)
    for i in range(total):
        t = i / sr
        env = math.exp(-t * 10.0)
        shake[i] = shake_flt.process(rng.uniform(-1.0, 1.0)) * env * 0.3 * vel_norm

    out = _zeros(total)
    for i in range(total):
        out[i] = jingles[i] * 0.5 + shell[i] * 0.2 + shake[i]

    out = _chain(out, sr, [("hp", 800.0, 0.6, 0.0)])
    _apply_fade(out, sr, 0.06)
    return _normalize(out)


# ---- Bongo / Conga / Timbale (hand drums) ---------------------------------

def _hand_drum_engine(kind: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))

    presets = {
        "bongo_high":       {"fundamental": 420.0, "decay": 10.0, "num_modes": 18, "muted": False},
        "bongo_low":        {"fundamental": 280.0, "decay": 9.0,  "num_modes": 18, "muted": False},
        "conga_high_open":  {"fundamental": 300.0, "decay": 8.0,  "num_modes": 20, "muted": False},
        "conga_high_mute":  {"fundamental": 310.0, "decay": 22.0, "num_modes": 16, "muted": True},
        "conga_low":        {"fundamental": 200.0, "decay": 7.0,  "num_modes": 20, "muted": False},
        "timbale_high":     {"fundamental": 400.0, "decay": 8.0,  "num_modes": 20, "muted": False},
        "timbale_low":      {"fundamental": 310.0, "decay": 7.0,  "num_modes": 20, "muted": False},
        "surdo_open":       {"fundamental": 65.0,  "decay": 4.5,  "num_modes": 22, "muted": False},
        "surdo_mute":       {"fundamental": 65.0,  "decay": 12.0, "num_modes": 18, "muted": True},
    }
    cfg = {**presets.get(kind, presets["bongo_high"]), **kw.get("params", {})}
    f0 = cfg["fundamental"]

    # Membrane modes
    modes = _build_membrane_modes(
        f0, sr, vel, rng,
        num_modes=cfg["num_modes"],
        decay_base=cfg["decay"],
        decay_slope=0.5,
        amp_falloff=0.55,
        vel_timbre=0.04,
    )

    # Slight pitch sweep for hand drums
    membrane = _zeros(total)
    sweep_rate = 25.0 if cfg["muted"] else 15.0
    start_ratio = 1.15
    for mode in modes:
        ph = mode.phase
        base_inc = mode.phase_inc
        env = 1.0
        decay = mode.decay_per_sample
        amp = mode.amp
        for i in range(total):
            t = i / sr
            sweep = 1.0 + (start_ratio - 1.0) * math.exp(-t * sweep_rate)
            membrane[i] += math.sin(ph) * amp * env
            ph += base_inc * sweep
            env *= decay

    # Shell resonance
    shell = _render_shell(f0, sr, total, vel, rng,
                          decay_rate=cfg["decay"] * 1.2,
                          level=0.12)

    # Hand slap attack (softer than stick)
    impact = _render_impact(sr, total, vel, rng,
                            brightness=0.35,
                            click_level=0.15,
                            noise_level=0.20,
                            duration_ms=4.0)

    out = _zeros(total)
    for i in range(total):
        out[i] = membrane[i] * 0.55 + shell[i] + impact[i]

    out = _tanh_softclip(out, 1.15)
    out = _chain(out, sr, [("hp", max(40.0, f0 * 0.35), 0.6, 0.0)])
    out = _dc_block(out)
    if cfg["muted"]:
        _apply_fade(out, sr, 0.04)
    else:
        _apply_fade(out, sr, 0.025)
    return _normalize(out)


# ---- Agogo ---------------------------------------------------------------

def _agogo_engine(low: bool, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    f0 = 600.0 if low else 820.0
    f0 = kw.get("params", {}).get("fundamental", f0)

    # Metallic bell: several strong modes
    mode_ratios = [
        (1.0, 1.0), (2.08, 0.5), (3.42, 0.3),
        (4.15, 0.2), (5.3, 0.12), (6.8, 0.08),
    ]
    modes: List[_Mode] = []
    for ratio, amp_base in mode_ratios:
        freq = f0 * ratio * (1 + rng.uniform(-0.005, 0.005))
        decay = 10.0 * (ratio ** 0.35)
        modes.append(_Mode(
            freq=freq, amp=amp_base * vel_norm,
            decay_per_sample=math.exp(-decay / sr),
            phase=rng.uniform(0, TAU),
            phase_inc=TAU * freq / sr,
        ))

    body = _render_modes(modes, total)
    impact = _render_impact(sr, total, vel, rng,
                            brightness=0.45, click_level=0.20,
                            noise_level=0.08, duration_ms=2.0)

    out = _zeros(total)
    for i in range(total):
        out[i] = body[i] + impact[i]

    out = _chain(out, sr, [("hp", 400.0, 0.7, 0.0)])
    _apply_fade(out, sr, 0.03)
    return _normalize(out)


# ---- Claves / Wood Block / Castanets (wood percussion) --------------------

def _wood_engine(kind: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    presets = {
        "claves":          {"freq": 2500.0, "decay": 30.0, "modes": 4, "click": 0.3},
        "woodblock_high":  {"freq": 1100.0, "decay": 25.0, "modes": 5, "click": 0.25},
        "woodblock_low":   {"freq": 750.0,  "decay": 22.0, "modes": 5, "click": 0.25},
        "castanets":       {"freq": 2000.0, "decay": 45.0, "modes": 3, "click": 0.35},
    }
    cfg = {**presets.get(kind, presets["claves"]), **kw.get("params", {})}
    f0 = cfg["freq"]

    # Wood resonance: a few bright, fast-decaying modes
    mode_ratios = [(1.0, 1.0), (2.2, 0.45), (3.7, 0.2), (5.1, 0.12), (6.9, 0.06)]
    modes: List[_Mode] = []
    for idx in range(min(cfg["modes"], len(mode_ratios))):
        ratio, amp_base = mode_ratios[idx]
        freq = f0 * ratio * (1 + rng.uniform(-0.008, 0.008))
        decay = cfg["decay"] * (ratio ** 0.4)
        modes.append(_Mode(
            freq=freq, amp=amp_base * vel_norm,
            decay_per_sample=math.exp(-decay / sr),
            phase=rng.uniform(0, TAU),
            phase_inc=TAU * freq / sr,
        ))

    body = _render_modes(modes, total)

    # Very short click
    impact = _render_impact(sr, total, vel, rng,
                            brightness=0.6,
                            click_level=cfg["click"],
                            noise_level=0.10,
                            duration_ms=1.5)

    out = _zeros(total)
    for i in range(total):
        out[i] = body[i] + impact[i]

    out = _chain(out, sr, [("hp", min(f0 * 0.3, 400.0), 0.7, 0.0)])
    _apply_fade(out, sr, 0.015)
    return _normalize(out)


# ---- Triangle (percussion) -----------------------------------------------

def _triangle_perc_engine(open_: bool, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    f0 = kw.get("params", {}).get("fundamental", 2200.0)

    # Triangle instrument: nearly harmonic but with slight inharmonicity
    mode_ratios = [
        (1.0, 1.0), (2.0, 0.55), (3.01, 0.32),
        (4.03, 0.20), (5.05, 0.12), (6.08, 0.08),
        (7.12, 0.05), (8.18, 0.03),
    ]
    decay_base = 8.0 if open_ else 35.0

    modes: List[_Mode] = []
    for ratio, amp_base in mode_ratios:
        freq = f0 * ratio * (1 + rng.uniform(-0.003, 0.003))
        decay = decay_base * (ratio ** 0.3)
        modes.append(_Mode(
            freq=freq, amp=amp_base * vel_norm,
            decay_per_sample=math.exp(-decay / sr),
            phase=rng.uniform(0, TAU),
            phase_inc=TAU * freq / sr,
        ))

    body = _render_modes(modes, total)
    impact = _render_impact(sr, total, vel, rng,
                            brightness=0.7, click_level=0.15,
                            noise_level=0.05, duration_ms=1.0)

    out = _zeros(total)
    for i in range(total):
        out[i] = body[i] + impact[i]

    out = _chain(out, sr, [("hp", 1000.0, 0.7, 0.0)])
    if not open_:
        _apply_fade(out, sr, 0.04)
    else:
        _apply_fade(out, sr, 0.08)
    return _normalize(out)


# ---- Shaker / Cabasa / Maracas -------------------------------------------

def _shaker_engine(style: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    presets = {
        "maracas":  {"rate": 11.0, "decay": 9.0,  "color": 0.15, "hp_freq": 3000.0},
        "cabasa":   {"rate": 8.0,  "decay": 10.0, "color": 0.25, "hp_freq": 2500.0},
        "shaker":   {"rate": 9.0,  "decay": 8.5,  "color": 0.20, "hp_freq": 2800.0},
        "jingle":   {"rate": 7.0,  "decay": 7.0,  "color": 0.30, "hp_freq": 2200.0},
    }
    cfg = {**presets.get(style, presets["shaker"]), **kw.get("params", {})}

    # Granular shaker: many tiny particle impacts
    hp_flt = _bq(sr, "hp", cfg["hp_freq"], 0.9)
    bp_flt = _bq(sr, "bp", cfg["hp_freq"] * 1.5, 2.0)
    out = _zeros(total)

    # Envelope with slight pulsing for "shake" feel
    for i in range(total):
        t = i / sr
        jitter = rng.uniform(-0.003, 0.003)
        shake_env = max(0.0, math.sin(TAU * cfg["rate"] * (t + jitter)))
        shake_env = shake_env ** 0.5  # soften the envelope shape
        overall_env = math.exp(-t * cfg["decay"])

        noise = rng.uniform(-1.0, 1.0)
        filtered = hp_flt.process(noise) * 0.6 + bp_flt.process(noise) * 0.4
        out[i] = filtered * shake_env * overall_env * vel_norm

    _apply_fade(out, sr, 0.05)
    return _normalize(out)


# ---- Whistle -------------------------------------------------------------

def _whistle_engine(style: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    short = style == "short"
    f_start = 1600.0 if short else 1400.0
    f_end = 2400.0 if short else 2700.0
    gliss = 5.0 if short else 2.5
    decay = 12.0 if short else 5.0
    vib_rate = 5.0
    vib_depth = 0.015 if short else 0.025
    noise_mix = 0.12

    params = kw.get("params", {})
    f_start = params.get("freq_start", f_start)
    f_end = params.get("freq_end", f_end)

    phase = rng.uniform(0, TAU)
    noise_flt = _bq(sr, "bp", (f_start + f_end) / 2, 3.0)
    attack_time = 0.008
    out = _zeros(total)

    for i in range(total):
        t = i / sr
        # Pitch glissando
        freq = f_end + (f_start - f_end) * math.exp(-t * gliss)
        # Vibrato
        vibrato = 1.0 + vib_depth * math.sin(TAU * vib_rate * t)
        phase += TAU * freq * vibrato / sr
        # Tone with overtone
        tone = math.sin(phase) + 0.15 * math.sin(2 * phase + 0.3)
        # Breath noise
        breath = noise_flt.process(rng.uniform(-1.0, 1.0))
        # Attack/decay envelope
        if t < attack_time:
            env = (t / attack_time) ** 1.5
        else:
            env = math.exp(-(t - attack_time) * decay)
        value = tone * (1 - noise_mix) + breath * noise_mix
        out[i] = value * env * vel_norm

    _apply_fade(out, sr, 0.04)
    return _normalize(out)


# ---- Guiro ---------------------------------------------------------------

def _guiro_engine(style: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    short = style == "short"
    strokes = 1 if short else 2
    stroke_len = 0.12 if short else 0.20
    interval = 0.16
    ridge_freq = rng.uniform(18.0, 26.0)

    scratch_flt = _bq(sr, "bp", 2800.0, 2.5)
    body_flt = _bq(sr, "bp", 800.0, 1.5)
    out = _zeros(total)

    stroke_times = [n * interval for n in range(strokes)]
    for i in range(total):
        t = i / sr
        value = 0.0
        for start in stroke_times:
            if t < start or t > start + stroke_len:
                continue
            local = (t - start) / stroke_len
            ridge = math.sin(TAU * ridge_freq * local)
            noise = scratch_flt.process(rng.uniform(-1.0, 1.0))
            body = body_flt.process(rng.uniform(-1.0, 1.0)) * 0.3
            env = (1 - local) * math.exp(-local * 4.0)
            value += (noise * 0.55 + ridge * 0.25 + body * 0.2) * env
        out[i] = value * vel_norm

    _apply_fade(out, sr, 0.03)
    return _normalize(out)


# ---- Cuica ---------------------------------------------------------------

def _cuica_engine(state: str, **kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    mute = state == "mute"
    f_start = 240.0 if mute else 330.0
    f_end = 160.0 if mute else 270.0
    sweep = 4.5 if mute else 3.0
    noise_level = 0.25 if mute else 0.35
    decay = 7.0 if mute else 4.5

    params = kw.get("params", {})
    f_start = params.get("freq_start", f_start)
    f_end = params.get("freq_end", f_end)

    phase = rng.uniform(0, TAU)
    wobble_ph = rng.uniform(0, TAU)
    wobble_inc = TAU * 9.0 / sr
    friction_flt = _bq(sr, "bp", (f_start + f_end), 2.0)
    out = _zeros(total)

    for i in range(total):
        t = i / sr
        freq = f_end + (f_start - f_end) * math.exp(-t * sweep)
        wobble_ph += wobble_inc
        vibrato = 1.0 + 0.06 * math.sin(wobble_ph)
        phase += TAU * freq * vibrato / sr
        tone = math.sin(phase) + 0.35 * math.sin(2 * phase + 0.6)
        friction = friction_flt.process(rng.uniform(-1.0, 1.0))
        env = math.exp(-t * decay)
        value = tone * (1 - noise_level) + friction * noise_level
        out[i] = value * env * vel_norm

    _apply_fade(out, sr, 0.05)
    return _normalize(out)


# ---- Vibraslap -----------------------------------------------------------

def _vibraslap_engine(**kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    # Pendulum body hit + rattle of metal rivets
    num_rattles = rng.randint(8, 14)
    rattle_times = sorted([rng.uniform(0.0, dur * 0.5) for _ in range(num_rattles)])

    rattle_flt = _bq(sr, "bp", 3200.0, 3.0)
    out = _zeros(total)

    for i in range(total):
        t = i / sr
        value = 0.0
        for rt in rattle_times:
            dt = max(0.0, t - rt)
            if dt > 0.15:
                continue
            env = math.exp(-dt * 25.0)
            value += rattle_flt.process(rng.uniform(-1.0, 1.0)) * env * 0.3

        # Pendulum swing
        pendulum = math.sin(TAU * 3.5 * t) * math.exp(-t * 2.5) * 0.4
        out[i] = (value + pendulum) * vel_norm

    _apply_fade(out, sr, 0.1)
    return _normalize(out)


# ---- Bell Tree -----------------------------------------------------------

def _bell_tree_engine(**kw: Any) -> Signal:
    sr = kw["sample_rate"]
    dur = kw["duration"]
    vel = kw["velocity"]
    rng = random.Random(kw["seed"])
    total = max(1, int(sr * dur))
    vel_norm = vel / 127.0

    # Cascading bell strikes descending in pitch
    num_bells = rng.randint(8, 14)
    bell_interval = dur * 0.5 / num_bells
    out = _zeros(total)

    for b in range(num_bells):
        t_start = b * bell_interval + rng.uniform(0, bell_interval * 0.3)
        offset = int(t_start * sr)
        # Each bell has a few inharmonic modes
        f0 = 4000.0 - b * 200.0 + rng.uniform(-50, 50)
        decay = 6.0 + b * 0.5
        for freq_ratio, amp in [(1.0, 0.8), (2.3, 0.3), (3.5, 0.15)]:
            freq = f0 * freq_ratio
            ph = rng.uniform(0, TAU)
            inc = TAU * freq / sr
            decay_ps = math.exp(-decay / sr)
            env = 1.0
            bell_dur = min(total - offset, int(sr * 0.5))
            for i in range(bell_dur):
                if offset + i < total:
                    out[offset + i] += math.sin(ph) * amp * env * vel_norm * 0.3
                ph += inc
                env *= decay_ps

    _apply_fade(out, sr, 0.1)
    return _normalize(out)


# ============================================================================
# Round-robin variation
# ============================================================================

def _apply_rr_variation(samples: Signal, rr_index: int, seed: int) -> Signal:
    """Apply micro-variations for round-robin differentiation.

    Each round-robin hit gets:
      - Slight amplitude jitter (±4%)
      - Spectral tilt (gentle brightening/darkening)
      - Tiny colored-noise floor injection
      - Subtle timing shift via phase-based resampling
    """
    total = len(samples)
    if total == 0 or rr_index == 0:
        return samples

    vseed = (seed ^ 0xA511E9B3) + rr_index * 7919
    rng = random.Random(vseed & 0xFFFFFFFF)

    # Amplitude jitter
    amp = 1.0 + rng.uniform(-0.04, 0.04)

    # Spectral tilt (position-dependent gain)
    tilt = rng.uniform(-0.025, 0.025)

    out: Signal = []
    denom = max(1, total - 1)
    for i, s in enumerate(samples):
        pos = (i / denom) * 2.0 - 1.0  # -1 .. +1
        out.append(s * amp * (1.0 + tilt * pos))

    # Subtle noise injection
    noise_level = rng.uniform(0.002, 0.008)
    noise_color = rng.uniform(0.3, 0.7)
    noise = _colored(total, rng, noise_color)
    for i in range(total):
        out[i] += noise[i] * noise_level

    return out


# ============================================================================
# Recipe map
# ============================================================================

def _wrap(factory: Callable[..., Signal]) -> Callable[..., Signal]:
    """Wrap an engine so it applies round-robin variation automatically."""
    def wrapper(**kw: Any) -> Signal:
        samples = factory(**kw)
        rr = kw.get("rr_index", 0)
        if rr > 0 and samples:
            samples = _apply_rr_variation(samples, rr, kw.get("seed", 0))
        return samples
    return wrapper


_RECIPES: Dict[str, Callable[..., Signal]] = {
    # Kicks
    "acoustic_kick":        _wrap(lambda **kw: _kick_engine("standard", **kw)),
    "acoustic_kick_tight":  _wrap(lambda **kw: _kick_engine("tight", **kw)),
    # Snares
    "acoustic_snare":       _wrap(lambda **kw: _snare_engine("standard", **kw)),
    "acoustic_snare_bright": _wrap(lambda **kw: _snare_engine("bright", **kw)),
    # Side stick
    "acoustic_side_stick":  _wrap(_side_stick_engine),
    # Hand clap
    "acoustic_clap":        _wrap(_clap_engine),
    # Toms
    "acoustic_tom":         _wrap(_tom_engine),
    # Hi-hats
    "acoustic_hat_closed":  _wrap(lambda **kw: _hat_engine("closed", **kw)),
    "acoustic_hat_pedal":   _wrap(lambda **kw: _hat_engine("pedal", **kw)),
    "acoustic_hat_open":    _wrap(lambda **kw: _hat_engine("open", **kw)),
    # Cymbals
    "acoustic_crash":       _wrap(lambda **kw: _cymbal_engine("crash", **kw)),
    "acoustic_crash_bright": _wrap(lambda **kw: _cymbal_engine("crash_bright", **kw)),
    "acoustic_china":       _wrap(lambda **kw: _cymbal_engine("china", **kw)),
    "acoustic_splash":      _wrap(lambda **kw: _cymbal_engine("splash", **kw)),
    "acoustic_ride":        _wrap(lambda **kw: _cymbal_engine("ride", **kw)),
    "acoustic_ride_bright": _wrap(lambda **kw: _cymbal_engine("ride_bright", **kw)),
    "acoustic_ride_bell":   _wrap(lambda **kw: _cymbal_engine("ride_bell", **kw)),
    # Cowbell
    "acoustic_cowbell":     _wrap(_cowbell_engine),
    # Tambourine
    "acoustic_tambourine":  _wrap(_tambourine_engine),
    # Vibraslap
    "acoustic_vibraslap":   _wrap(_vibraslap_engine),
    # Latin hand drums
    "acoustic_bongo_high":  _wrap(lambda **kw: _hand_drum_engine("bongo_high", **kw)),
    "acoustic_bongo_low":   _wrap(lambda **kw: _hand_drum_engine("bongo_low", **kw)),
    "acoustic_conga_high_mute": _wrap(lambda **kw: _hand_drum_engine("conga_high_mute", **kw)),
    "acoustic_conga_high_open": _wrap(lambda **kw: _hand_drum_engine("conga_high_open", **kw)),
    "acoustic_conga_low":   _wrap(lambda **kw: _hand_drum_engine("conga_low", **kw)),
    "acoustic_timbale_high": _wrap(lambda **kw: _hand_drum_engine("timbale_high", **kw)),
    "acoustic_timbale_low": _wrap(lambda **kw: _hand_drum_engine("timbale_low", **kw)),
    # Agogo
    "acoustic_agogo_high":  _wrap(lambda **kw: _agogo_engine(False, **kw)),
    "acoustic_agogo_low":   _wrap(lambda **kw: _agogo_engine(True, **kw)),
    # Shakers
    "acoustic_cabasa":      _wrap(lambda **kw: _shaker_engine("cabasa", **kw)),
    "acoustic_maracas":     _wrap(lambda **kw: _shaker_engine("maracas", **kw)),
    "acoustic_shaker":      _wrap(lambda **kw: _shaker_engine("shaker", **kw)),
    "acoustic_jingle_bell": _wrap(lambda **kw: _shaker_engine("jingle", **kw)),
    # Whistles
    "acoustic_whistle_short": _wrap(lambda **kw: _whistle_engine("short", **kw)),
    "acoustic_whistle_long":  _wrap(lambda **kw: _whistle_engine("long", **kw)),
    # Guiro
    "acoustic_guiro_short": _wrap(lambda **kw: _guiro_engine("short", **kw)),
    "acoustic_guiro_long":  _wrap(lambda **kw: _guiro_engine("long", **kw)),
    # Wood percussion
    "acoustic_claves":      _wrap(lambda **kw: _wood_engine("claves", **kw)),
    "acoustic_woodblock_high": _wrap(lambda **kw: _wood_engine("woodblock_high", **kw)),
    "acoustic_woodblock_low":  _wrap(lambda **kw: _wood_engine("woodblock_low", **kw)),
    "acoustic_castanets":   _wrap(lambda **kw: _wood_engine("castanets", **kw)),
    # Cuica
    "acoustic_cuica_mute":  _wrap(lambda **kw: _cuica_engine("mute", **kw)),
    "acoustic_cuica_open":  _wrap(lambda **kw: _cuica_engine("open", **kw)),
    # Triangle
    "acoustic_triangle_mute": _wrap(lambda **kw: _triangle_perc_engine(False, **kw)),
    "acoustic_triangle_open": _wrap(lambda **kw: _triangle_perc_engine(True, **kw)),
    # Surdo
    "acoustic_surdo_mute":  _wrap(lambda **kw: _hand_drum_engine("surdo_mute", **kw)),
    "acoustic_surdo_open":  _wrap(lambda **kw: _hand_drum_engine("surdo_open", **kw)),
    # Bell tree
    "acoustic_bell_tree":   _wrap(_bell_tree_engine),
}
