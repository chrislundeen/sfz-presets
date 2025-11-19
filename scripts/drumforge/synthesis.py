from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

TAU = 2 * math.pi
Signal = List[float]


def render_instrument(
    recipe: str,
    *,
    sample_rate: int,
    duration: float,
    velocity: int,
    round_robin_index: int,
    params: Dict[str, Any] | None = None,
) -> Signal:
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
    if round_robin_index > 0 and samples:
        samples = _apply_round_robin_variation(samples, round_robin_index, seed)
    return samples


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _velocity_gain(velocity: int, curve: float = 1.0, floor: float = 0.05) -> float:
    return max(floor, (velocity / 127.0) ** curve)


def _normalize(samples: Signal, target: float = 0.96) -> Signal:
    peak = max((abs(s) for s in samples), default=0.0)
    if peak == 0:
        return samples
    scale = target / peak
    return [s * scale for s in samples]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _pitch_curve(t: float, start: float, end: float, sweep: float) -> float:
    if sweep <= 0:
        return end
    return end + (start - end) * math.exp(-t * sweep)


def _ad_env(t: float, attack: float, decay: float, curve: float = 2.2) -> float:
    if attack > 0 and t < attack:
        return (t / attack) ** curve
    if decay <= 0:
        return 1.0
    offset = max(0.0, t - attack)
    return math.exp(-offset * decay)


def _colored_noise(total: int, rng: random.Random, color: float = 0.3) -> Signal:
    alpha = max(0.0, min(0.999, color))
    prev = 0.0
    out: Signal = []
    for _ in range(total):
        white = rng.uniform(-1.0, 1.0)
        prev = alpha * prev + (1 - alpha) * white
        out.append(prev)
    return out


def _highpass_noise(signal: Signal, amount: float = 0.5) -> Signal:
    if not signal:
        return signal
    filtered: Signal = []
    prev = signal[0]
    for sample in signal:
        hp = sample - prev
        prev = sample * (1 - amount) + prev * amount
        filtered.append(hp)
    return filtered


def _blend(a: Signal, b: Signal, mix: float) -> Signal:
    total = min(len(a), len(b))
    result = []
    for i in range(total):
        result.append(a[i] * (1 - mix) + b[i] * mix)
    return result + a[total:]


def _tanh_drive(samples: Signal, drive: float) -> Signal:
    if drive <= 1.0:
        return samples
    return [math.tanh(sample * drive) for sample in samples]


def _apply_fade(samples: Signal, sample_rate: int, fade: float) -> None:
    length = min(len(samples), int(sample_rate * fade))
    if length <= 0:
        return
    for i in range(1, length + 1):
        samples[-i] *= i / length


def _apply_round_robin_variation(samples: Signal, rr_index: int, seed: int) -> Signal:
    total = len(samples)
    if total == 0:
        return samples
    variation_seed = (seed ^ 0xA511E9B3) + rr_index * 7919
    rng = random.Random(variation_seed & 0xFFFFFFFF)
    amp = 1.0 + rng.uniform(-0.035, 0.035)
    tilt = rng.uniform(-0.02, 0.02)
    shaped: Signal = []
    denom = max(1, total - 1)
    for i, sample in enumerate(samples):
        pos = (i / denom) * 2.0 - 1.0  # -1 .. 1
        shaped_sample = sample * (1.0 + tilt * pos)
        shaped.append(shaped_sample * amp)
    noise_mix = rng.uniform(0.003, 0.01)
    if noise_mix > 0.0:
        noise_color = rng.uniform(0.3, 0.75)
        noise = _colored_noise(total, rng, noise_color)
        shaped = [shaped[i] + noise[i] * noise_mix for i in range(total)]
    return shaped


@dataclass
class _Biquad:
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float
    z1: float = 0.0
    z2: float = 0.0

    def process(self, sample: float) -> float:
        result = sample * self.b0 + self.z1
        self.z1 = sample * self.b1 + self.z2 - self.a1 * result
        self.z2 = sample * self.b2 - self.a2 * result
        return result


def _make_biquad(
    sample_rate: int,
    mode: str,
    freq: float,
    q: float = 0.707,
    gain_db: float = 0.0,
) -> _Biquad:
    freq = max(1.0, min(sample_rate * 0.45, freq))
    omega = 2 * math.pi * freq / sample_rate
    sn = math.sin(omega)
    cs = math.cos(omega)
    alpha = sn / (2 * max(0.001, q))
    A = 10 ** (gain_db / 40)

    if mode == "lowpass":
        b0 = (1 - cs) / 2
        b1 = 1 - cs
        b2 = (1 - cs) / 2
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha
    elif mode == "highpass":
        b0 = (1 + cs) / 2
        b1 = -(1 + cs)
        b2 = (1 + cs) / 2
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha
    elif mode == "bandpass":
        b0 = sn / 2
        b1 = 0.0
        b2 = -sn / 2
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha
    elif mode == "notch":
        b0 = 1.0
        b1 = -2 * cs
        b2 = 1.0
        a0 = 1 + alpha
        a1 = -2 * cs
        a2 = 1 - alpha
    elif mode == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cs
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cs
        a2 = 1 - alpha / A
    else:
        raise ValueError(f"Unsupported filter mode '{mode}'")

    inv_a0 = 1.0 / a0
    return _Biquad(
        b0=b0 * inv_a0,
        b1=b1 * inv_a0,
        b2=b2 * inv_a0,
        a1=a1 * inv_a0,
        a2=a2 * inv_a0,
    )


def _apply_filter_chain(
    samples: Signal,
    sample_rate: int,
    stages: Sequence[Tuple[str, float, float, float]] | None,
) -> Signal:
    if not stages:
        return samples
    data = samples[:]
    filters = [
        _make_biquad(sample_rate, mode, freq, q, gain)
        for mode, freq, q, gain in stages
    ]
    for i, sample in enumerate(data):
        val = sample
        for filt in filters:
            val = filt.process(val)
        data[i] = val
    return data


def _modal_stack(
    *,
    sample_rate: int,
    duration: float,
    velocity: int,
    base_freq: float,
    partials: List[Tuple[float, float] | Tuple[float, float, float]],
    decay: float,
    noise_level: float = 0.0,
    noise_color: float = 0.3,
    detune: float = 0.0,
    seed: int,
) -> Signal:
    total = max(1, int(sample_rate * duration))
    rng = random.Random(seed)
    gain = _velocity_gain(velocity, curve=1.0)
    noise_buf = None
    if noise_level:
        noise_buf = _highpass_noise(_colored_noise(total, rng, noise_color), 0.8)
    states = []
    for entry in partials:
        ratio = entry[0]
        weight = entry[1]
        damp = entry[2] if len(entry) > 2 else 1.0
        freq = base_freq * ratio * (1 + detune * rng.uniform(-1, 1))
        states.append(
            {
                "freq": freq,
                "weight": weight,
                "phase": rng.uniform(0, TAU),
                "damp": max(0.2, damp),
            }
        )
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        env = math.exp(-t * decay)
        value = 0.0
        for state in states:
            state["phase"] += TAU * state["freq"] / sample_rate
            partial_env = math.exp(-t * decay * state["damp"])
            value += math.sin(state["phase"]) * state["weight"] * partial_env
        if noise_buf is not None:
            value = value * (1 - noise_level) + noise_buf[i] * noise_level
        samples.append(value * env * gain)
    return samples


# ---------------------------------------------------------------------------
# Core synthesis engines
# ---------------------------------------------------------------------------


def _kick_engine(profile: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    base_params = {
        "body_freq": 125.0,
        "body_end": 48.0,
        "pitch_decay": 6.0,
        "body_decay": 7.0,
        "body_level": 0.9,
        "click_level": 0.35,
        "click_decay": 120.0,
        "noise_level": 0.12,
        "noise_decay": 50.0,
        "drive": 1.6,
        "sub_level": 0.45,
        "sub_decay": 3.5,
        "sub_freq": 42.0,
    }
    if profile == "acoustic":
        base_params.update(
            {
                "body_freq": 140.0,
                "body_end": 55.0,
                "body_decay": 8.5,
                "pitch_decay": 7.0,
                "click_level": 0.28,
                "noise_level": 0.18,
                "drive": 1.3,
            }
        )
    elif profile == "modern":
        base_params.update(
            {
                "body_freq": 115.0,
                "body_end": 38.0,
                "pitch_decay": 5.0,
                "sub_level": 0.6,
                "drive": 1.9,
            }
        )
    params = {**base_params, **kwargs.get("params", {})}

    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=1.25)
    click_noise = _highpass_noise(_colored_noise(total, rng, 0.05), 0.92)
    hiss_noise = _colored_noise(total, rng, 0.75)
    body_phase = rng.uniform(0, TAU)
    sub_phase = rng.uniform(0, TAU)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        freq = _pitch_curve(t, params["body_freq"], params["body_end"], params["pitch_decay"])
        body_phase += TAU * freq / sample_rate
        body = math.sin(body_phase) + 0.25 * math.sin(2 * body_phase + 0.35)
        body *= math.exp(-t * params["body_decay"])

        sub_phase += TAU * params["sub_freq"] / sample_rate
        sub = math.sin(sub_phase) * math.exp(-t * params["sub_decay"])

        click_env = math.exp(-t * params["click_decay"])
        click = click_noise[i] * click_env

        hiss_env = math.exp(-t * params["noise_decay"])
        hiss = hiss_noise[i] * hiss_env

        value = (
            body * params["body_level"]
            + sub * params["sub_level"]
            + click * params["click_level"]
            + hiss * params["noise_level"]
        )
        samples.append(value * gain)
    samples = _tanh_drive(samples, params["drive"])
    samples = _apply_filter_chain(
        samples,
        sample_rate,
        [("highpass", 28.0, 0.5, 0.0), ("peaking", 110.0, 1.0, 2.0)],
    )
    _apply_fade(samples, sample_rate, 0.02)
    return _normalize(samples)


def _snare_engine(flavor: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    base_params = {
        "tone_freq": 190.0,
        "tone_ratio": 2.7,
        "tone_decay": 9.0,
        "noise_decay": 18.0,
        "snap_mix": 0.6,
        "drive": 1.4,
        "body_mix": 0.4,
    }
    if flavor == "electric":
        base_params.update(
            {
                "tone_freq": 220.0,
                "tone_ratio": 2.2,
                "tone_decay": 6.5,
                "noise_decay": 20.0,
                "snap_mix": 0.72,
                "drive": 1.9,
            }
        )
    params = {**base_params, **kwargs.get("params", {})}

    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=1.1)
    tone_phase = rng.uniform(0, TAU)
    ring_phase = rng.uniform(0, TAU)
    noise_filter = _make_biquad(sample_rate, "bandpass", 3600.0, q=3.5)
    body_filter = _make_biquad(sample_rate, "bandpass", params["tone_freq"] * 2, q=1.2)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        freq = params["tone_freq"] * (1 + 0.12 * math.exp(-t * 14.0))
        tone_phase += TAU * freq / sample_rate
        ring_phase += TAU * (freq * params["tone_ratio"]) / sample_rate
        tone = math.sin(tone_phase) * math.exp(-t * params["tone_decay"])
        ring = math.sin(ring_phase + 0.3) * math.exp(-t * (params["tone_decay"] * 0.8))

        white = rng.uniform(-1.0, 1.0)
        snap = noise_filter.process(white) * math.exp(-t * params["noise_decay"])
        shell = body_filter.process(white) * math.exp(-t * (params["noise_decay"] * 0.35))

        transient = 0.0
        if i < int(sample_rate * 0.0025):
            transient = rng.uniform(-1.0, 1.0) * (1.0 - i / (sample_rate * 0.0025))

        value = (
            (tone * params["body_mix"] + ring * (1 - params["body_mix"])) * (1 - params["snap_mix"])
            + snap * params["snap_mix"]
            + shell * 0.15
            + transient * 0.4
        )
        samples.append(value * gain)
    samples = _tanh_drive(samples, params["drive"])
    samples = _apply_filter_chain(samples, sample_rate, [("highpass", 140.0, 0.7, 0.0)])
    _apply_fade(samples, sample_rate, 0.02)
    return _normalize(samples)


def _side_stick(**kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.9)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        click = math.exp(-t * 220.0)
        tone = math.sin(TAU * 1200 * t + rng.uniform(-0.2, 0.2)) * math.exp(-t * 60.0)
        noise = rng.uniform(-1.0, 1.0) * math.exp(-t * 160.0)
        samples.append((tone * 0.5 + noise * 0.5) * click * gain)
    _apply_fade(samples, sample_rate, 0.008)
    return _normalize(samples)


def _clap(**kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=1.0)
    delays = [0.0, 0.012, 0.024, 0.039]
    trail_filter = _make_biquad(sample_rate, "highpass", 1800.0, q=0.7)
    samples = [0.0] * total
    for delay in delays:
        offset = int(delay * sample_rate)
        for i in range(offset, total):
            t = (i - offset) / sample_rate
            env = math.exp(-t * 32.0)
            burst = trail_filter.process(rng.uniform(-1.0, 1.0)) * env * gain
            samples[i] += burst
    samples = _apply_filter_chain(samples, sample_rate, [("highpass", 600.0, 0.8, 0.0)])
    _apply_fade(samples, sample_rate, 0.04)
    return _normalize(samples)


def _tom_engine(**kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    params = {
        "freq": 160.0,
        "sweep": 2.7,
        "decay": 7.0,
        "noise_level": 0.07,
        "detune": 0.01,
    }
    params.update(kwargs.get("params", {}))

    total = max(1, int(sample_rate * duration))
    rng = random.Random(kwargs["seed"])
    gain = _velocity_gain(velocity, curve=1.05)
    noise_buf = _colored_noise(total, rng, 0.6)
    samples: Signal = []
    tone_phase = rng.uniform(0, TAU)
    overtone_phase = rng.uniform(0, TAU)
    for i in range(total):
        t = i / sample_rate
        freq = _pitch_curve(t, params["freq"], params["freq"] * 0.72, params["sweep"])
        tone_phase += TAU * freq / sample_rate
        overtone_phase += TAU * (freq * 1.52) / sample_rate
        tone = math.sin(tone_phase)
        tone += 0.4 * math.sin(overtone_phase + 0.2)
        tone *= math.exp(-t * params["decay"])
        noise = noise_buf[i] * math.exp(-t * (params["decay"] * 2.2))
        value = tone * (1 - params["noise_level"]) + noise * params["noise_level"]
        samples.append(value * gain)
    samples = _apply_filter_chain(samples, sample_rate, [("highpass", 60.0, 0.7, 0.0)])
    _apply_fade(samples, sample_rate, 0.02)
    return _normalize(samples)


def _hat_engine(mode: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.85)
    params = {
        "decay": 36.0 if mode == "closed" else 9.5,
        "tone_mix": 0.45,
        "partials": [3200, 4700, 6200, 7800, 11000, 14000],
    }
    params.update(kwargs.get("params", {}))
    tone_mix = params.get("tone_mix", 0.45)
    partials = params.get("partials", [])
    states = [
        {
            "freq": float(freq) * (1 + 0.01 * rng.uniform(-1, 1)),
            "phase": rng.uniform(0, TAU),
        }
        for freq in partials
    ]
    noise_filter = _make_biquad(sample_rate, "highpass", 6500.0, q=0.85)
    tone_filter = _make_biquad(sample_rate, "bandpass", 9000.0, q=2.2)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        env = math.exp(-t * params["decay"])
        tone = 0.0
        for state in states:
            state["phase"] += TAU * state["freq"] / sample_rate
            tone += math.sin(state["phase"])
        tone = tone_filter.process(tone / max(1, len(states)))
        noise = noise_filter.process(rng.uniform(-1.0, 1.0))
        sample = (tone * tone_mix + noise * (1 - tone_mix)) * env * gain
        samples.append(sample)
    fade = 0.015 if mode == "closed" else 0.08
    _apply_fade(samples, sample_rate, fade)
    return _normalize(samples)


def _cymbal_engine(profile: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    catalog = {
        "crash": {
            "decay": 2.8,
            "noise_mix": 0.52,
            "partials": [(420, 0.6), (650, 0.45), (930, 0.35), (1500, 0.25), (2200, 0.2)],
        },
        "crash_bright": {
            "decay": 2.4,
            "noise_mix": 0.48,
            "partials": [(520, 0.6), (780, 0.5), (1100, 0.4), (1800, 0.25), (2600, 0.18)],
        },
        "china": {
            "decay": 2.2,
            "noise_mix": 0.45,
            "partials": [(360, 0.55), (540, 0.5), (820, 0.35), (1250, 0.28), (2100, 0.18)],
        },
        "splash": {
            "decay": 4.2,
            "noise_mix": 0.35,
            "partials": [(600, 0.4), (900, 0.3), (1400, 0.22), (2000, 0.18)],
        },
        "ride": {
            "decay": 1.7,
            "noise_mix": 0.4,
            "partials": [(450, 0.5), (720, 0.45), (1020, 0.4), (1520, 0.3)],
            "bell": (1400, 0.35),
        },
        "ride_bright": {
            "decay": 1.5,
            "noise_mix": 0.36,
            "partials": [(520, 0.46), (820, 0.4), (1180, 0.33), (1700, 0.25)],
            "bell": (1500, 0.38),
        },
        "ride_bell": {
            "decay": 3.6,
            "noise_mix": 0.18,
            "partials": [(860, 0.65), (1280, 0.5), (1720, 0.35), (2140, 0.28), (2800, 0.18)],
            "bell": (1280, 0.45),
        },
    }
    params = {**catalog.get(profile, catalog["crash"]), **kwargs.get("params", {})}
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.9)
    bell_phase = rng.uniform(0, TAU)
    partial_states = [
        {
            "freq": freq * (1 + 0.02 * rng.uniform(-1, 1)),
            "weight": weight,
            "phase": rng.uniform(0, TAU),
        }
        for freq, weight in params["partials"]
    ]
    noise_filter = _make_biquad(sample_rate, "highpass", 4800.0, q=0.9)
    shimmer_filter = _make_biquad(sample_rate, "bandpass", 9000.0, q=2.5)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        env = math.exp(-t * params["decay"])
        tone = 0.0
        for state in partial_states:
            wobble = 1 + 0.0015 * math.sin(2 * math.pi * 4.0 * t)
            state["phase"] += TAU * state["freq"] * wobble / sample_rate
            tone += math.sin(state["phase"]) * state["weight"]
        tone = shimmer_filter.process(tone)
        if "bell" in params:
            bell_freq, bell_weight = params["bell"]
            bell_phase += TAU * bell_freq / sample_rate
            tone += math.sin(bell_phase) * bell_weight * math.exp(-t * 6.2)
        noise = noise_filter.process(rng.uniform(-1.0, 1.0))
        sample = (tone * (1 - params["noise_mix"]) + noise * params["noise_mix"]) * env * gain
        samples.append(sample)
    _apply_fade(samples, sample_rate, 0.18)
    return _normalize(samples)


def _cowbell_engine(**kwargs: Any) -> Signal:
    params = {"decay": 6.0, **kwargs.get("params", {})}
    return _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=560.0,
        partials=[(1.0, 0.7, 1.0), (1.35, 0.55, 0.8), (2.9, 0.35, 1.6)],
        decay=params["decay"],
        noise_level=0.02,
        detune=0.01,
        seed=kwargs["seed"],
    )


def _noise_shaker(style: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.85)
    rate = 10.0 if style == "maracas" else 7.5
    decay = 8.0 if style == "maracas" else 9.5
    jitter = 0.003
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        burst = max(0.0, math.sin(TAU * rate * (t + rng.uniform(-jitter, jitter))))
        noise = rng.uniform(-1.0, 1.0) * burst
        samples.append(noise * math.exp(-t * decay) * gain)
    _apply_fade(samples, sample_rate, 0.05)
    return _normalize(samples)


def _tambourine(**kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.95)
    jitter_filter = _make_biquad(sample_rate, "highpass", 3200.0, q=0.9)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        env = math.exp(-t * 11.0)
        burst = jitter_filter.process(rng.uniform(-1.0, 1.0) + 0.3 * math.sin(TAU * 80 * t))
        samples.append(burst * env * gain)
    _apply_fade(samples, sample_rate, 0.05)
    return _normalize(samples)


def _whistle_engine(style: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.9)
    short = style == "short"
    params = {
        "freq_start": 1600.0 if short else 1400.0,
        "freq_end": 2300.0 if short else 2600.0,
        "gliss": 5.0 if short else 2.5,
        "vibrato_rate": 5.0,
        "vibrato_depth": 0.015 if short else 0.02,
        "noise_mix": 0.12,
        "attack": 0.005,
        "decay": 11.0 if short else 5.5,
    }
    params.update(kwargs.get("params", {}))
    phase = rng.uniform(0, TAU)
    noise_filter = _make_biquad(sample_rate, "highpass", 1200.0, q=0.9)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        freq = _pitch_curve(t, params["freq_start"], params["freq_end"], params["gliss"])
        vibrato = 1 + params["vibrato_depth"] * math.sin(TAU * params["vibrato_rate"] * t)
        phase += TAU * freq * vibrato / sample_rate
        tone = math.sin(phase) + 0.2 * math.sin(2 * phase + 0.3)
        breath = noise_filter.process(rng.uniform(-1.0, 1.0))
        env = _ad_env(t, params["attack"], params["decay"])
        value = tone * (1 - params["noise_mix"]) + breath * params["noise_mix"]
        samples.append(value * env * gain)
    _apply_fade(samples, sample_rate, 0.04)
    return _normalize(samples)


def _guiro_engine(style: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.85)
    short = style == "short"
    strokes = 1 if short else 2
    stroke_len = 0.12 if short else 0.18
    interval = 0.14
    scratch_filter = _make_biquad(sample_rate, "bandpass", 2600.0, q=2.5)
    samples: Signal = []
    stroke_times = [n * interval for n in range(strokes)]
    for i in range(total):
        t = i / sample_rate
        value = 0.0
        for start in stroke_times:
            if t < start or t > start + stroke_len:
                continue
            local = (t - start) / stroke_len
            ridge = math.sin(TAU * 22.0 * local)
            noise = scratch_filter.process(rng.uniform(-1.0, 1.0))
            env = (1 - local) * math.exp(-local * 5.0)
            value += (noise * 0.7 + ridge * 0.3) * env
        samples.append(value * gain)
    _apply_fade(samples, sample_rate, 0.03)
    return _normalize(samples)


def _cuica_engine(state: str, **kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.9)
    mute = state == "mute"
    params = {
        "freq_start": 240.0 if mute else 320.0,
        "freq_end": 160.0 if mute else 260.0,
        "sweep": 4.5 if mute else 3.2,
        "noise_level": 0.25 if mute else 0.35,
    }
    params.update(kwargs.get("params", {}))
    phase = rng.uniform(0, TAU)
    wobble_phase = rng.uniform(0, TAU)
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        freq = _pitch_curve(t, params["freq_start"], params["freq_end"], params["sweep"])
        wobble_phase += TAU * 9.0 / sample_rate
        vibrato = 1 + 0.05 * math.sin(wobble_phase)
        phase += TAU * freq * vibrato / sample_rate
        tone = math.sin(phase) + 0.35 * math.sin(2 * phase + 0.6)
        noise = rng.uniform(-1.0, 1.0)
        env = math.exp(-t * (6.5 if mute else 4.2))
        value = tone * (1 - params["noise_level"]) + noise * params["noise_level"]
        samples.append(value * env * gain)
    _apply_fade(samples, sample_rate, 0.05)
    return _normalize(samples)


def _bongo_engine(low: bool, **kwargs: Any) -> Signal:
    freq = 280 if low else 420
    params = {"freq": freq, "decay": 9.0, "noise_level": 0.04, **kwargs.get("params", {})}
    kwargs = {**kwargs, "params": params}
    return _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=params["freq"],
        partials=[(1.0, 0.9), (1.5, 0.45), (1.9, 0.3)],
        decay=params["decay"],
        noise_level=params["noise_level"],
        seed=kwargs["seed"],
    )


def _conga_engine(kind: str, **kwargs: Any) -> Signal:
    base = 200 if kind == "low" else 300
    mute = "mute" in kind
    params = {"freq": base, **kwargs.get("params", {})}
    samples = _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=params["freq"],
        partials=[(1.0, 0.9), (1.6, 0.5), (2.2, 0.25)],
        decay=10.0 if not mute else 18.0,
        noise_level=0.06 if not mute else 0.02,
        noise_color=0.4,
        seed=kwargs["seed"],
    )
    if mute:
        _apply_fade(samples, kwargs["sample_rate"], 0.05)
    return _normalize(samples)


def _timbale_engine(low: bool, **kwargs: Any) -> Signal:
    freq = 310 if low else 400
    samples = _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=freq,
        partials=[(1.0, 1.0), (1.7, 0.7), (2.5, 0.4), (3.1, 0.25)],
        decay=6.0,
        noise_level=0.05,
        seed=kwargs["seed"],
    )
    return _normalize(samples)


def _agogo_engine(low: bool, **kwargs: Any) -> Signal:
    base = 620 if low else 830
    samples = _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=base,
        partials=[(1.0, 0.9), (2.1, 0.4), (3.4, 0.2)],
        decay=12.0,
        noise_level=0.02,
        seed=kwargs["seed"],
    )
    return _normalize(samples)


def _woodblock_engine(low: bool, **kwargs: Any) -> Signal:
    base = 720 if low else 980
    samples = _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=base,
        partials=[(1.0, 1.0), (2.2, 0.5), (3.7, 0.2)],
        decay=20.0,
        noise_level=0.01,
        seed=kwargs["seed"],
    )
    _apply_fade(samples, kwargs["sample_rate"], 0.04)
    return _normalize(samples)


def _claves(**kwargs: Any) -> Signal:
    samples = _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=1500.0,
        partials=[(1.0, 1.0), (1.8, 0.4), (2.8, 0.2)],
        decay=25.0,
        noise_level=0.02,
        seed=kwargs["seed"],
    )
    _apply_fade(samples, kwargs["sample_rate"], 0.03)
    return _normalize(samples)


def _triangle_engine(open_state: str, **kwargs: Any) -> Signal:
    decay = 12.0 if open_state == "open" else 28.0
    samples = _modal_stack(
        sample_rate=kwargs["sample_rate"],
        duration=kwargs["duration"],
        velocity=kwargs["velocity"],
        base_freq=2200.0,
        partials=[(1.0, 1.0), (2.5, 0.4), (5.0, 0.25)],
        decay=decay,
        noise_level=0.01,
        seed=kwargs["seed"],
    )
    if open_state == "mute":
        _apply_fade(samples, kwargs["sample_rate"], 0.04)
    return _normalize(samples)


def _vibraslap(**kwargs: Any) -> Signal:
    sample_rate = kwargs["sample_rate"]
    duration = kwargs["duration"]
    velocity = kwargs["velocity"]
    rng = random.Random(kwargs["seed"])
    total = max(1, int(sample_rate * duration))
    gain = _velocity_gain(velocity, curve=0.9)
    rattles = 6
    hits = [rng.uniform(0.0, duration * 0.6) for _ in range(rattles)]
    hits.sort()
    samples: Signal = []
    for i in range(total):
        t = i / sample_rate
        value = 0.0
        for hit in hits:
            delta = max(0.0, t - hit)
            env = math.exp(-delta * 18.0)
            value += rng.uniform(-1.0, 1.0) * env
        pendulum = math.sin(TAU * 3.0 * t) * math.exp(-t * 2.0)
        samples.append((value * 0.4 + pendulum * 0.6) * gain)
    _apply_fade(samples, sample_rate, 0.1)
    return _normalize(samples)


# ---------------------------------------------------------------------------
# Recipe map
# ---------------------------------------------------------------------------


_RECIPES: Dict[str, Callable[..., Signal]] = {
    "analog_kick": lambda **kw: _kick_engine("modern", **kw),
    "analog_kick_acoustic": lambda **kw: _kick_engine("acoustic", **kw),
    "analog_snare": lambda **kw: _snare_engine("hybrid", **kw),
    "analog_snare_electric": lambda **kw: _snare_engine("electric", **kw),
    "analog_side_stick": _side_stick,
    "analog_clap": _clap,
    "analog_tom": _tom_engine,
    "analog_hat_closed": lambda **kw: _hat_engine("closed", **kw),
    "analog_hat_open": lambda **kw: _hat_engine("open", **kw),
    "analog_hat_pedal": lambda **kw: _hat_engine("pedal", **kw),
    "analog_crash": lambda **kw: _cymbal_engine("crash", **kw),
    "analog_crash_bright": lambda **kw: _cymbal_engine("crash_bright", **kw),
    "analog_china": lambda **kw: _cymbal_engine("china", **kw),
    "analog_splash": lambda **kw: _cymbal_engine("splash", **kw),
    "analog_ride": lambda **kw: _cymbal_engine("ride", **kw),
    "analog_ride_bright": lambda **kw: _cymbal_engine("ride_bright", **kw),
    "analog_ride_bell": lambda **kw: _cymbal_engine("ride_bell", **kw),
    "analog_cowbell": _cowbell_engine,
    "analog_shaker": lambda **kw: _noise_shaker("maracas", **kw),
    "analog_cabasa": lambda **kw: _noise_shaker("cabasa", **kw),
    "analog_tambourine": _tambourine,
    "analog_whistle_short": lambda **kw: _whistle_engine("short", **kw),
    "analog_whistle_long": lambda **kw: _whistle_engine("long", **kw),
    "analog_guiro_short": lambda **kw: _guiro_engine("short", **kw),
    "analog_guiro_long": lambda **kw: _guiro_engine("long", **kw),
    "analog_cuica_mute": lambda **kw: _cuica_engine("mute", **kw),
    "analog_cuica_open": lambda **kw: _cuica_engine("open", **kw),
    "analog_bongo_low": lambda **kw: _bongo_engine(True, **kw),
    "analog_bongo_high": lambda **kw: _bongo_engine(False, **kw),
    "analog_conga_low": lambda **kw: _conga_engine("low", **kw),
    "analog_conga_high": lambda **kw: _conga_engine("high", **kw),
    "analog_conga_low_mute": lambda **kw: _conga_engine("low_mute", **kw),
    "analog_conga_high_mute": lambda **kw: _conga_engine("high_mute", **kw),
    "analog_timbale_low": lambda **kw: _timbale_engine(True, **kw),
    "analog_timbale_high": lambda **kw: _timbale_engine(False, **kw),
    "analog_agogo_low": lambda **kw: _agogo_engine(True, **kw),
    "analog_agogo_high": lambda **kw: _agogo_engine(False, **kw),
    "analog_woodblock_low": lambda **kw: _woodblock_engine(True, **kw),
    "analog_woodblock_high": lambda **kw: _woodblock_engine(False, **kw),
    "analog_claves": _claves,
    "analog_triangle_open": lambda **kw: _triangle_engine("open", **kw),
    "analog_triangle_closed": lambda **kw: _triangle_engine("closed", **kw),
    "analog_triangle_mute": lambda **kw: _triangle_engine("mute", **kw),
    "analog_vibraslap": _vibraslap,
}
