#!/usr/bin/env python3
"""Generate multi-cycle 24-bit / 48 kHz waveforms for an exact 440 Hz loop.

Each waveform instrument is self-contained in its own directory under output/sfz/:
    output/sfz/<name>/
        <name>.sfz
        samples/<name>_440hz_cycle.wav

A shared waveform library is also written to output/waveforms/ for use by
synth-style SFZ files.
"""
from __future__ import annotations

import math
import shutil
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, List

SAMPLE_RATE = 48_000
BIT_DEPTH = 24
TARGET_FREQ = 440.0
TARGET_FREQ_FRACTION = Fraction(440, 1)
SAMPLES_PER_CYCLE = Fraction(SAMPLE_RATE, 1) / TARGET_FREQ_FRACTION
MULTI_CYCLES = SAMPLES_PER_CYCLE.denominator
CYCLE_SAMPLES = SAMPLES_PER_CYCLE.numerator
NUM_SAMPLES = CYCLE_SAMPLES + 1  # duplicate endpoint to force-zero loop
LOOP_DURATION_SEC = CYCLE_SAMPLES / SAMPLE_RATE
ACTUAL_FREQ = SAMPLE_RATE * MULTI_CYCLES / (NUM_SAMPLES - 1)
MAX_AMPLITUDE = (1 << (BIT_DEPTH - 1)) - 1
OUTPUT_ROOT = Path("output")
SFZ_ROOT = OUTPUT_ROOT / "sfz"
WAVEFORM_LIB = OUTPUT_ROOT / "waveforms"

WaveFunction = Callable[[int, int], float]


@dataclass(frozen=True)
class WaveRecipe:
    name: str
    wav_filename: str
    sfz_name: str
    generator: WaveFunction
    description: str


def _unit_phase(n: int, total: int) -> float:
    """Return normalized phase for the whole buffer [0,1]."""
    if total <= 1:
        return 0.0
    return n / float(total - 1)


def _cycle_phase(n: int, total: int) -> float:
    """Return phase measured in cycles across the full buffer."""
    return _unit_phase(n, total) * MULTI_CYCLES


def sine_wave(n: int, total: int) -> float:
    return math.sin(2 * math.pi * _cycle_phase(n, total))


def square_wave(n: int, total: int) -> float:
    val = math.sin(2 * math.pi * _cycle_phase(n, total))
    return 1.0 if val >= 0 else -1.0


def triangle_wave(n: int, total: int) -> float:
    return (2.0 / math.pi) * math.asin(math.sin(2 * math.pi * _cycle_phase(n, total)))


def saw_wave(n: int, total: int) -> float:
    phase = _cycle_phase(n, total) % 1.0
    return 2.0 * (phase - 0.5)


def complex_even(n: int, total: int) -> float:
    phase = _cycle_phase(n, total)
    partials = [2, 4, 6, 8, 10]
    return sum(math.sin(2 * math.pi * phase * p) / p for p in partials)


def complex_odd(n: int, total: int) -> float:
    phase = _cycle_phase(n, total)
    partials = [1, 3, 5, 7, 9]
    return sum((1 / p) * math.sin(2 * math.pi * phase * p) for p in partials)


def complex_detuned(n: int, total: int) -> float:
    phase = _cycle_phase(n, total)
    offsets = [0.0, 0.01, -0.015, 0.027]
    weights = [0.6, 0.25, 0.1, 0.05]
    return sum(
        w * math.sin(2 * math.pi * (phase * (1 + detune)))
        for w, detune in zip(weights, offsets)
    )


def complex_cluster(n: int, total: int) -> float:
    phase = _cycle_phase(n, total)
    angles = [0.0, math.pi / 5, math.pi / 2, math.pi * 0.9]
    weights = [0.4, 0.3, 0.2, 0.1]
    return sum(w * math.sin(2 * math.pi * phase + angle) for w, angle in zip(weights, angles))


def complex_formant(n: int, total: int) -> float:
    phase = _cycle_phase(n, total)
    unit_phase = _unit_phase(n, total)
    partials = [(1, 1.0), (5, 0.6), (7, 0.3), (11, 0.25), (13, 0.18)]
    envelope = math.sin(math.pi * unit_phase) ** 2  # accentuate middle
    return envelope * sum(weight * math.sin(2 * math.pi * phase * harmonic) for harmonic, weight in partials)


def _soft_clip(value: float, drive: float) -> float:
    return math.tanh(value * drive)


def _pseudo_noise(index: int) -> float:
    return math.sin(2 * math.pi * (index * 0.61803398875 + 0.13))


def complex_even_saw_warm(n: int, total: int) -> float:
    phase = _cycle_phase(n, total)
    unit_phase = _unit_phase(n, total)
    partials = [2, 4, 6, 8, 10, 12]
    base = sum((1 / p) * math.sin(2 * math.pi * phase * p) for p in partials)
    drive = 1.4 + 0.6 * math.sin(2 * math.pi * unit_phase)
    blend = 0.65 + 0.25 * math.sin(2 * math.pi * unit_phase * 3)
    shaped = _soft_clip(base, drive)
    return blend * shaped + (1 - blend) * base


def complex_even_saw_warm_noise(n: int, total: int) -> float:
    base = complex_even_saw_warm(n, total)
    unit_phase = _unit_phase(n, total)
    noise_amount = 0.02 + 0.015 * math.sin(2 * math.pi * unit_phase * 2.3)
    noise = noise_amount * _pseudo_noise(n)
    return base + noise


def normalize_cycle(samples: List[float]) -> List[float]:
    if not samples:
        return samples
    samples[0] = 0.0
    samples[-1] = 0.0
    peak = max(abs(s) for s in samples) or 1.0
    scale = 0.98 / peak  # leave headroom to avoid clipping
    return [s * scale for s in samples]


def to_24bit_pcm(value: float) -> bytes:
    clamped = max(-0.999999, min(0.999999, value))
    int_val = int(round(clamped * MAX_AMPLITUDE))
    return int_val.to_bytes(3, byteorder="little", signed=True)


def write_wave(path: Path, samples: List[float]) -> None:
    pcm = bytearray()
    for sample in samples:
        pcm.extend(to_24bit_pcm(sample))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(3)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)


def generate_samples(generator: WaveFunction) -> List[float]:
    raw = [generator(i, NUM_SAMPLES) for i in range(NUM_SAMPLES)]
    return normalize_cycle(raw)


RECIPES = [
    WaveRecipe("sine", "sine_440hz_cycle.wav", "sine", sine_wave, "Pure sine wave"),
    WaveRecipe("square", "square_440hz_cycle.wav", "square", square_wave, "Classic square wave"),
    WaveRecipe("triangle", "triangle_440hz_cycle.wav", "triangle", triangle_wave, "Triangle wave"),
    WaveRecipe("saw", "saw_440hz_cycle.wav", "saw", saw_wave, "Rising sawtooth"),
    WaveRecipe("complex_even", "complex_even_440hz_cycle.wav", "complex_even", complex_even, "Even harmonic stack"),
    WaveRecipe("complex_odd", "complex_odd_440hz_cycle.wav", "complex_odd", complex_odd, "Odd harmonic emphasis"),
    WaveRecipe("complex_detuned", "complex_detuned_440hz_cycle.wav", "complex_detuned", complex_detuned, "Detuned partial blend"),
    WaveRecipe("complex_cluster", "complex_cluster_440hz_cycle.wav", "complex_cluster", complex_cluster, "Cluster with phase offsets"),
    WaveRecipe("complex_formant", "complex_formant_440hz_cycle.wav", "complex_formant", complex_formant, "Formant-style envelope"),
    WaveRecipe("complex_even_saw_warm", "complex_even_saw_warm_440hz_cycle.wav", "complex_even_saw_warm", complex_even_saw_warm, "Even saw with gentle drive"),
    WaveRecipe(
        "complex_even_saw_warm_noise",
        "complex_even_saw_warm_noise_440hz_cycle.wav",
        "complex_even_saw_warm_noise",
        complex_even_saw_warm_noise,
        "Warm saw with soft noise",
    ),
]


def generate_sfz(sfz_path: Path, wav_filename: str) -> None:
    """Write a minimal single-waveform SFZ file."""
    content = (
        "<group>\n"
        "lovel=0\n"
        "hivel=127\n"
        "<region>\n"
        f"sample=samples/{wav_filename}\n"
        "lokey=0\n"
        "hikey=127\n"
        "pitch_keycenter=69\n"
        "loop_mode=loop_continuous\n"
        "loop_start=0\n"
        f"loop_end={NUM_SAMPLES - 2}\n"
    )
    sfz_path.parent.mkdir(parents=True, exist_ok=True)
    sfz_path.write_text(content, encoding="utf-8")


def main() -> None:
    print(f"Target frequency: {TARGET_FREQ} Hz")
    print(
        "Loop geometry: "
        f"{MULTI_CYCLES} cycles across {NUM_SAMPLES - 1} samples "
        f"({LOOP_DURATION_SEC * 1_000:.3f} ms)"
    )
    print(f"Actual fundamental: {ACTUAL_FREQ:.6f} Hz")
    print(f"Writing self-contained instruments to {SFZ_ROOT.resolve()}")
    WAVEFORM_LIB.mkdir(parents=True, exist_ok=True)
    for recipe in RECIPES:
        samples = generate_samples(recipe.generator)
        inst_dir = SFZ_ROOT / recipe.sfz_name
        wav_path = inst_dir / "samples" / recipe.wav_filename
        sfz_path = inst_dir / f"{recipe.sfz_name}.sfz"
        write_wave(wav_path, samples)
        generate_sfz(sfz_path, recipe.wav_filename)
        # Also copy to shared waveform library
        shutil.copy2(wav_path, WAVEFORM_LIB / recipe.wav_filename)
        print(f" • {inst_dir}/ ({recipe.description})")
    print(f"Waveform library updated: {WAVEFORM_LIB.resolve()}")


if __name__ == "__main__":
    main()
