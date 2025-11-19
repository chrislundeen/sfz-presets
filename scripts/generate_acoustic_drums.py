#!/usr/bin/env python3
"""Render acoustic drum kit from JSON configuration.

This is the generation script for the physically-informed acoustic engine
(drumforge_acoustic).  It mirrors generate_drums.py but imports from the
new module so the existing analog kit is untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List

from drumforge_acoustic.synthesis import render_instrument


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate acoustic drum samples from config")
    parser.add_argument("config", type=Path, help="Path to kit configuration JSON")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render only the first N instruments (debug helper)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def float_to_pcm16(samples: Iterable[float]) -> bytes:
    buf = bytearray()
    for sample in samples:
        clamped = max(-1.0, min(1.0, sample))
        value = int(round(clamped * 32767))
        buf.extend(value.to_bytes(2, byteorder="little", signed=True))
    return bytes(buf)


def write_wav(path: Path, samples: List[float], sample_rate: int) -> None:
    ensure_dir(path.parent)
    pcm = float_to_pcm16(samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def render_instrument_variants(
    instrument: Dict[str, Any],
    *,
    velocities: List[int],
    default_round_robins: int,
    sample_rate: int,
    output_root: Path,
) -> List[Path]:
    rendered: List[Path] = []
    inst_velocities = instrument.get("velocities") or velocities
    round_robins = instrument.get("round_robins", default_round_robins)
    duration = instrument.get("duration_ms", 800) / 1000.0
    for velocity in inst_velocities:
        for rr_index in range(round_robins):
            samples = render_instrument(
                instrument["recipe"],
                sample_rate=sample_rate,
                duration=duration,
                velocity=velocity,
                round_robin_index=rr_index,
                params=instrument.get("params") or {},
            )
            filename = (
                f"{instrument['id']}_n{instrument['midi_note']}"
                f"_v{velocity}_rr{rr_index + 1}.wav"
            )
            outfile = output_root / instrument["id"] / filename
            write_wav(outfile, samples, sample_rate)
            rendered.append(outfile)
    return rendered


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    instruments = config.get("instruments", [])
    if args.limit is not None:
        instruments = instruments[: args.limit]
    sample_rate = config.get("sample_rate", 44100)
    velocities = config.get("default_velocities", [30, 60, 90, 110, 127])
    default_rounds = config.get("default_round_robins", 3)
    output_root = Path(config.get("output_root", "output/sfz/acoustic_kit/samples"))
    ensure_dir(output_root)

    generated_files: List[Path] = []
    for instrument in instruments:
        generated_files.extend(
            render_instrument_variants(
                instrument,
                velocities=velocities,
                default_round_robins=default_rounds,
                sample_rate=sample_rate,
                output_root=output_root,
            )
        )
        print(
            f"Rendered {instrument['id']} -> {len(generated_files)} total files so far"
        )

    print(f"\nFinished. {len(generated_files)} files written to {output_root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
