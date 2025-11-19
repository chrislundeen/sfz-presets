#!/usr/bin/env python3
"""Generate SFZ mappings from a drum kit configuration JSON."""
from __future__ import annotations

import argparse
import json
from math import floor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_VELOCITIES = [50, 90, 110, 127]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an SFZ file from a kit config")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to kit JSON (e.g. scripts/configs/analog_kit.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination SFZ path (defaults to <config_name>.sfz in repo root)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def velocity_ranges(values: Sequence[int]) -> List[tuple[int, int, int]]:
    values = sorted(values)
    ranges: List[tuple[int, int, int]] = []
    low = 1
    for idx, val in enumerate(values):
        if idx + 1 < len(values):
            next_val = values[idx + 1]
            hi = int(floor((val + next_val) / 2))
        else:
            hi = 127
        ranges.append((low, hi, val))
        low = hi + 1
    if ranges:
        ranges[-1] = (ranges[-1][0], 127, ranges[-1][2])
    return ranges


def emit_sfz(
    config: Dict[str, Any],
    *,
    output_path: Path,
    default_path: str,
) -> None:
    default_vels: Sequence[int] = config.get("default_velocities", DEFAULT_VELOCITIES)
    default_rr = config.get("default_round_robins", 1)

    lines: List[str] = []
    lines.append("<control>")
    lines.append(f"default_path={default_path}")
    lines.append("")
    lines.append("<global>")
    lines.append("loop_mode=one_shot")
    lines.append("ampeg_release=0.05")
    lines.append("")

    for instrument in config.get("instruments", []):
        note = instrument["midi_note"]
        group_label = instrument.get("display_name", instrument["id"])
        velocities = instrument.get("velocities") or default_vels
        vel_ranges = velocity_ranges(velocities)
        round_robins = instrument.get("round_robins", default_rr)

        lines.append("<group>")
        lines.append(f"group_label={group_label}")
        lines.append(f"lokey={note}")
        lines.append(f"hikey={note}")
        lines.append(f"pitch_keycenter={note}")
        lines.append("")

        for lovel, hivel, vel in vel_ranges:
            for rr_idx in range(round_robins):
                sample_rel = (
                    f"{instrument['id']}/{instrument['id']}_n{note}_v{vel}_rr{rr_idx + 1}.wav"
                )
                lines.append("<region>")
                lines.append(f"sample={sample_rel}")
                lines.append(f"lokey={note}")
                lines.append(f"hikey={note}")
                lines.append(f"pitch_keycenter={note}")
                lines.append(f"lovel={lovel}")
                lines.append(f"hivel={hivel}")
                if round_robins > 1:
                    lines.append(f"seq_length={round_robins}")
                    lines.append(f"seq_position={rr_idx + 1}")
                lines.append("")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config_name = Path(args.config).stem
    default_output = Path("output/sfz") / config_name / f"{config_name}.sfz"
    output = args.output or default_output
    output = output if output.is_absolute() else Path.cwd() / output
    default_path = "samples/"

    emit_sfz(config, output_path=output, default_path=default_path)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
