# SFZ Creation Toolkit

A fully procedural audio toolkit that synthesizes sample-based instruments from scratch. Every WAV and SFZ file is generated deterministically by Python scripts — delete the entire `output/` directory and re-run to get it all back.

## Quickstart

```bash
# 1. Generate waveform instruments + shared waveform library
scripts/generate_wavs.py

# 2. Synthesize drum kit samples (300 WAVs)
scripts/generate_drums.py scripts/configs/analog_kit.json

# 3. Generate the drum kit SFZ mapping
scripts/make_sfz_from_config.py scripts/configs/analog_kit.json
```

Load any `.sfz` file from `output/sfz/` into an SFZ-compatible sampler (Sforzando, sfizz, etc.).

---

## Repository Layout

```
output/                          # All generated artifacts (fully reproducible)
  waveforms/                     #   Shared single-cycle waveform library
    sine_440hz_cycle.wav
    saw_440hz_cycle.wav
    ...
  sfz/                           #   Self-contained SFZ instruments
    sine/
      sine.sfz
      samples/
        sine_440hz_cycle.wav
    analog_kit/
      analog_kit.sfz
      samples/
        kick_deep/
        snare_acoustic/
        hihat_closed/
        ...
scripts/                         # Source code
  generate_wavs.py               #   Waveform renderer
  generate_drums.py              #   JSON-driven drum renderer
  make_sfz_from_config.py        #   SFZ mapping generator
  configs/
    analog_kit.json              #   Drum kit configuration
  drumforge/                     #   Synthesis engine library
    synthesis.py                 #     15 engines, 47 recipes
configs/
  analog_kit.json                # Root-level config copy
docs/
  GM-drums-spec.md               # General MIDI percussion reference
```

### Self-Contained Instruments

Every instrument under `output/sfz/` is a standalone directory containing the `.sfz` file and a `samples/` folder with all referenced WAVs. Copy any instrument folder to another machine and it will work without external dependencies.

### `output/waveforms/`

A flat directory containing all 11 standard single-cycle waveforms. Use these as building blocks when hand-crafting new synth-style SFZ patches — they can be referenced directly or copied into a new instrument's `samples/` folder.

---

## Scripts

### `generate_wavs.py` — Waveform Renderer

Synthesizes 11 single-cycle waveforms as 24-bit / 48 kHz mono WAVs, tuned to exactly 440 Hz.

Each waveform renders 11 complete cycles across 1,200 samples (25 ms). This multi-cycle approach is necessary because 48,000 ÷ 440 = 1200/11 — a single period cannot land on integer sample boundaries. The buffer starts and ends at zero for seamless looping.

For each waveform the script:
1. Writes a self-contained instrument to `output/sfz/<name>/` (SFZ + WAV).
2. Copies the WAV to `output/waveforms/` as a shared library.

#### Available Waveforms

| Waveform | Description |
|---|---|
| `sine` | Pure sine wave |
| `square` | Classic square wave |
| `triangle` | Triangle wave |
| `saw` | Rising sawtooth |
| `complex_even` | Even harmonic stack (partials 2, 4, 6, 8, 10) |
| `complex_odd` | Odd harmonic emphasis (partials 1, 3, 5, 7, 9) |
| `complex_detuned` | Weighted detuned partials for chorus-like width |
| `complex_cluster` | Phase-offset partial cluster |
| `complex_formant` | Formant-style envelope shaping higher harmonics |
| `complex_even_saw_warm` | Even harmonics with soft-clip saturation |
| `complex_even_saw_warm_noise` | Warm saw with pseudo-noise texture |

#### Extending

Add a `WaveRecipe` entry to the `RECIPES` list with a generator function `(n: int, total: int) -> float` that returns a sample value for position `n` out of `total`.

---

### `generate_drums.py` — Drum Kit Renderer

Reads a kit configuration JSON and renders one-shot WAV samples for every instrument × velocity × round-robin combination.

```bash
scripts/generate_drums.py scripts/configs/analog_kit.json
scripts/generate_drums.py scripts/configs/analog_kit.json --limit 5  # first 5 instruments only
```

Output format: 16-bit / 44.1 kHz mono WAVs.

Each instrument's samples land in a subdirectory named by instrument ID:
```
output/sfz/analog_kit/samples/<id>/<id>_n<note>_v<velocity>_rr<index>.wav
```

#### How rendering works

1. The JSON config defines instruments with a `recipe` name, MIDI note, duration, optional velocity/round-robin overrides, and synthesis `params`.
2. Each recipe maps to a synthesis engine in `drumforge/synthesis.py`.
3. A deterministic seed derived from `(recipe, velocity, rr_index)` ensures reproducible output.
4. Round-robin indices > 0 receive automatic micro-variation (amplitude jitter ±3.5%, spectral tilt, colored noise).

---

### `make_sfz_from_config.py` — SFZ Mapping Generator

Converts a drum kit JSON config into an SFZ file with correctly mapped velocity layers and round-robin sequencing.

```bash
scripts/make_sfz_from_config.py scripts/configs/analog_kit.json
scripts/make_sfz_from_config.py scripts/configs/analog_kit.json --output /path/to/custom.sfz
```

The generated SFZ uses `default_path=samples/` so it remains portable alongside its local `samples/` directory. Velocity zones are auto-derived from the config's velocity targets, and `seq_length`/`seq_position` opcodes handle round-robin cycling.

---

## Kit Configuration

Kit configs live at `scripts/configs/` (e.g., `analog_kit.json`).

### Top-Level Fields

| Field | Description | Default |
|---|---|---|
| `name` | Kit identifier | — |
| `sample_rate` | Render sample rate (Hz) | `44100` |
| `bit_depth` | Bits per sample | `16` |
| `output_root` | Where samples are written | `output/sfz/analog_kit/samples` |
| `default_velocities` | Velocity targets for layers | `[50, 90, 110, 127]` |
| `default_round_robins` | Fallback round-robin count | `1` |

### Instrument Fields

| Field | Description |
|---|---|
| `id` | Unique identifier, used in filenames and directory names |
| `display_name` | Human-readable name (used as `group_label` in SFZ) |
| `midi_note` | GM percussion note number (35–81) |
| `recipe` | Synthesis recipe key (see engine table below) |
| `duration_ms` | Render length in milliseconds |
| `velocities` | Optional per-instrument velocity overrides |
| `round_robins` | Optional per-instrument round-robin count (always odd: 1, 3, 5, or 7) |
| `params` | Recipe-specific synthesis parameters |

---

## Synthesis Engine — `drumforge/synthesis.py`

A 978-line pure-Python drum synthesizer with no external audio dependencies. Everything is built from oscillators, noise generators, envelopes, and filters.

### Engines (15)

| Engine | Instruments |
|---|---|
| `_kick_engine` | Acoustic and modern bass drums — pitch sweep + body resonance |
| `_snare_engine` | Hybrid and electric snares — tone oscillator + noise layer |
| `_side_stick` | Rim click — short tonal transient |
| `_clap` | Hand clap — multi-burst noise with comb filtering |
| `_tom_engine` | Floor toms through rack toms — tuned body with pitch sweep |
| `_hat_engine` | Closed, open, and pedal hi-hats — metallic partials with variable decay |
| `_cymbal_engine` | Crashes, rides, splashes, china — inharmonic partial clusters |
| `_cowbell_engine` | Cowbell — dual-frequency metallic tone |
| `_noise_shaker` | Maracas and cabasa — shaped noise bursts |
| `_tambourine` | Tambourine — jingle partials over noise |
| `_whistle_engine` | Short and long whistles — sine with vibrato |
| `_guiro_engine` | Short and long guiro — amplitude-modulated rasp |
| `_cuica_engine` | Mute and open cuica — pitch-gliding tone |
| `_bongo_engine` / `_conga_engine` / `_timbale_engine` | Latin hand drums |
| `_agogo_engine` / `_woodblock_engine` / `_triangle_engine` | Auxiliary metallic/wood percussion |

Plus standalone recipes: `_claves`, `_vibraslap`.

### Recipe Map (47 recipes)

Each recipe in the `_RECIPES` dict maps a string key (e.g., `"analog_kick"`) to an engine call with the appropriate profile/variant. The config JSON references these keys in the `recipe` field.

### Round-Robin Variation

Round-robin counts are always odd (1, 3, 5, or 7). For `rr_index > 0`, `_apply_round_robin_variation()` applies:
- Amplitude jitter: ±3.5% random gain
- Spectral tilt: subtle high-frequency roll-off or boost
- Colored noise injection: low-level textural variation

This ensures consecutive hits on the same instrument sound realistically different while remaining deterministic (same seed = same output).

---

## General MIDI Coverage

The Analog Kit maps the full GM Level 1 percussion spec (MIDI notes 35–81), covering:

- **Kicks** (35–36) — acoustic and modern
- **Snares/Rims** (37–40) — side stick, acoustic snare, hand clap, electric snare
- **Toms** (41, 43, 45, 47–48, 50) — six sizes from 16" floor to 10" rack
- **Hi-Hats** (42, 44, 46) — closed, pedal, open (mutually exclusive via voice groups)
- **Cymbals** (49, 51–53, 55, 57, 59) — crashes, rides, splash, china, ride bell
- **Latin Percussion** (60–68, 78–79) — bongos, congas, timbales, agogos, cuica
- **Auxiliary** (54, 56, 58, 69–77, 80–81) — tambourine, cowbell, cabasa, maracas, whistles, guiro, claves, wood blocks, triangle, vibraslap

See [docs/GM-drums-spec.md](docs/GM-drums-spec.md) for the full reference table with velocity guidance and implementation notes.

---

## Regenerating Everything

```bash
scripts/generate_wavs.py
scripts/generate_drums.py scripts/configs/analog_kit.json
scripts/make_sfz_from_config.py scripts/configs/analog_kit.json
```

The workflow is fully deterministic. Every run produces identical output for the same code and config. The entire `output/` directory can be deleted and rebuilt from scratch.

---

## Extending the Toolkit

### Add a new waveform
1. Write a generator function in `generate_wavs.py`.
2. Add a `WaveRecipe` to the `RECIPES` list.
3. Re-run `generate_wavs.py`.

### Add a new drum sound
1. Write or reuse an engine function in `drumforge/synthesis.py`.
2. Register it in the `_RECIPES` dict.
3. Add an instrument entry to your kit JSON config.
4. Re-run `generate_drums.py` and `make_sfz_from_config.py`.

### Create a new drum kit
1. Duplicate `scripts/configs/analog_kit.json`.
2. Adjust instrument selections, recipes, velocities, round-robin counts, and params.
3. Run both `generate_drums.py` and `make_sfz_from_config.py` with the new config.

---

## Requirements

- Python 3.13+ (no external packages required — stdlib only)
- Any SFZ-compatible sampler for playback
