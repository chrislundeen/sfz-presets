# General MIDI Drum Map Reference

This document captures the General MIDI (GM) Level 1 percussion specification so every drum-oriented instrument in this workspace can be aligned with a consistent, velocity-aware map.

## Top-Level Rules

- **Channel 10** (1-based) is reserved for percussion in GM Level 1. MIDI channels 1–9 and 11–16 remain melodic.
- **Program changes are ignored** on channel 10; the pitch of each note number selects the instrument instead.
- **Velocity is meaningful** for every articulation. Fixed-velocity playback is allowed, but GM-compliant renderers should expose a dynamic range of at least 1–127.
- **Controller 4 (Foot Controller)** optionally modulates hi-hat openness; **CC7 (Volume)** and **CC10 (Pan)** behave as they do on melodic channels.
- **Pitch Bend** is undefined for percussion, though some engines use it for special effects (e.g., bending snares or tablas). Staying at the neutral position (value 8192) preserves compatibility.

## Velocity Guidance

| Velocity Range | Typical Meaning | Suggested Response |
| --- | --- | --- |
| 1–30 | Ghost notes / light taps | Minimal noise bed, attenuated sustain |
| 31–90 | Standard playing | Expose full tonal body, moderate drive |
| 91–115 | Accents | Introduce extra harmonics, transient bite |
| 116–127 | Rimshots / crashes / flams | Allow gentle saturation but avoid clipping |

Kits targeting realism often add round-robin layers per velocity band. When only a single sample exists, scale amplitude + filter brightness to keep the mapping believable.

## GM Level 1 Percussion Key Map (Channel 10)

> Notes outside 35–87 are undefined in GM Level 1; GM Level 2 and XG/GS variants add more articulations but remain backward compatible with this base map.

| MIDI Note # | Name | Category | Notes |
| --- | --- | --- | --- |
| 35 | Acoustic Bass Drum | Kick | Deep, wide transient; sometimes mapped to sub or felt kick |
| 36 | Bass Drum 1 | Kick | Tighter modern kick |
| 37 | Side Stick | Snare/Rim | Short rim-click, minimal overtone |
| 38 | Acoustic Snare | Snare/Rim | Primary snare voice |
| 39 | Hand Clap | Snare/Rim | Multi-slap noise layer |
| 40 | Electric Snare | Snare/Rim | Bright, fast-decay snare |
| 41 | Low Floor Tom | Tom | 16" style |
| 42 | Closed Hi-Hat | Cymbal | Foot-closed or tight stick |
| 43 | High Floor Tom | Tom | 14" |
| 44 | Pedal Hi-Hat | Cymbal | Triggered by foot controller |
| 45 | Low Tom | Tom | 13" |
| 46 | Open Hi-Hat | Cymbal | Sustain controlled by note-off/CC4 |
| 47 | Low-Mid Tom | Tom | 12" |
| 48 | Hi-Mid Tom | Tom | 11" |
| 49 | Crash Cymbal 1 | Cymbal | Primary crash |
| 50 | High Tom | Tom | 10" |
| 51 | Ride Cymbal 1 | Cymbal | Bow hit |
| 52 | Chinese Cymbal | Cymbal | Trashy accent |
| 53 | Ride Bell | Cymbal | Focused bell strike |
| 54 | Tambourine | Aux Perc | Jingle hit |
| 55 | Splash Cymbal | Cymbal | Fast crash |
| 56 | Cowbell | Aux Perc | Medium cowbell |
| 57 | Crash Cymbal 2 | Cymbal | Alternate crash |
| 58 | Vibraslap | Aux Perc | Long rattle |
| 59 | Ride Cymbal 2 | Cymbal | Alternate ride (often brighter) |
| 60 | High Bongo | Latin | Tuned high |
| 61 | Low Bongo | Latin | Tuned low |
| 62 | Mute High Conga | Latin | Palm-muted |
| 63 | Open High Conga | Latin | Open tone |
| 64 | Low Conga | Latin | Open tone |
| 65 | High Timbale | Latin | Stick hit |
| 66 | Low Timbale | Latin | Stick hit |
| 67 | High Agogo | Latin | Bell |
| 68 | Low Agogo | Latin | Bell |
| 69 | Cabasa | Aux Perc | Scraped bead cylinder |
| 70 | Maracas | Aux Perc | Pair shake |
| 71 | Short Whistle | Aux Perc | Single chirp |
| 72 | Long Whistle | Aux Perc | Sustained |
| 73 | Short Guiro | Aux Perc | Downstroke scrape |
| 74 | Long Guiro | Aux Perc | Upstroke scrape |
| 75 | Claves | Aux Perc | Wood clave |
| 76 | High Wood Block | Aux Perc | Small block |
| 77 | Low Wood Block | Aux Perc | Large block |
| 78 | Mute Cuica | Latin | Stick-damped cuica |
| 79 | Open Cuica | Latin | Open cuica |
| 80 | Mute Triangle | Aux Perc | Damped strike |
| 81 | Open Triangle | Aux Perc | Sustained |
| 82 | Shaker | Aux Perc | Generic shaker (sometimes "Jingle Bell" in GM2) |
| 83 | Jingle Bell | Aux Perc | Hand bell |
| 84 | Bell Tree | Aux Perc | Down-swept bell tree |
| 85 | Castanets | Aux Perc | Flamenco castanets |
| 86 | Mute Surdo | Latin | Damped low surdo |
| 87 | Open Surdo | Latin | Open surdo |

## Implementation Notes

- **Hi-Hat handling:** Note 42 (closed), 44 (pedal), and 46 (open) should be mutually exclusive via voice groups. CC4 (0–127) may further refine openness; values below ~40 typically force closed behavior.
- **Release tails:** Cymbals (49, 51, 52, 55, 57, 59) and sustained percussion (58, 81, 84) should use one-shot envelopes or generous release times to avoid abrupt truncation.
- **Layering:** GM does not mandate round robins, but multi-layering improves realism. Use the same note number so MIDI files remain portable.
- **Extended specs:** GM Level 2, Roland GS, and Yamaha XG add additional mappings (brush kits, electronic kits, etc.). Stick to the Level 1 table above to guarantee compatibility and treat any extras as optional overlays.

Keep this reference alongside `scripts/configs/analog_kit.json` so new kits, samplers, or exports remain fully GM-compliant.
