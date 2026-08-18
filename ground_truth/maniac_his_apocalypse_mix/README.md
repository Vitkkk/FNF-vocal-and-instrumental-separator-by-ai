# Ground Truth 001 — Maniac (His Apocalypse Mix)

## Purpose

This is the first supervised reference case for the FNF-specific separator.

Available reference material:

- full mix;
- associated instrumental render;
- leaked/publicly circulating vocal stem used only as a comparison target.

The audio files themselves are **not** stored in this repository.

## Observed metadata

All three source files were decoded to stereo 44.1 kHz float audio for analysis.

Approximate durations:

- full mix: ~415.09 s;
- instrumental: ~415.14 s;
- vocal reference: ~412.06 s.

The instrumental aligns closely with the mix after an offset of approximately **940 samples**, equivalent to **21.315 ms at 44.1 kHz**.

The offset is stable through large portions of the track, which indicates that the mix and instrumental are not primarily suffering from BPM drift or a global time-stretch mismatch.

## Important result

A simple model of the form:

```text
mix(t) = a * instrumental(t - delta_i) + b * vocals(t - delta_v)
```

does **not** explain the published files well enough.

The instrumental can correlate extremely well with corresponding regions of the mix after alignment, but subtracting it leaves a large residual that still contains instrumental material.

The vocal reference also does not behave like a stem that can simply be dropped sample-for-sample into the final uploaded mix. Across multiple windows, direct linear fitting gives a near-zero useful coefficient for the supplied vocal reference in many regions.

Therefore, for this example:

```text
mix != instrumental_upload + vocals_upload
```

in the strict sample-domain sense.

Possible reasons include different renders, mastering, compression/limiting, EQ, phase changes, stem revisions, or different export/encoding chains.

## Why this matters for training

This case proves that the separator must not be designed as only a clever subtraction engine.

The actual supervised task should be:

```text
input:  final FNF mix
output: estimated vocals
output: estimated instrumental
```

The supplied vocal stem acts as a target/reference, even when it cannot perfectly reconstruct the uploaded final mix through direct addition.

## Baseline experiments

### Baseline A — aligned subtraction

1. Estimate instrumental offset.
2. Correct global gain.
3. Subtract instrumental from mix.

Result: large instrumental leakage remains.

### Baseline B — adaptive spectral subtraction

1. Align instrumental.
2. Estimate frequency-dependent complex transfer function.
3. Compensate broad EQ/phase differences.
4. Estimate slowly varying gain.
5. Subtract predicted instrumental in the STFT domain.

Result: audible change and some attenuation, but instrumental leakage remains clearly present.

This means future evaluation must compare against the actual vocal target, not against how 'different from the mix' the residual sounds.

## Dataset identity

Internal ID:

```text
gt_001_maniac_his_apocalypse_mix
```

Recommended local layout:

```text
datasets/
  gt_001_maniac_his_apocalypse_mix/
    mix.wav
    instrumental.wav
    vocals_target.wav
    metadata.json
```

Do not commit copyrighted source audio unless redistribution rights are explicitly available.
