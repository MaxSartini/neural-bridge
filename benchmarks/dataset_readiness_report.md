# Dataset Readiness Report: Downloadable Affect/Emotion Benchmarks

Generated: 2026-06-07

## Decision

The next clean benchmark target is **VEATIC**.

Rationale:

- It provides actual videos and aligned continuous valence/arousal ratings.
- Its repository defines a direct local folder contract: `video/${video_id}.mp4` and `rating_averaged/${video_id}_valence.csv` / `${video_id}_arousal.csv`.
- It has an explicit train/test protocol: first 70% of frames for training, last 30% for testing.
- It is smaller and cleaner than LIRIS-ACCEDE for an immediate adapter.
- It is more directly downloadable than LIRIS-ACCEDE, which requires EULA/email approval.
- CASE is scientifically attractive but its cited direct archive URL currently returns 404, and the paper says videos are not directly included because of copyright.

No large dataset was downloaded.

## Readiness Matrix

| Dataset | Media availability | Annotation availability | Access/gating | Label types | Temporal resolution | Stimuli | Estimated storage | Direct media/label alignment | TRIBE video/audio/text support | Temporal benchmark support | Best use |
|---|---|---|---|---|---:|---:|---:|---|---|---|---|
| LIRIS-ACCEDE | Yes after approval; video clips/movies available through dataset access | Yes; valence/arousal, fear for some collections, continuous and discrete subsets | EULA required; free-email requests refused; download link sent by email | Valence, arousal, fear, violence/classes depending subset | Discrete clips; continuous whole-movie labels; MediaEval 2017 10s/5s windows; MediaEval 2018 1Hz VA | 9,800 clips / 30 continuous movies / MediaEval subsets | Likely large; multi-hour video collection | Strong if official package acquired; not immediate | Yes | Yes | Aggregate affect and temporal affect after access |
| CASE | Source videos not directly included; metadata gives URLs/time windows | Yes in archive; per-participant continuous valence/arousal and physiology | Published DLR zip URL currently returns 404; videos require external URLs/editing | Valence/arousal continuous joystick traces; intended video category labels | 20 Hz annotations after interpolation | 8 emotional stimuli plus start/end/blue screen | Archive unknown; not reachable now | Weak until source videos are manually acquired/edited | Yes after video reconstruction | Yes | Temporal affect and physiology validation, but not immediate |
| VEATIC | Yes via project Google Drive dataset | Yes; averaged valence/arousal CSVs and individual psychophysics ratings for subset | Research-use download link; Google Drive-hosted; copyright remains with original video owners | Continuous valence/arousal per frame for target characters | Per-frame ratings | 124 video clips | Unknown from headers; likely moderate, not huge relative to LIRIS | Strong; official folder contract aligns media and labels by video ID/frame | Yes | Yes | Best immediate temporal affect benchmark |
| EEV subset | Not bundled; references public video platform IDs | Yes; CSVs with 15 evoked-expression labels at 6 Hz | GitHub CSVs accessible; videos can disappear from source platform | 15 continuous evoked expression labels, not VA | 6 Hz | 5,153 videos / 370 hours | Labels alone: train ~812 MB, val ~211 MB; media huge and not bundled | Weak; media availability unstable | Yes if individual videos are reacquired | Yes | Behaviour/reaction-expression prediction, not clean immediate TRIBE benchmark |
| FindingEmo | Images not bundled; URL list only | Yes; valence, arousal, dominance, emotion labels | Dataset/source code available; package available; images not redistributed | Image-level V/A/D and discrete emotion labels | Static image-level | 25,000 images | Labels small; image download depends URLs | Weak for video; static only unless converted to pseudo-video | Image/text only; no real video/audio | No real temporal benchmark | Fallback aggregate image affect only |

## Dataset Notes

### LIRIS-ACCEDE

Status: strong but gated.

The project states that access requires printing/signing/scanning an EULA and emailing it to the dataset maintainers. It also states requests from free email addresses are refused and download links can take up to a week.

Relevant source facts:

- LIRIS-ACCEDE contains Creative Commons video clips and annotations.
- It includes Discrete LIRIS-ACCEDE: 9,800 short excerpts from 160 movies.
- It includes Continuous LIRIS-ACCEDE: 30 whole movies with continuous valence/arousal self-assessments.
- MediaEval 2017/2018 subsets provide valence/arousal/fear tasks with temporal structure.

Recommendation:

Use LIRIS after access is granted. Do not block the current benchmark path on it.

### CASE

Status: scientifically attractive but not immediately usable as a media benchmark.

CASE contains 30 participants watching 8 affective video stimuli. It has simultaneous continuous valence/arousal annotations and physiology. The paper reports 20 Hz annotation data and exact video-duration metadata.

Blocking issue:

- The cited archive URL `https://rmc.dlr.de/download/CASE_dataset/CASE_dataset.zip` currently returns HTTP 404.
- The paper says the videos are not directly shared in the dataset because of copyright; metadata contains URLs/time windows for reconstructing them.

Recommendation:

If we can obtain the CASE archive or a working mirror, CASE is a good secondary temporal benchmark. It is not the next immediate adapter unless archive access is restored.

### VEATIC

Status: acquired, validated, and ready for gated TRIBE extraction.

VEATIC has 124 annotated videos, averaged valence/arousal ratings, and direct video/label file naming. The project describes this local structure:

```text
dataset/
  video/${video_id}.mp4
  rating_averaged/${video_id}_valence.csv
  rating_averaged/${video_id}_arousal.csv
```

The official split is frame-temporal:

- first 70% of frames: train
- last 30% of frames: test

The local manifest also carries:

- Mode A: official VEATIC first-70% / last-30% frame split.
- Mode B: blocked temporal split with a 10% middle gap.
- Mode C: leave-video-out grouped split.

Implemented local adapter:

[build_veatic_manifest.py](/Users/maxsartini/Neural Bridge/backend/scripts/build_veatic_manifest.py)

Implemented annotation-only split baseline runner:

[run_veatic_annotation_baseline.py](/Users/maxsartini/Neural Bridge/backend/scripts/run_veatic_annotation_baseline.py)

Example command after dataset download:

```bash
backend/.venv/bin/python backend/scripts/build_veatic_manifest.py \
  --root "/Volumes/onn. Drive/Neural Bridge/datasets/veatic" \
  --output benchmarks/veatic/veatic_manifest_1hz.jsonl \
  --report benchmarks/veatic/veatic_manifest_1hz.report.json \
  --sample-hz 1 \
  --strict
```

The adapter validates:

- video exists,
- valence/arousal files exist,
- valence/arousal lengths match,
- `ffprobe` can read duration and frame rate,
- label-derived duration aligns to media duration,
- rows carry train/test split without mixing OpenLAV or Emo-FilM contracts.

Current local validation:

- dataset root: `/Volumes/onn. Drive/Neural Bridge/datasets/veatic`
- valid videos: 124
- rejected videos: 0
- manifest rows at 1 Hz: 10,357
- strict duration/frame alignment: passed
- one-video cortical+subcortical TRIBE smoke: passed on video `52`
- readiness report: [veatic_readiness_report.md](/Users/maxsartini/Neural Bridge/benchmarks/veatic/veatic_readiness_report.md)

### EEV

Status: useful but not a clean immediate media benchmark.

EEV provides train/val/test CSVs on GitHub. It does not bundle videos; it references public video-platform IDs. The train CSV alone is ~812 MB and validation is ~211 MB. Labels are 15 evoked-expression dimensions sampled at 6 Hz.

This is valuable for behaviour/reaction-response prediction, but media availability can decay over time and downloading source videos would be a separate unstable acquisition job.

Recommendation:

Use later as a reaction-expression benchmark, not as the next TRIBE media benchmark.

### FindingEmo

Status: fallback only.

FindingEmo is image-based. It provides 25,000 image URLs and valence/arousal/emotion labels, but it does not distribute images directly and does not provide real temporal video.

Recommendation:

Use only if we need a static-image aggregate affect sanity check. It is not a proper replacement for Emo-FilM/VEATIC temporal benchmarking.

## Next Step

Acquire VEATIC from the official Google Drive link into:

`/Volumes/onn. Drive/Neural Bridge/datasets/veatic`

Expected post-extract structure:

```text
/Volumes/onn. Drive/Neural Bridge/datasets/veatic/
  video/
    0.mp4
    1.mp4
    ...
  rating_averaged/
    0_valence.csv
    0_arousal.csv
    ...
```

Then run the VEATIC manifest builder in strict mode. Only after the manifest passes should TRIBE extraction begin.

## Sources

- LIRIS-ACCEDE: https://liris-accede.ec-lyon.fr/
- CASE paper: https://arxiv.org/abs/1812.02782
- VEATIC project: https://veatic.github.io/
- VEATIC code: https://github.com/AlbusPeter/VEATIC
- EEV code/data: https://github.com/google-research-datasets/eev
- FindingEmo: https://iiw.kuleuven.be/onderzoek/eavise/findingemo/home
