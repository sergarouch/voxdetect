# Voxdetect

Voxdetect trains a small neural net to tell **human voice** from **everything else**, and it can collect anonymous clips from your community with silence, noise, English, and other languages kept as separate labels.

The model is a convolutional-recurrent network on 1-second log-mel spectrograms (16 kHz mono, 64 mel bands). A clip longer than a second is scored in overlapping windows and averaged.

## Requirements

- Python 3.11 or newer
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (`sudo apt install ffmpeg` on Debian/Ubuntu). wav, flac, and mp3 are all decoded with ffmpeg.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Use the CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the CPU index if you have an NVIDIA GPU. `pip install -e .` keeps that build as long as it already satisfies the `torch` requirement.

## Collect clips from your community

```bash
voxdetect collect --host 0.0.0.0 --port 8741
```

Open `http://127.0.0.1:8741` and share that address with contributors. On another computer, use this machine’s address and the same port, for example `http://192.168.1.20:8741`. The page is public to anyone with the link: no account and no name field. It discards the original filename, does not play clips back to other visitors, and does not write addresses into the catalog.

The page shows a short record step for each category: set the device down and do not speak; point at noise and discard it if you hear words; read the English line with no name; or pick another language (Not sure if you do not know the code).

Each clip is one of four labels:

| Label | Stored when | Extra field |
| --- | --- | --- |
| `silence` | No distinct sound | none |
| `noise` | Sound, but nobody is speaking | `noise_kind`: room, traffic, music, typing, crowd, appliance, outdoors, other |
| `english` | English speech | `language` is always `en` |
| `non_english` | Any other spoken language | `language` is an ISO 639-1 code such as `es` or `zh`, or `und` if the code is unknown |

Clips land in `data/collected/`:

```text
data/collected/schema.json
data/collected/submissions.jsonl
data/collected/labels.csv
data/collected/audio/silence/<id>.wav
data/collected/audio/noise/<noise_kind>/<id>.wav
data/collected/audio/english/<id>.wav
data/collected/audio/non_english/<language>/<id>.wav
```

`submissions.jsonl` is the catalog. One JSON object per line, with `label`, `language`, `noise_kind`, `prompt_id`, duration, and a relative path. There is no name column.

When you have at least two of the four labels, import them without collapsing the classes:

```bash
voxdetect import --collected data/collected --out data/manifests
voxdetect train --manifests data/manifests --epochs 20 --out runs/best.pt
```

That trains a four-way model. `voxdetect infer` then prints `silence`, `noise`, `english`, or `non_english`.

## Import training data

You need **both** classes. A few hundred clips of each is a useful start. Variety of noise matters as much as variety of voices.

| | |
| --- | --- |
| Sample rate | Any. Files are resampled to 16 kHz mono. |
| Length | At least 0.3 seconds. Training uses a 1-second crop. |
| Speech | Meetings, calls, read speech, different people |
| Non-speech | Silence, fans, traffic, music, typing, crowds without a clear voice |

Two layouts work. Re-running import rewrites the manifests and does not modify your audio.

### Folder of recordings

The folder name is the label. `speech`, `voice`, and `positive` are voice. `non_speech`, `noise`, `silence`, `negative`, and `background` are not. Nested folders are fine. Other audio at the top level is skipped and listed in `rejected.csv`.

```text
my_recordings/
  speech/
    meeting_01.wav
    call_02.flac
  non_speech/
    office_hum.wav
    street.mp3
```

```bash
voxdetect import --input ./my_recordings --out data/manifests
```

### Label CSV

Use this when the files are already organized your own way. `path` may be absolute or relative to the CSV. `label` may be `speech` / `non_speech` or `1` / `0`.

```csv
path,label
clips/hello.wav,speech
clips/traffic.wav,non_speech
clips/kitchen.mp3,0
```

```bash
voxdetect import --csv ./labels.csv --out data/manifests
```

Import decodes every file, drops anything missing, unreadable, or shorter than 0.3 seconds, and writes:

- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test.csv`
- `data/manifests/rejected.csv`

The split is 80% train, 10% validation, 10% test, stratified by class, with seed 42. Paths in the manifests are absolute, so leave the audio where it was (or run import again after you move it).

The summary printed at the end is the class balance. If one class is 90% or more of the clips, accuracy can look high while the model never learns the other class.

Public sets that match these labels, if you do not have recordings yet: [LibriSpeech](https://www.openslr.org/12) or [Mozilla Common Voice](https://commonvoice.mozilla.org/) for speech, and [MUSAN](https://www.openslr.org/17/) (noise and music) for non-speech. Unpack them into the two folders above and run the same command.

## Train, evaluate, score

```bash
voxdetect train --manifests data/manifests --epochs 20 --out runs/best.pt
voxdetect evaluate --checkpoint runs/best.pt --manifest data/manifests/test.csv
voxdetect infer --checkpoint runs/best.pt --audio path/to/clip.wav
```

`infer` also accepts a folder. Each line is a probability and a label (`speech` when the probability is at least 0.5).

```text
0.910  speech      path/to/clip.wav
```

The checkpoint stores the network weights and the feature settings used to train them, so scoring stays aligned with training.

## Check the pipeline before you have recordings

`synthetic` writes a handful of tones and noise clips. They only prove import and training run. They will not teach the model what a real voice sounds like.

```bash
voxdetect synthetic --out data/raw/synthetic
voxdetect import --input data/raw/synthetic --out data/manifests
voxdetect train --manifests data/manifests --epochs 5 --out runs/best.pt
voxdetect infer --checkpoint runs/best.pt --audio data/raw/synthetic/speech
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```
