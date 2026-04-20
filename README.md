# CWformer

A causal streaming neural Morse code (CW) decoder. It uses a fully causal Conformer architecture (~19.5M parameters) with CTC loss that processes audio left-to-right with no bidirectional attention, eliminating the window-stitching artifacts of the original [CWNet](https://github.com/parsimo2010/CWNet) bidirectional model.

CWformer decodes CW from audio in real time — feed it audio from a USB sound card, a file, or stdin, and it emits decoded text as characters are confirmed. It targets 15–40 WPM across all common key types (straight key, bug, paddle, cootie) at SNR > 5–8 dB, with under 2.5 seconds of latency from audio to character emission. Current results show it performs decently down to 10 WPM and it has been tested up to 35 WPM with good accuracy, suggesting it can meet the 40 WPM goal. Accuracy degrades progressively worse as speed drops below 10 WPM, and at some high speeed the accuracy will degrade due to the 20ms time resolution of the inputs.

## How It Works

Audio is processed causally: each frame only sees past context, never the future. During inference, model state (KV caches and convolution buffers) carries forward between processing chunks, so there are no windows to stitch together. The model is trained so that characters sharing element prefixes (E=`.`, I=`..`, S=`...`, H=`....`, 5=`.....`) are held until the inter-character space is confirmed, ensuring that only correct decodes are emitted.

```
Audio (16 kHz mono)
  → Incremental log-mel spectrogram (40 bins, 25ms/10ms)
  → Causal ConvSubsampling (2× time reduction → 50 fps)
  → 12× Causal Conformer blocks (d=256, 4 heads, conv kernel=31)
  → CTC head → greedy decode
  → Text
```

## Project Structure

```
CWformer/
├── config.py                    # MorseConfig, TrainingConfig
├── vocab.py                     # CTC vocabulary (52 classes)
├── morse_table.py               # ITU Morse code table + binary trie
├── morse_generator.py           # Synthetic training data generation
├── qso_corpus.py                # Realistic ham radio QSO text corpus
├── quantize_cwformer.py         # ONNX export + INT8 quantization
├── benchmark_cwformer.py        # Structured SNR×WPM×key benchmark
├── benchmark_random_sweep.py    # Random parameter sweep benchmark
│
├── neural_decoder/
│   ├── cwformer.py              # CW-Former model (forward + forward_streaming)
│   ├── conformer.py             # Causal Conformer blocks (attention, conv, state)
│   ├── mel_frontend.py          # Mel spectrogram (batch + streaming)
│   ├── rope.py                  # Rotary Position Embeddings
│   ├── dataset_audio.py         # Streaming IterableDataset
│   ├── train_cwformer.py        # Training loop (curriculum learning)
│   └── inference_cwformer.py    # CWFormerStreamingDecoder
│
├── deploy/
│   ├── inference_onnx.py        # ONNX Runtime streaming inference
│   └── ctc_decode.py            # Pure-numpy CTC beam search + LM
│
└── recordings/                  # Real HF band noise recordings for augmentation
```

## Running on a Raspberry Pi 5

This section walks through setting up CWformer on a Raspberry Pi 5 running Raspberry Pi OS (Bookworm, 64-bit) to decode CW from a USB sound card in real time using the quantized ONNX model.

### Prerequisites

Raspberry Pi OS Bookworm ships with Python 3.11. Install the required system packages and create a virtual environment:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libsndfile1 libportaudio2 git

python3 -m venv ~/cwformer-env
source ~/cwformer-env/bin/activate

pip install --upgrade pip
pip install numpy soundfile onnxruntime sounddevice
```

### Download the Model

Download the latest `cwformer-onnx-v0.1.0.zip` file from the [Releases](https://github.com/parsimo2010/CWformer/releases) page and place it in your home directory. Then run these commands:

```bash
unzip ~/cwformer-onnx-v0.1.0.zip -d ~/onnx-deployment
cd ~/onnx-deployment
```

### Streaming from a USB Sound Card

Plug in your USB sound card and find its device index:

```bash
source ~/cwformer-env/bin/activate
cd ~/onnx-deployment
python deploy/inference_onnx.py --model deploy/cwformer_streaming_int8.onnx --list-devices
```

Start decoding from the device (replace `2` with your device index):

```bash
python ~/onnx-deployment/inference_onnx.py --model ~/onnx-deployment/cwformer_streaming_int8.onnx --device 2
```

Omit the device number to use the system default input device:

```bash
python ~/onnx-deployment/inference_onnx.py --model ~/onnx-deployment/cwformer_streaming_int8.onnx --device
```

### Streaming from stdin

You can pipe raw 16-bit signed PCM audio (16 kHz, mono, little-endian) into the decoder via stdin. This is useful for chaining with `arecord`, `sox`, `rtl_sdr`, or any other tool that produces a PCM stream:

```bash
# Example: pipe from arecord (ALSA)
arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 -t raw | \
  python ~/onnx-deployment/inference_onnx.py --model ~/onnx-deployment/cwformer_streaming_int8.onnx --stdin
```

## Other Ways to Run Inference

The commands assume you have cloned the repository and have a CWformer directory with folders named `neural_decoder` and `deploy`, and have trained model file in the deploy folder. If you do not, update the checkpiont or model argumentst to point to the correct file location. You will need to download the model files from the releases page because we don't track the model files in GitHub (they are large files).

**Decode an audio file (ONNX):**
```bash
python deploy/inference_onnx.py --model deploy/cwformer_streaming_int8.onnx --input recording.wav
```

**Decode an audio file (PyTorch checkpoint):**
```bash
python -m neural_decoder.inference_cwformer --checkpoint deploy/best_model.pt --input recording.wav
```

**Use the fp32 ONNX model** (slightly more accurate, slower on CPU):
```bash
python deploy/inference_onnx.py --model deploy/cwformer_streaming_fp32.onnx --input recording.wav
```

**Run the structured benchmark suite:**
```bash
python benchmark_cwformer.py --checkpoint deploy/best_model.pt --csv results.csv
```

## Training Your Own Model

Training uses synthetic Morse audio generated on the fly — no dataset download is needed. You will need a machine with a CUDA-capable GPU and PyTorch installed. AMD GPUs work via ROCm but are not speed-optimized or fully tested.

Install the training dependencies:

```bash
pip install torch torchaudio numpy soundfile
```

Training follows a three-stage curriculum. Each stage adds progressively harder conditions (lower SNR, wider WPM range, more augmentations). You may want to stop early on the clean and moderate stages once validation loss plateaus — additional epochs past that point won't help.

**Stage 1 — Clean** (high SNR, moderate speeds):
```bash
python -m neural_decoder.train_cwformer --scenario clean --ckpt-dir checkpoints_clean
```

**Stage 2 — Moderate** (medium SNR, timing variance, QRM):
```bash
python -m neural_decoder.train_cwformer --scenario moderate --checkpoint checkpoints_clean/best_model.pt --ckpt-dir checkpoints_moderate
```

**Stage 3 — Full** (low SNR, all augmentations, all key types):
```bash
python -m neural_decoder.train_cwformer --scenario full --checkpoint checkpoints_moderate/best_model.pt --ckpt-dir checkpoints_full
```

After training, export to ONNX for deployment:

```bash
python quantize_cwformer.py --checkpoint checkpoints_full/best_model.pt --output-dir deploy/
```

This produces `cwformer_streaming_fp32.onnx`, `cwformer_streaming_int8.onnx`, and `mel_config.json` in the `deploy/` directory.

## Authorship

The code in this repository was primarily written by [Claude Code](https://claude.ai/claude-code) (Anthropic), with architecture design, domain expertise, and direction from [Harris Butler](https://github.com/parsimo2010) (parsimo2010).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
