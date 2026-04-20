"""
config.py — Configuration for CWformer Morse code decoder.

Dataclasses covering audio generation and training:
  MorseConfig    — synthetic audio generation parameters
  TrainingConfig — training hyperparameters and curriculum settings

Use create_default_config(scenario) to get pre-built configs for:
  "test"     — tiny run (~5 epochs) to verify the pipeline end-to-end
  "clean"    — curriculum stage 1: high SNR, standard timing (200 epochs)
  "moderate" — curriculum stage 2: mid SNR, moderate bad-fist (300 epochs)
  "full"     — curriculum stage 3: low SNR, extreme bad-fist (500 epochs)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# MorseConfig — audio generation
# ---------------------------------------------------------------------------

@dataclass
class MorseConfig:
    """Synthetic Morse audio generation parameters.

    Timing ratios (dah_dit, ics, iws) are sampled independently per sample
    to cover the full range of real-world operator styles.
    """

    # Internal sample rate; all audio inputs are resampled to this at inference
    sample_rate: int = 16000

    # WPM range
    min_wpm: float = 10.0
    max_wpm: float = 40.0

    # Tone carrier frequency range (Hz)
    tone_freq_min: float = 500.0
    tone_freq_max: float = 900.0

    # Slow sinusoidal frequency drift (Hz peak deviation) — simulates VFO drift
    tone_drift: float = 3.0

    # SNR (dB) — measured against full-band white AWGN
    min_snr_db: float = 15.0
    max_snr_db: float = 40.0

    # Timing jitter: fraction of unit duration (std dev of Gaussian perturbation)
    # Actual per-sample jitter is drawn uniformly in [timing_jitter, timing_jitter_max]
    timing_jitter: float = 0.0
    timing_jitter_max: float = 0.05

    # Dah/dit ratio (ITU standard = 3.0; bad-fist operators can go down to 1.5)
    dah_dit_ratio_min: float = 2.5
    dah_dit_ratio_max: float = 3.5

    # Inter-character space factor (× standard 3-dit gap)
    # 1.0 = standard; <1.0 = compressed; >1.0 = expanded
    ics_factor_min: float = 0.8
    ics_factor_max: float = 1.2

    # Inter-word space factor (× standard 7-dit gap)
    iws_factor_min: float = 0.8
    iws_factor_max: float = 1.5

    # Text length range (characters, including spaces)
    min_chars: int = 20
    max_chars: int = 120

    # Signal amplitude variation across samples
    signal_amplitude_min: float = 0.5
    signal_amplitude_max: float = 0.9

    # AGC simulation — noise-floor modulation matching real HF radio AGC.
    # During marks the AGC reduces gain → background noise is suppressed.
    # During spaces the AGC releases → noise rises to full level over release_ms.
    # This creates the characteristic noise-floor drift seen between elements in
    # real recordings.  Noise is modulated *before* the IF filter so the effect
    # appears in the feature extractor's noise estimate.
    agc_probability: float = 0.0        # fraction of samples with AGC enabled
    agc_attack_ms: float = 50.0         # gain reduction time constant (ms)
    agc_release_ms: float = 400.0       # gain recovery time constant (ms)
    agc_depth_db_min: float = 6.0       # noise suppression at peak mark (dB, min)
    agc_depth_db_max: float = 15.0      # noise suppression at peak mark (dB, max)

    # QSB — slow sinusoidal signal fading within a sample (0.05–0.3 Hz).
    # Captures mark-to-mark amplitude variation from propagation.
    qsb_probability: float = 0.0
    qsb_depth_db_min: float = 3.0      # peak-to-peak fading range (dB, min)
    qsb_depth_db_max: float = 10.0     # peak-to-peak fading range (dB, max)

    # Key type weights: (straight_key, bug, paddle, cootie) probabilities.
    # Straight key: per-character speed variation, high jitter on all elements.
    # Bug (semi-automatic): consistent dits, variable dahs + spacing.
    # Paddle (electronic keyer): consistent elements, variable spacing only.
    # Cootie (sideswiper): alternating contacts, symmetric but high jitter,
    #   no inherent dit/dah length distinction — operator must time everything.
    key_type_weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    # Speed drift: slow WPM variation within a single transmission.
    # Fraction of base unit_dur (e.g. 0.15 = ±15%).  0.0 = constant speed.
    speed_drift_max: float = 0.0

    # Farnsworth timing: characters sent at a faster speed (char_wpm) but
    # inter-character and inter-word spaces stretched to achieve a slower
    # effective speed (the configured WPM).  Common in CW training and some
    # operators' natural style.  0.0 = disabled, otherwise probability of
    # applying Farnsworth timing to a given sample.
    farnsworth_probability: float = 0.0
    # The character speed multiplier: char_wpm = wpm * farnsworth_char_speed_mult.
    # E.g. 1.5 means characters sent 50% faster than overall effective speed.
    farnsworth_char_speed_min: float = 1.3
    farnsworth_char_speed_max: float = 2.0

    # Keying waveform shaping: rise/fall time range for mark envelopes.
    # Real transmitters have 2-8 ms rise/fall; 0 ms = hard keying (unrealistic).
    # The default 5 ms is a good middle ground.
    rise_time_ms_min: float = 3.0
    rise_time_ms_max: float = 8.0

    # QRM — interfering CW signals at nearby frequencies.
    # Simulates other operators transmitting on adjacent frequencies.
    qrm_probability: float = 0.0         # fraction of samples with QRM
    qrm_count_min: int = 1               # min number of interferers
    qrm_count_max: int = 3               # max number of interferers
    qrm_freq_offset_min: float = 100.0   # min frequency offset from target (Hz)
    qrm_freq_offset_max: float = 500.0   # max frequency offset from target (Hz)
    qrm_amplitude_min: float = 0.1       # min amplitude relative to main signal
    qrm_amplitude_max: float = 0.8       # max amplitude relative to main signal

    # QRN — impulsive atmospheric noise (static crashes from lightning).
    # Poisson-distributed impulses with random duration and amplitude.
    qrn_probability: float = 0.0         # fraction of samples with QRN
    qrn_rate_min: float = 0.5            # min impulse rate (per second)
    qrn_rate_max: float = 5.0            # max impulse rate (per second)
    qrn_duration_ms_min: float = 1.0     # min impulse duration (ms)
    qrn_duration_ms_max: float = 50.0    # max impulse duration (ms)
    qrn_amplitude_min: float = 0.3       # min impulse amplitude (relative to signal)
    qrn_amplitude_max: float = 2.0       # max impulse amplitude (relative to signal)

    # Receiver bandpass filter — simulates a real CW filter (200-500 Hz BW).
    # Applied after all signal mixing, before normalisation.
    # Real radios always have a filter; probability should be high (0.5-1.0).
    bandpass_probability: float = 0.0     # fraction of samples with bandpass
    bandpass_bw_min: float = 200.0        # min filter bandwidth (Hz)
    bandpass_bw_max: float = 500.0        # max filter bandwidth (Hz)
    bandpass_order_min: int = 4           # min Butterworth filter order
    bandpass_order_max: int = 4           # max Butterworth filter order (sampled per sample)

    # Real HF noise — mix recorded HF band noise instead of/with AWGN.
    # Bridges the synthetic-to-real gap by using actual band characteristics.
    hf_noise_probability: float = 0.0     # fraction of samples using real HF noise
    hf_noise_dir: str = "recordings"      # directory containing noise WAV files
    hf_noise_mix_ratio: float = 0.7       # fraction of noise that is real HF (rest is AWGN)

    # Multi-operator speed change — abrupt WPM changes between words.
    # Simulates operator changes on multi-op stations or natural speed variation.
    multi_op_probability: float = 0.0     # fraction of samples with speed changes
    multi_op_speed_change_min: float = 0.7   # min speed multiplier at change point
    multi_op_speed_change_max: float = 1.4   # max speed multiplier at change point

    # Random input gain (dB), applied AFTER peak-normalisation in
    # generate_sample(). Drawn log-uniformly in [lo, hi] dB per sample and
    # multiplied onto the waveform (then clipped to [-1, 1]). Teaches the
    # model to handle inputs that aren't peak-normalised, which matches
    # the real streaming inference regime where per-chunk peak varies.
    # (0.0, 0.0) disables the augmentation (preserves legacy behaviour).
    input_gain_db_range: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MorseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# FeatureConfig — STFT → SNR ratio feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """STFT-based adaptive threshold feature extraction parameters.

    MorseEventExtractor uses these parameters to compute per-frame E values
    and emit MorseEvent objects (mark/space intervals with duration +
    confidence).  The asymmetric EMA threshold is AGC-immune and requires
    no explicit noise floor estimation.
    """

    # Must match MorseConfig.sample_rate; audio resampled to this at inference
    sample_rate: int = 16000

    # STFT window and hop size (determines freq resolution and frame rate)
    # window=20ms → 50 Hz/bin at 16 kHz (320 samples); hop=5ms → 200 fps (80 samples)
    # 20ms window chosen for time resolution: at 40 WPM a dit/space is ~30ms,
    # so the window clears within 4 hops (20ms) of a transition — enabling
    # clean inter-element space detection. Cost: -4 dB peak-bin SNR vs 50ms.
    # 18 bins cover 300-1200 Hz at 50 Hz/bin — adequate for tone detection.
    # 16 kHz gives 8 kHz Nyquist; audio that arrives at 8 kHz is upsampled
    # (no new information above 4 kHz, but monitoring range is 300-1200 Hz
    # so this is irrelevant for signal detection).
    window_ms: float = 20.0
    hop_ms: float = 5.0

    # Frequency range to monitor (Hz)
    # Should cover the expected signal frequency plus margin for noise bins
    freq_min: int = 300
    freq_max: int = 1200

    # Blip filter: state changes that last this many frames or fewer are
    # absorbed back into the current interval and not emitted as events.
    # A transition is only confirmed after blip_threshold_frames + 1
    # consecutive frames in the new state.
    #
    # Default 2: rejects transitions ≤ 10 ms (at 5 ms/frame), which is
    # shorter than a dit at 90 WPM (≈ 13.3 ms).  Any feature of the audio
    # that cannot last at least 3 frames (15 ms) is treated as noise.
    #
    # Set to 1 to restore the original single-frame blip filter.
    # Set to 0 to disable blip filtering (every frame can cause a transition).
    blip_threshold_frames: int = 2

    # --- Adaptive FAST_DB ---
    # When enabled, EMA tracking speed adapts based on the current
    # mark-space spread (proxy for SNR).  At low spread (poor SNR) the
    # FAST_DB is reduced (more aggressive tracking) so faint marks are
    # followed faster; at high spread it stays conservative.
    adaptive_fast_db: bool = True
    fast_db_min: float = 4.0    # aggressive — used when spread ≈ MIN_SPREAD
    fast_db_max: float = 6.0    # conservative — used when spread is large

    # --- Threshold center weighting ---
    # Fraction of the adaptive threshold center attributed to mark_ema.
    # Higher = more conservative (fewer false marks, but absorbs weak marks).
    # 0.667 = original 2:1 toward mark; 0.5 = midpoint (equal weight).
    # 0.55 tested as optimal: +0.10 confidence with no false positive increase.
    center_mark_weight: float = 0.55

    # --- Adaptive blip filter ---
    # When enabled, the blip confirmation threshold varies with the
    # current mark-space spread: tighter confirmation at low spread
    # (low SNR), faster confirmation at high spread (high SNR).
    # When disabled, the fixed blip_threshold_frames is used everywhere.
    adaptive_blip: bool = True
    blip_threshold_low_snr: int = 3   # frames required at low spread
    blip_threshold_high_snr: int = 1  # frames required at high spread

    @property
    def fps(self) -> float:
        """Output frame rate in frames per second."""
        return 1000.0 / self.hop_ms

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model architecture parameters (retained for config serialization)."""

    in_features: int = 5
    hidden_size: int = 128
    n_rnn_layers: int = 3
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# TrainingConfig — training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 300

    # Synthetic samples generated per epoch (train / validation)
    samples_per_epoch: int = 5000
    val_samples: int = 500

    # DataLoader worker processes (0 = main process only)
    num_workers: int = 4

    # Checkpoint and logging
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50           # batches between mid-epoch loss prints

    # Streaming-mode validation: run val audio through CWFormerStreamingDecoder
    # every N epochs and log val_cer_stream alongside val_cer_full. Drift of
    # > ~1% absolute CER on converged in-distribution audio is a regression
    # signal (cuDNN kernel heuristics + chunked state plumbing). 0 disables
    # the feature; validation stays full-forward only.
    stream_val_every_n_epochs: int = 0
    stream_val_chunk_ms: int = 500
    stream_val_max_cache_sec: float = 30.0
    # Cap on how many val samples go through the streaming path per eval.
    # Per-sample streaming is serial (each chunk waits on the prior state),
    # so doing all 5000 val samples turns a 30 s full-forward eval into a
    # multi-hour wait. 50 samples × ~50 chars each ≈ 2500-char CER base —
    # enough resolution to spot >~1% drift between paths.
    stream_val_samples: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Full pipeline configuration (generation + features + model + training)."""

    morse: MorseConfig = field(default_factory=MorseConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return {
            "morse":    self.morse.to_dict(),
            "feature":  self.feature.to_dict(),
            "model":    self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(
            morse=MorseConfig.from_dict(d.get("morse", {})),
            feature=FeatureConfig.from_dict(d.get("feature", {})),
            model=ModelConfig.from_dict(d.get("model", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
        )


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

def create_default_config(scenario: str = "clean") -> Config:
    """Return a pre-configured Config for the given training scenario.

    Scenarios
    ---------
    test     — 5 epochs, tiny epoch size; verifies the full pipeline
    clean    — 200 epochs; high SNR, near-standard timing (curriculum stage 1)
    moderate — 300 epochs; mid SNR, moderate bad-fist (curriculum stage 2)
    full     — 500 epochs; low SNR, extreme bad-fist (curriculum stage 3)
    """
    cfg = Config()

    if scenario == "test":
        cfg.morse.min_snr_db = 20.0
        cfg.morse.max_snr_db = 40.0
        cfg.morse.min_wpm = 15.0
        cfg.morse.max_wpm = 25.0
        cfg.morse.min_chars = 15
        cfg.morse.max_chars = 40
        cfg.morse.dah_dit_ratio_min = 2.8
        cfg.morse.dah_dit_ratio_max = 3.2
        cfg.morse.ics_factor_min = 0.9
        cfg.morse.ics_factor_max = 1.1
        cfg.morse.iws_factor_min = 0.9
        cfg.morse.iws_factor_max = 1.1
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.02
        cfg.morse.tone_drift = 1.0
        cfg.training.num_epochs = 5
        cfg.training.samples_per_epoch = 500
        cfg.training.val_samples = 100
        cfg.training.num_workers = 0
        cfg.training.batch_size = 16
        cfg.training.learning_rate = 5e-4

    elif scenario == "clean":
        cfg.morse.min_snr_db = 15.0
        cfg.morse.max_snr_db = 40.0
        cfg.morse.min_wpm = 5.0
        cfg.morse.max_wpm = 50.0
        cfg.morse.min_chars = 30
        cfg.morse.max_chars = 150
        cfg.morse.dah_dit_ratio_min = 2.5
        cfg.morse.dah_dit_ratio_max = 3.5
        cfg.morse.ics_factor_min = 0.8
        cfg.morse.ics_factor_max = 1.2
        cfg.morse.iws_factor_min = 0.8
        cfg.morse.iws_factor_max = 1.5
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.05
        cfg.morse.tone_drift = 3.0
        cfg.training.batch_size = 512
        cfg.training.learning_rate = 1e-3
        cfg.training.num_epochs = 200
        cfg.training.samples_per_epoch = 100000
        cfg.training.val_samples = 5000
        cfg.training.num_workers = 4
        # Real-world augmentations (mild -- model learns basic task first)
        cfg.morse.agc_probability = 0.2
        # qsb_probability stays off for the clean stage
        cfg.morse.qsb_probability = 0.0
        # Key type: mostly paddles (easiest) for clean stage; no cootie yet
        cfg.morse.key_type_weights = (0.20, 0.20, 0.60, 0.0)
        # Farnsworth: mild introduction (10% of samples, mild stretch)
        cfg.morse.farnsworth_probability = 0.10
        cfg.morse.farnsworth_char_speed_min = 1.2
        cfg.morse.farnsworth_char_speed_max = 1.5
        # Bandpass filter: half of samples, wide filter, gentle slopes
        cfg.morse.bandpass_probability = 0.50
        cfg.morse.bandpass_bw_min = 400.0
        cfg.morse.bandpass_bw_max = 500.0
        cfg.morse.bandpass_order_min = 4
        cfg.morse.bandpass_order_max = 6
        # Real HF noise: mild introduction (15% of samples)
        cfg.morse.hf_noise_probability = 0.15
        cfg.morse.hf_noise_mix_ratio = 0.5
        # Input gain: disabled for the clean stage (waveform stays peak-normalised)
        cfg.morse.input_gain_db_range = (0.0, 0.0)

    elif scenario == "moderate":
        cfg.morse.min_snr_db = 5.0
        cfg.morse.max_snr_db = 35.0
        cfg.morse.min_wpm = 5.0
        cfg.morse.max_wpm = 50.0
        cfg.morse.min_chars = 25
        cfg.morse.max_chars = 175
        cfg.morse.dah_dit_ratio_min = 1.8
        cfg.morse.dah_dit_ratio_max = 3.8
        cfg.morse.ics_factor_min = 0.6
        cfg.morse.ics_factor_max = 1.6
        cfg.morse.iws_factor_min = 0.6
        cfg.morse.iws_factor_max = 2.0
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.15
        cfg.morse.tone_drift = 4.0
        cfg.training.batch_size = 512
        cfg.training.learning_rate = 1e-3
        cfg.training.num_epochs = 300
        cfg.training.samples_per_epoch = 75000
        cfg.training.val_samples = 5000
        cfg.training.num_workers = 4
        # Real-world augmentations (moderate strength)
        cfg.morse.agc_probability = 0.4
        cfg.morse.agc_depth_db_max = 18.0
        cfg.morse.qsb_probability = 0.25
        cfg.morse.qsb_depth_db_max = 12.0
        # Key type: balanced mix, introduce cootie
        cfg.morse.key_type_weights = (0.25, 0.25, 0.35, 0.15)
        # Speed drift: mild ±8% WPM variation
        cfg.morse.speed_drift_max = 0.08
        # Farnsworth: moderate (20% of samples)
        cfg.morse.farnsworth_probability = 0.20
        cfg.morse.farnsworth_char_speed_min = 1.3
        cfg.morse.farnsworth_char_speed_max = 1.8
        # QRM: light introduction (15% of samples, 1-2 interferers)
        cfg.morse.qrm_probability = 0.15
        cfg.morse.qrm_count_max = 2
        cfg.morse.qrm_amplitude_max = 0.5
        # QRN: light introduction (15% of samples)
        cfg.morse.qrn_probability = 0.15
        cfg.morse.qrn_rate_max = 3.0
        cfg.morse.qrn_amplitude_max = 1.0
        # Bandpass filter: narrower filters, sharper slopes
        cfg.morse.bandpass_probability = 0.60
        cfg.morse.bandpass_bw_min = 250.0
        cfg.morse.bandpass_bw_max = 500.0
        cfg.morse.bandpass_order_min = 4
        cfg.morse.bandpass_order_max = 8
        # Real HF noise: moderate (30% of samples)
        cfg.morse.hf_noise_probability = 0.30
        cfg.morse.hf_noise_mix_ratio = 0.6
        # Multi-operator: disabled at moderate stage
        cfg.morse.multi_op_probability = 0.0
        cfg.morse.multi_op_speed_change_min = 0.8
        cfg.morse.multi_op_speed_change_max = 1.3
        # Input gain: ±6 dB log-uniform per sample
        cfg.morse.input_gain_db_range = (-6.0, 6.0)

    elif scenario == "full":
        cfg.morse.min_snr_db = -5.0
        cfg.morse.max_snr_db = 30.0
        cfg.morse.min_wpm = 5.0
        cfg.morse.max_wpm = 50.0
        cfg.morse.min_chars = 20
        cfg.morse.max_chars = 200
        cfg.morse.dah_dit_ratio_min = 1.3
        cfg.morse.dah_dit_ratio_max = 4.0
        cfg.morse.ics_factor_min = 0.5
        cfg.morse.ics_factor_max = 2.0
        cfg.morse.iws_factor_min = 0.5
        cfg.morse.iws_factor_max = 2.5
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.25
        cfg.morse.tone_drift = 5.0
        # sqrt(512/128) * 6e-4 ~ 1.2e-3; rounded to 1e-3.
        cfg.training.batch_size = 512
        cfg.training.learning_rate = 1e-3
        cfg.training.num_epochs = 500
        cfg.training.samples_per_epoch = 50000
        cfg.training.val_samples = 5000
        cfg.training.num_workers = 4
        # Real-world augmentations (full strength for curriculum stage 3)
        cfg.morse.agc_probability = 0.5
        cfg.morse.agc_depth_db_max = 22.0
        cfg.morse.qsb_probability = 0.50
        cfg.morse.qsb_depth_db_max = 18.0
        # Key type: weighted toward harder key types (straight key, bug, cootie)
        cfg.morse.key_type_weights = (0.30, 0.30, 0.20, 0.20)
        # Speed drift: ±15% WPM variation within a transmission
        cfg.morse.speed_drift_max = 0.15
        # Farnsworth: full range (25% of samples)
        cfg.morse.farnsworth_probability = 0.25
        cfg.morse.farnsworth_char_speed_min = 1.3
        cfg.morse.farnsworth_char_speed_max = 2.0
        # QRM: full strength (30% of samples, 1-3 interferers)
        cfg.morse.qrm_probability = 0.30
        cfg.morse.qrm_count_max = 3
        cfg.morse.qrm_amplitude_max = 0.8
        # QRN: full strength (25% of samples)
        cfg.morse.qrn_probability = 0.25
        cfg.morse.qrn_rate_max = 5.0
        cfg.morse.qrn_amplitude_max = 2.0
        # Bandpass filter: wider filters, slightly reduced probability
        cfg.morse.bandpass_probability = 0.60
        cfg.morse.bandpass_bw_min = 200.0
        cfg.morse.bandpass_bw_max = 500.0
        cfg.morse.bandpass_order_min = 4
        cfg.morse.bandpass_order_max = 8
        # Real HF noise: full (50% of samples)
        cfg.morse.hf_noise_probability = 0.50
        cfg.morse.hf_noise_mix_ratio = 0.7
        # Multi-operator: full (15% of samples, wider speed range)
        cfg.morse.multi_op_probability = 0.15
        cfg.morse.multi_op_speed_change_min = 0.7
        cfg.morse.multi_op_speed_change_max = 1.4
        # Input gain: ±12 dB log-uniform per sample
        cfg.morse.input_gain_db_range = (-12.0, 12.0)

    else:
        raise ValueError(
            f"Unknown scenario: {scenario!r}.  Choose from: test, clean, moderate, full."
        )

    return cfg


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for s in ("test", "clean", "moderate", "full"):
        cfg = create_default_config(s)
        print(
            f"[{s:8s}]  SNR={cfg.morse.min_snr_db:.0f}–{cfg.morse.max_snr_db:.0f} dB  "
            f"WPM={cfg.morse.min_wpm:.0f}–{cfg.morse.max_wpm:.0f}  "
            f"dah/dit={cfg.morse.dah_dit_ratio_min:.1f}–{cfg.morse.dah_dit_ratio_max:.1f}  "
            f"epochs={cfg.training.num_epochs}"
        )
