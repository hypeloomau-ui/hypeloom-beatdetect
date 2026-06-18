"""Hypeloom beat-detection — Replicate model.

Contract:
  input : audio_url (mp3/wav)
  output: { bpm, beat_times[], downbeat_times[], time_signature, onset_envelope_peak_s, engine }

Engine chain (first that works wins):
  1) madmom  — RNNDownBeatProcessor + DBNDownBeatTracking (true trained downbeats).  PRIMARY.
  2) librosa — beat_track for tempo/beats + onset-energy phase heuristic for downbeats. FALLBACK.
  3) numpy   — detect_fallback.detect (zero-dependency). LAST RESORT.

n8n calls this with the same submit->poll pattern as fal (Replicate /predictions queue).
Run ONCE per track and cache the result on the track's Airtable record — never per render.
"""
import os, tempfile, urllib.request, traceback
import numpy as np
from cog import BasePredictor, Input

FPS = 100  # madmom activation frame rate


def _download(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[1] or ".mp3"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "hypeloom-beatdetect/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def _onset_peak_librosa(y, sr):
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr)
    # smooth over ~1s and take the global energy peak (the "drop")
    win = max(1, int(librosa.time_to_frames(1.0, sr=sr)))
    if win > 1:
        env = np.convolve(env, np.ones(win) / win, mode="same")
    peak_frame = int(np.argmax(env))
    return float(librosa.frames_to_time(peak_frame, sr=sr))


def _detect_madmom(path):
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    act = RNNDownBeatProcessor()(path)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=FPS)
    db = proc(act)                       # rows of [time_s, beat_position(1..N)]
    beat_times = db[:, 0]
    positions = db[:, 1].astype(int)
    downbeat_times = db[positions == 1, 0]
    time_sig = int(positions.max()) if len(positions) else 4
    if len(beat_times) > 3:
        bpm = float(60.0 / np.median(np.diff(beat_times)))
    else:
        bpm = 0.0
    return beat_times, downbeat_times, time_sig, bpm


def _detect_librosa(path):
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # downbeat phase: of the 4 candidate phases, pick the one whose beats carry the most onset energy
    ts = 4
    env = librosa.onset.onset_strength(y=y, sr=sr)
    ef = np.clip(librosa.time_to_frames(beat_times, sr=sr), 0, len(env) - 1)
    e = env[ef]
    best_p = max(range(ts), key=lambda p: e[p::ts].sum())
    downbeat_times = beat_times[best_p::ts]
    bpm = float(np.atleast_1d(tempo)[0])
    return beat_times, downbeat_times, ts, bpm, (y, sr)


class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(self, audio_url: str = Input(description="URL to the track mp3/wav")) -> dict:
        path = _download(audio_url)
        engine = None
        beat_times = downbeat_times = None
        time_sig = 4
        bpm = 0.0
        peak = None
        yz = None
        try:
            beat_times, downbeat_times, time_sig, bpm = _detect_madmom(path)
            engine = "madmom"
        except Exception:
            print("madmom failed:\n" + traceback.format_exc())
            try:
                beat_times, downbeat_times, time_sig, bpm, yz = _detect_librosa(path)
                engine = "librosa"
            except Exception:
                print("librosa failed:\n" + traceback.format_exc())
                import detect_fallback, soundfile as sf
                y, sr = sf.read(path)
                r = detect_fallback.detect(y, sr)
                return {**r, "engine": "numpy-fallback"}

        # onset / drop peak
        try:
            if yz is None:
                import librosa
                yz = librosa.load(path, sr=None, mono=True)
            peak = _onset_peak_librosa(yz[0], yz[1])
        except Exception:
            peak = float(downbeat_times[len(downbeat_times) // 2]) if len(downbeat_times) else None

        return {
            "bpm": round(float(bpm), 2),
            "beat_times": [round(float(t), 3) for t in beat_times],
            "downbeat_times": [round(float(t), 3) for t in downbeat_times],
            "time_signature": int(time_sig),
            "onset_envelope_peak_s": round(float(peak), 3) if peak is not None else None,
            "engine": engine,
        }
