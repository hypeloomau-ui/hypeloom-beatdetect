"""Reference beat detector (pure numpy). Validates the data contract + downstream
assembly math, and is the zero-dependency fallback if madmom AND librosa both fail
to load inside the Replicate model.
Contract: (mono float32 y, sr) -> {bpm, beat_times, downbeat_times, time_signature, onset_envelope_peak_s}"""
import numpy as np

def _stft_mag(y, n_fft=2048, hop=512):
    win = np.hanning(n_fft).astype(np.float32)
    if len(y) < n_fft:
        return np.zeros((0, n_fft//2+1), np.float32)
    n = 1 + (len(y) - n_fft)//hop
    out = np.empty((n, n_fft//2+1), np.float32)
    for i in range(n):
        out[i] = np.abs(np.fft.rfft(y[i*hop:i*hop+n_fft]*win))
    return out

def onset_envelope(y, sr, hop=512):
    S = _stft_mag(y, hop=hop)
    if S.shape[0] < 4:
        return np.zeros(1, np.float32)
    flux = np.diff(np.log1p(S), axis=0)
    flux[flux < 0] = 0.0
    env = flux.sum(axis=1)
    env = np.convolve(env, np.hanning(5)/np.hanning(5).sum(), mode='same')
    if env.max() > 0:
        env /= env.max()
    return np.concatenate([[0.0], env]).astype(np.float32)

def estimate_tempo(env, sr, hop, bpm_min=60, bpm_max=190):
    e = env - env.mean()
    ac = np.correlate(e, e, mode='full')[len(e)-1:]
    fps = sr/hop
    lag_min, lag_max = int(fps*60/bpm_max), min(int(fps*60/bpm_min), len(ac)-1)
    if lag_max <= lag_min:
        return 120.0
    lags = np.arange(lag_min, lag_max)
    bpms = 60.0*fps/lags
    prior = np.exp(-0.5*(np.log2(bpms/120.0)/0.9)**2)
    seg = ac[lag_min:lag_max]*prior
    return float(60.0*fps/lags[int(np.argmax(seg))])

def track_beats(env, sr, hop, bpm):
    fps = sr/hop
    period = 60.0/bpm*fps
    N = len(env)
    if N < 2:
        return np.array([0.0])
    tightness = 100.0
    backlink = -np.ones(N, int)
    cumscore = env.astype(np.float64).copy()
    window = np.arange(-int(2*period), -int(period/2)+1)
    for i in range(N):
        idx = i+window
        idx = idx[idx >= 0]
        if idx.size == 0:
            continue
        dt = (i-idx)/period
        txcost = -tightness*(np.log(np.maximum(dt,1e-9)))**2
        scores = cumscore[idx]+txcost
        j = int(np.argmax(scores))
        if scores[j]+env[i] > cumscore[i]:
            cumscore[i] = scores[j]+env[i]
            backlink[i] = idx[j]
    tail = int(np.argmax(cumscore[int(N*0.5):]))+int(N*0.5) if N > 4 else N-1
    beats, k, guard = [], tail, 0
    while k >= 0 and guard < N+5:
        beats.append(k); k = backlink[k]; guard += 1
    return np.array(sorted(beats), float)/fps

def find_downbeats(beat_times, env, sr, hop, ts=4):
    fps = sr/hop
    frames = np.clip((beat_times*fps).astype(int), 0, len(env)-1)
    e = env[frames]
    best_p, best_s = 0, -1
    for p in range(ts):
        s = e[p::ts].sum()
        if s > best_s:
            best_s, best_p = s, p
    return beat_times[best_p::ts]

def onset_peak_seconds(env, sr, hop, smooth_s=1.0):
    fps = sr/hop
    w = max(1, int(smooth_s*fps))
    sm = np.convolve(env, np.ones(w)/w, mode='same')
    return float(np.argmax(sm)/fps)

def detect(y, sr, hop=512):
    y = np.asarray(y, np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    env = onset_envelope(y, sr, hop)
    bpm = estimate_tempo(env, sr, hop)
    beats = track_beats(env, sr, hop, bpm)
    ts = 4
    downs = find_downbeats(beats, env, sr, hop, ts)
    peak = onset_peak_seconds(env, sr, hop)
    if len(beats) > 3:
        bpm = float(60.0/np.median(np.diff(beats)))
    return {"bpm": round(bpm,2),
            "beat_times": [round(float(b),3) for b in beats],
            "downbeat_times": [round(float(b),3) for b in downs],
            "time_signature": ts,
            "onset_envelope_peak_s": round(peak,3)}
