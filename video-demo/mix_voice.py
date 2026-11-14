"""Build the final demo video: voices + sound effects + a soft music bed, on the no-voice render.

Run:  .venv-audio/bin/python video-demo/mix_voice.py      (from the project root)
Needs: voice/*-raw.m4a (kept out of git), FairView-Demo-NoVoice.mp4, ffmpeg, VoiceFixer.
Writes: FairView-Demo-Final.mp4

Plain-English summary
---------------------
1. Clean every recording with VoiceFixer (an AI model that removes noise and rebuilds the
   high frequencies a compressed phone recording throws away), then level both voices to
   the same loudness so Kabir and Vishnu sound like they're in the same room.
2. Cut each recording to its lines and place it at the right moment in the video (offsets
   found from word timestamps, so every line starts on its cue).
3. Add sound effects (Pixabay licence, free to use) on the things you see happen.
4. Make a quiet original music bed (just soft chords, generated here, no copyright) that
   ducks down whenever someone talks.
"""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "voice"
SFX = Path.home() / ".claude/skills/media-use/audio/assets/sfx"
SR = 48000
LENGTH = 118.0

# recording -> (cut start s, cut end s, where it starts in the video s, gain dB)
CLIPS = {
    "vishnu-1": (2.40, 22.30, 0.00, 0),
    # Kabir's re-recorded lines (phone mic, lossless), one clip per line so each starts on its cue
    "kabir2-pt1#1": (4.69, 9.00, 20.00, 0),
    "kabir2-pt1#2": (10.07, 15.70, 25.00, 0),
    "kabir2-pt1#3": (17.11, 22.50, 32.00, 0),
    "kabir2-pt1#4": (23.19, 31.30, 38.00, 0),
    "kabir2-pt1#5": (33.31, 38.60, 48.00, 0),
    "kabir2-pt1#6": (38.91, 41.20, 54.05, 0),
    "kabir2-pt1#7": (42.13, 46.10, 57.00, 0),
    "kabir2-pt1#8": (47.73, 50.30, 62.85, 0),
    "kabir2-pt1#9": (50.83, 54.70, 66.00, 0),
    "kabir2-pt2#1": (6.15, 11.50, 103.85, 0),
    "kabir2-pt2#2": (16.19, 19.75, 114.00, 0),
    "vishnu-2": (2.40, 34.20, 70.82, 0),
    # "vishnu-3" (the line said together) is left out: two voices on top of each other blurred "isn't", so Kabir says it solo
}
# sound effect -> list of (video time s, gain dB)
EFFECTS = {
    "whoosh-cinematic.mp3": [(0.0, -16)],
    "whoosh-short.mp3": [(7.8, -14), (19.8, -14), (31.8, -14), (47.8, -14), (66.0, -18), (87.8, -14), (103.8, -14)],
    "pop.mp3": [(11.6, -10)],                                   # APPROVE stamp
    "error.mp3": [(14.4, -16)],                                 # DENY stamp
    "impact-bass-1.mp3": [(41.3, -12)],                         # 3.0x
    "click.mp3": [(54.1, -8), (59.1, -8), (62.9, -8), (74.1, -8), (77.1, -8), (82.0, -8)],   # real demo clicks
    "impact-bass-2.mp3": [(83.0, -14)],                         # 1 in 9
    "key-press.mp3": [(94.9, -12), (95.9, -12), (96.9, -12)],   # bars rising
    "chime.mp3": [(110.9, -12)],                                # as the slogan appears on screen
}


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


LIGHT = "aresample=48000:filter_size=256:phase_shift=10:cutoff=0.97,highpass=f=80,afftdn=nr=6:nf=-50,equalizer=f=250:t=q:w=1.0:g=-1.5,equalizer=f=3000:t=q:w=0.9:g=2,deesser=i=0.3"
LEVEL = "acompressor=threshold=-24dB:ratio=2.5:attack=8:release=200:makeup=4,loudnorm=I=-16:TP=-2:LRA=7"
SPLIT_HZ = 5500


def load48(path):
    """Any audio file -> mono 48 kHz float array, resampled with a long high-quality filter (no aliasing / glitches)."""
    tmp = V / "_tmp48.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-af", "aresample=48000:filter_size=256:phase_shift=10:cutoff=0.97", "-ac", "1", str(tmp)])
    x, _ = sf.read(tmp)
    tmp.unlink()
    return x


def clean(name):
    """Vishnu: light cleanup only (his recording is already good).
    Kabir (compressed 24 kHz phone recording): a split. Below 5.5 kHz, where words and
    consonants live, keep his REAL voice, gently denoised. Above 5.5 kHz, add only the
    brightness that VoiceFixer (AI) rebuilt. That way the AI can't blur any words."""
    raw, out = V / f"{name}-raw.m4a", V / f"{name}-ready.wav"
    if not name.startswith("kabir-"):   # Vishnu and Kabir's lossless re-record: light cleanup only
        match = ",highshelf=f=6000:g=-6" if name.startswith("kabir2") else ""   # Kabir's phone mic is brighter; match Vishnu
        run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", LIGHT + match + "," + LEVEL, "-ar", str(SR), "-ac", "1", str(out)])
        return out
    vf = V / f"{name}-vf.wav"
    if not vf.exists():
        tmp = V / f"{name}-in.wav"
        run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-ar", "44100", "-ac", "1", str(tmp)])
        from voicefixer import VoiceFixer
        VoiceFixer().restore(input=str(tmp), output=str(vf), cuda=False, mode=0)
        tmp.unlink()
    lo_f, hi_f = V / f"{name}-lo.wav", V / f"{name}-hi.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af",
         f"aresample=48000:filter_size=256:phase_shift=10:cutoff=0.97,highpass=f=80,arnndn=m={V / 'rnnoise-std.rnnn'}:mix=0.55,afftdn=nr=6:nf=-48,"
         f"equalizer=f=250:t=q:w=1.0:g=-2.5,equalizer=f=3000:t=q:w=0.9:g=3,lowpass=f={SPLIT_HZ},lowpass=f={SPLIT_HZ}", "-ac", "1", str(lo_f)])
    run(["ffmpeg", "-y", "-v", "error", "-i", str(vf), "-af",
         f"aresample=48000:filter_size=256:phase_shift=10:cutoff=0.97,highpass=f={SPLIT_HZ},highpass=f={SPLIT_HZ}", "-ac", "1", str(hi_f)])
    lo, hi = sf.read(lo_f)[0], sf.read(hi_f)[0]
    # line the two up exactly (VoiceFixer can shift by a few ms)
    k = min(len(lo), len(hi), SR * 15)
    a, b = np.diff(lo[:k]), np.diff(hi[:k])
    xc = np.fft.irfft(np.fft.rfft(b, 2 * k) * np.conj(np.fft.rfft(a, 2 * k)))
    lag = int(np.argmax(np.r_[xc[-960:], xc[:960]])) - 960
    hi = np.roll(hi, -lag)
    n = min(len(lo), len(hi))
    mix_f = V / f"{name}-split.wav"
    sf.write(mix_f, lo[:n] + 0.4 * hi[:n], SR)   # 0.4: matches Vishnu's brightness
    run(["ffmpeg", "-y", "-v", "error", "-i", str(mix_f), "-af", "deesser=i=0.3," + LEVEL, "-ar", str(SR), "-ac", "1", str(out)])
    return out


def place(track, path, at, gain_db, start=0.0, end=None, fade=0.08):
    x = load48(path)
    x = x[int(start * SR): None if end is None else int(end * SR)].copy()
    n = int(fade * SR)
    if len(x) > 2 * n:
        x[:n] *= np.linspace(0, 1, n)
        x[-n:] *= np.linspace(1, 0, n)
    i = int(at * SR)
    x = x[: max(0, len(track) - i)]
    track[i: i + len(x)] += x * 10 ** (gain_db / 20)


def music_bed():
    """Soft original chords (Cmaj7 - Am7 - Fmaj7 - G6), 8 s each, slow swell. No samples, no copyright."""
    chords = [[261.6, 329.6, 392.0, 493.9], [220.0, 261.6, 329.6, 392.0], [174.6, 220.0, 261.6, 329.6], [196.0, 246.9, 293.7, 329.6]]
    t = np.arange(int(LENGTH * SR)) / SR
    bed = np.zeros_like(t)
    seg = 8.0
    for k in range(int(LENGTH // seg) + 2):
        notes = chords[k % 4]
        s, e = k * seg - 1.5, (k + 1) * seg + 1.5          # overlap for a smooth crossfade
        m = (t >= s) & (t < e)
        tt = t[m] - s
        env = np.sin(np.pi * np.clip(tt / (e - s), 0, 1)) ** 2
        for f in notes:
            for detune in (-0.6, 0.6):                       # two slightly detuned voices = warm pad
                bed[m] += env * np.sin(2 * np.pi * (f + detune) * tt) * 0.08
            bed[m] += env * np.sin(2 * np.pi * f / 2 * tt) * 0.05   # soft octave below
    return bed / np.abs(bed).max() * 0.5


def main():
    n = int(LENGTH * SR)
    voice, fx = np.zeros(n), np.zeros(n)
    for key, (a, b, at, g) in CLIPS.items():
        place(voice, clean(key.split("#")[0]), at, g, start=a, end=b)
    for f, hits in EFFECTS.items():
        for at, g in hits:
            place(fx, SFX / f, at, g)
    sf.write(V / "voice-track.wav", voice, SR)
    sf.write(V / "fx-track.wav", fx, SR)
    sf.write(V / "bed-track.wav", music_bed(), SR)

    # Mix: bed ducks under the voices (sidechain), effects on top, then broadcast-style loudness.
    mix = V / "final-mix.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(V / "voice-track.wav"), "-i", str(V / "bed-track.wav"), "-i", str(V / "fx-track.wav"),
         "-filter_complex",
         "[0]asplit=2[v][sc];"
         "[1]lowpass=f=2500,volume=-26dB,afade=t=in:d=2,afade=t=out:st=114:d=4[bedq];"
         "[bedq][sc]sidechaincompress=threshold=0.03:ratio=4:attack=60:release=900[bedd];"
         "[v][bedd][2]amix=inputs=3:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=9[out]",
         "-map", "[out]", "-ar", str(SR), "-ac", "2", str(mix)])
    out = ROOT / "FairView-Demo-Final.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(ROOT / "FairView-Demo-NoVoice.mp4"), "-i", str(mix),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)])
    print("->", out)


if __name__ == "__main__":
    main()
