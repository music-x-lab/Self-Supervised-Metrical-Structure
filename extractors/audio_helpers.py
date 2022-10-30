from mir import io
from mir.extractors import ExtractorBase
import numpy as np
import librosa


def phase_vocoder(D, time_steps, hop_length=None):
    # adapted from librosa

    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    # time_steps = np.arange(0, D.shape[1], rate, dtype=np.float)

    # Create an empty output array
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):

        columns = D[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = ((1.0 - alpha) * np.abs(columns[:, 0])
               + alpha * np.abs(columns[:, 1]))

        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (np.angle(columns[:, 1])
                  - np.angle(columns[:, 0])
                  - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch

def time_stretch(y, time_steps, hop_length, energy_spec_only):

    win_length = hop_length * 4
    n_fft = hop_length * 4
    # adapted from librosa
    import librosa.core as core

    # Construct the short-term Fourier transform (STFT)
    stft = core.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    time_steps = time_steps*stft.shape[1]
    time_steps[time_steps<0]=0
    time_steps[time_steps>=stft.shape[1]]=stft.shape[1]
    # Stretch by phase vocoding
    stft_stretch = phase_vocoder(stft, time_steps)

    if (energy_spec_only):
        return None, np.abs(stft_stretch).T
    # Invert the STFT
    y_stretch = core.istft(
        stft_stretch, dtype=y.dtype, hop_length=hop_length, win_length=win_length)

    return y_stretch, np.abs(stft_stretch).T

class BeatSyncAudio(ExtractorBase):

    def get_feature_class(self):
        return io.MusicIO

    def extract(self, entry, **kwargs):
        beat_div = kwargs['beat_div']
        subbeat_div = kwargs['subbeat_div']
        beat = np.array(entry.beat)
        audio_length = len(entry.music) / entry.prop.sr
        subbeat = np.interp(np.arange(len(beat) * beat_div * subbeat_div) / (beat_div * subbeat_div), np.arange(len(beat)), beat[:, 0])
        time_steps = subbeat / audio_length
        hop_length = entry.prop.hop_length
        y, stft_stretch = time_stretch(entry.music, time_steps, hop_length, False)
        return y

class BeatSyncEnergySpec(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self, entry, **kwargs):
        beat_div = kwargs['beat_div']
        subbeat_div = kwargs['subbeat_div']
        beat = np.array(entry.beat)
        audio_length = len(entry.music) / entry.prop.sr
        subbeat = np.interp(np.arange(len(beat) * beat_div * subbeat_div) / (beat_div * subbeat_div), np.arange(len(beat)), beat[:, 0])
        time_steps = subbeat / audio_length
        hop_length = entry.prop.hop_length
        _, stft_stretch = time_stretch(entry.music, time_steps, hop_length, True)
        return stft_stretch

class STFT(ExtractorBase):

    def get_feature_class(self):
        return io.SpectrogramIO

    def extract(self,entry,**kwargs):
        if('source' in kwargs):
            music=entry.dict[kwargs['source']].get(entry)
        else:
            music=entry.music
        result=librosa.core.stft(
            music,
            n_fft=entry.prop.win_size,
            hop_length=entry.prop.hop_length,
        ).T
        return abs(result).astype(np.float16)
