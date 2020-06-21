import matplotlib
# %matplotlib inline
import matplotlib.pylab as plt
# import matplotlib.pylab as plt

#import IPython.display as ipd
import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import wave
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')

#%% md

#### Setup hparams

#%%

hparams = create_hparams()
hparams.sampling_rate = 22050

#%% md

#### Load model from checkpoint

#%%

checkpoint_path = "checkpoints/tacotron2_statedict.pt"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

#%% md

#### Load WaveGlow for mel2audio synthesis and denoiser

#%%

waveglow_path = 'checkpoints/waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

#%% md

#### Prepare text input

#%%

text = "I love you, Weishing, my wife! "
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

#%% md

#### Decode text input and plot results

#%%

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
# plot_data((mel_outputs.float().data.cpu().numpy()[0],
#            mel_outputs_postnet.float().data.cpu().numpy()[0],
#            alignments.float().data.cpu().numpy()[0].T))

#%% md

#### Synthesize audio from spectrogram using WaveGlow

#%%

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)


def _get_normalization_factor(max_abs_value, normalize):
    if not normalize and max_abs_value > 1:
        raise ValueError('Audio data must be between -1 and 1 when normalize=False.')
    return max_abs_value if normalize else 1


def _validate_and_normalize_with_numpy(data, normalize):
    import numpy as np

    data = np.array(data, dtype=float)
    if len(data.shape) == 1:
        nchan = 1
    elif len(data.shape) == 2:
        # In wave files,channels are interleaved. E.g.,
        # "L1R1L2R2..." for stereo. See
        # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
        # for channel ordering
        nchan = data.shape[0]
        data = data.T.ravel()
    else:
        raise ValueError('Array audio input must be a 1D or 2D array')

    max_abs_value = np.max(np.abs(data))
    normalization_factor = _get_normalization_factor(max_abs_value, normalize)
    scaled = data / normalization_factor * 32767
    return scaled.astype('<h').tostring(), nchan


d = audio[0].data.cpu().numpy()
scaled, nchan = _validate_and_normalize_with_numpy(d, True)
# print(hparams.sampling_rate)
# # librosa.output.write_wav("output/output4.wav", audio[0].data.cpu().numpy(), hparams.sampling_rate)
with wave.open("output/process.wav", 'wb') as wf:
    wf.setnchannels(1)
    wf.setframerate(hparams.sampling_rate)
    wf.setsampwidth(2)
    wf.setcomptype('NONE','NONE')
    wf.writeframes(scaled)


# ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

#%% md

#### (Optional) Remove WaveGlow bias

#%%

audio_denoised = denoiser(audio, strength=0.01)[:, 0]
# ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
