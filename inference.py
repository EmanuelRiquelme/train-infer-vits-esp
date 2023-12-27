import utils
from models import SynthesizerTrn
from phonemizer import phonemize
from unidecode import unidecode
import soundfile as sf
from text.symbols import symbols
import re
import commons
import torch
import argparse

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

parser = argparse.ArgumentParser(description="Inference script with parameters")
parser.add_argument('--json_file', type=str, required=True, help='Path to JSON file containing the parameters of the model')
parser.add_argument('--model_path', type=str, required=True, help='Path to model')
parser.add_argument('--audio_output_name', type=str, required=True, help='Name of the audio output file')
parser.add_argument('--text', type=str, required=True, help='Text to be processed')

_whitespace_re = re.compile(r'\s+')

def convert_to_ascii(text):
  return unidecode(text)

def lowercase(text):
  return text.lower()

def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)

def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = phonemize(text, language='es', backend='espeak', strip=True)
  text = collapse_whitespace(text)
  return text


def tex2seq(text):
    clean_text = transliteration_cleaners(text)
    sequence = []
    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id]
    return sequence

def get_text(text, hps):
    text_norm = tex2seq(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def save_audio(array,output_name = 'logs/ljs_base/test2.wav'):
    max_wav_value = 32768.0
    normalized_audio_data = array / max_wav_value
    sf.write(output_name, array, 22500)

def inference(args):
    hps = utils.get_hparams_from_file(args.json_file)
    model_path = args.model_path 
    stn_tst = args.text
    output_name = args.audio_output_name
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
        ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    stn_tst = get_text(stn_tst, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    save_audio(audio,output_name = output_name)
    print(f'audio file {output_name} saved')
if __name__ == '__main__':
    args = parser.parse_args()
    inference(args)
