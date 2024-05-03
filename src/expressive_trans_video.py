import torch
import torchaudio
from pathlib import Path
from fairseq2.data import SequenceData
from fairseq2.data.audio import WaveformToFbankConverter

from seamless_communication.cli.expressivity.predict.pretssel_generator import (
    PretsselGenerator,
)
from seamless_communication.cli.m4t.predict import (
    add_inference_arguments,
    set_generation_opts,
)
from seamless_communication.inference import Translator
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.store import add_gated_assets

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# in_file = "/root/autodl-fs/audio_samples/小杨哥带货.m4a"
# # in_file="/root/autodl-fs/audio_samples/董宇辉带货.m4a"
# # in_file="/root/autodl-fs/audio_samples/现场台词.mp3"
# # in_file="/root/autodl-fs/audio_samples/小Lin说.m4a"
# wav, sample_rate = torchaudio.load(in_file)
# print("原音频信息", wav.shape, wav.dtype, sample_rate)
# # display(Audio(in_file, rate=16000, autoplay=False, normalize=True))
# wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)
# # wav[1] = wav[0]  # Debug 尝试做成单通道模拟双通道的模式？
# print("重采样音频信息", wav.shape, wav.dtype)
# wav = wav.transpose(0, 1)
# print("最终输入音频信息", wav.shape, wav.dtype)


def remove_prosody_tokens_from_text(text: str) -> str:
    # filter out prosody tokens, there is only emphasis '*', and pause '='
    text = text.replace("*", "").replace("=", "")
    text = " ".join(text.split())
    return text

def process(wav):
    # !expressivity_predict {in_file} --tgt_lang eng \
    # --model_name seamless_expressivity --vocoder_name vocoder_pretssel \
    # --gated-model-dir /root/autodl-tmp/SeamlessExpressive --output_path {out_file}

    class args:
        duration_factor = 0.8
        gated_model_dir = Path('/root/autodl-fs/SeamlessExpressive')
        model_name = 'seamless_expressivity'
        no_repeat_ngram_size = 4
        src_lang = None
        task = None
        text_generation_beam_size = 5
        text_generation_max_len_a = 1
        text_generation_max_len_b = 200
        text_generation_ngram_blocking = False
        text_unk_blocking = False
        tgt_lang = 'eng'
        unit_generation_beam_size = 5
        unit_generation_max_len_a = 25
        unit_generation_max_len_b = 50
        unit_generation_ngram_blocking = False
        unit_generation_ngram_filtering = False
        vocoder_name = 'vocoder_pretssel'

    add_gated_assets(args.gated_model_dir)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    translator = Translator(
        args.model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )

    pretssel_generator = PretsselGenerator(
        args.vocoder_name,
        vocab_info=unit_tokenizer.vocab_info,
        device=device,
        dtype=dtype,
    )


    fbank_extractor = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2 ** 15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype,
    )
    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(args.vocoder_name)
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)
    data = fbank_extractor(
        {
            "waveform": wav,
            "sample_rate": 16000,
        }
    )
    fbank = data["fbank"]
    gcmvn_fbank = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
    std, mean = torch.std_mean(fbank, dim=0)
    fbank = fbank.subtract(mean).divide(std)

    src = SequenceData(
        seqs=fbank.unsqueeze(0),
        seq_lens=torch.LongTensor([fbank.shape[0]]),
        is_ragged=False,
    )
    src_gcmvn = SequenceData(
        seqs=gcmvn_fbank.unsqueeze(0),
        seq_lens=torch.LongTensor([gcmvn_fbank.shape[0]]),
        is_ragged=False,
    )

    print(f">>> src_gcmvn is {src_gcmvn}")

    # src_gcmvn['seqs'] = src_gcmvn['seqs']*0.7

    text_generation_opts, unit_generation_opts = set_generation_opts(args)
    text_output, unit_output = translator.predict(
        src,
        "s2st",
        args.tgt_lang,
        text_generation_opts=text_generation_opts,
        unit_generation_opts=unit_generation_opts,
        unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
        duration_factor=args.duration_factor,
        prosody_encoder_input=src_gcmvn,
    )

    print(f">>> src_gcmvn is {src_gcmvn}")
    print(f">>> text_output len is {len(str(text_output[0]))}")
    print(f">>> unit_output len is {len(unit_output.units[0])}")
    print(f">>> text with prosody:\n{text_output}")
    text = remove_prosody_tokens_from_text(str(text_output[0]))
    print(f">>> text w/o prosody:\n{text}")


    speech_output = pretssel_generator.predict(
        unit_output.units,
        tgt_lang=args.tgt_lang,
        prosody_encoder_input=src_gcmvn,
    )

    wav_arr = speech_output.audio_wavs[0][0].to(torch.float32).cpu().numpy()
    print(wav_arr.shape,wav_arr.dtype,speech_output.sample_rate)
    return wav_arr, speech_output.sample_rate, text
    # audio_play = Audio(wav_arr, rate=speech_output.sample_rate, autoplay=False, normalize=True)
    # display(audio_play)