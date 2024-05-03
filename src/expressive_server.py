
import torchaudio
import torch
import numpy as np

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
import logging
import base64
from flask import Flask,request,render_template
import json

logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)



device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
class args:
    duration_factor=1.0  # speed nar 语速
    gated_model_dir=Path('/root/autodl-fs/SeamlessExpressive')
    model_name='seamless_expressivity'
    no_repeat_ngram_size=4
    src_lang=None
    task=None
    text_generation_beam_size=5
    text_generation_max_len_a=1
    text_generation_max_len_b=200
    text_generation_ngram_blocking=False
    text_unk_blocking=False
    tgt_lang='eng'
    unit_generation_beam_size=5
    unit_generation_max_len_a=25
    unit_generation_max_len_b=50
    unit_generation_ngram_blocking=False
    unit_generation_ngram_filtering=False
    vocoder_name='vocoder_pretssel'
add_gated_assets(args.gated_model_dir)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
class ExpressiveModel:
    def __init__(self):
        unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
        self.translator = Translator(
            args.model_name,
            vocoder_name_or_card=None,
            device=device,
            dtype=dtype,
        )

        self.pretssel_generator = PretsselGenerator(
            args.vocoder_name,
            vocab_info=unit_tokenizer.vocab_info,
            device=device,
            dtype=dtype,
        )

        self.fbank_extractor = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2 ** 15,
            channel_last=True,
            standardize=False,
            device=device,
            dtype=dtype,
        )

        _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(args.vocoder_name)
        self.gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
        self.gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

    def predict(self, wav, sample_rate):
        logging.debug("原音频信息, %s, %s, %s" % (wav.shape, wav.dtype, sample_rate))
        # wav, sample_rate = torchaudio.load(args.input)  # INPUT
        wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)
        logging.debug("重采样音频信息, %s, %s" % (wav.shape, wav.dtype))
        wav = wav.transpose(0, 1)
        logging.debug("最终输入音频信息, %s, %s" % (wav.shape, wav.dtype))

        data = self.fbank_extractor(
            {
                "waveform": wav,
                "sample_rate": 16000,
            }
        )
        fbank = data["fbank"]
        gcmvn_fbank = fbank.subtract(self.gcmvn_mean).divide(self.gcmvn_std)
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

        text_generation_opts, unit_generation_opts = set_generation_opts(args)
        text_output, unit_output = self.translator.predict(
            src,
            "s2st",
            args.tgt_lang,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
            duration_factor=args.duration_factor,
            prosody_encoder_input=src_gcmvn,
        )

        speech_output = self.pretssel_generator.predict(
            unit_output.units,
            tgt_lang=args.tgt_lang,
            prosody_encoder_input=src_gcmvn,
        )


        text = text_output[0]
        wav_arr = speech_output.audio_wavs[0][0].to(torch.float32).cpu().numpy()
        wav_sr = speech_output.sample_rate
        return wav_arr, wav_sr, text


if __name__ == '__main__':
    app = Flask(__name__, static_folder="/home/zhoutong", static_url_path="")
    M = ExpressiveModel()

    # POST | 如果是json
    @app.route("/test_json", methods=['POST'])
    def test_json():
        if request.method != "POST":
            return None

        data = request.get_data()
        info = json.loads(data)
        # print(info)
        audio_buffer = base64.b64decode(info['buffer'])
        sr = 16000
        dtype = np.int16  # np.float32 or np.int16
        audio_arr = np.frombuffer(audio_buffer, dtype=dtype)
        # M.predict要求是双通道float32的音频，而输入是单通道int16
        if len(audio_arr.shape) == 1:
            audio_arr = np.vstack([audio_arr, audio_arr])
        audio_arr = audio_arr.astype(np.float32) / 32768.0  # 归一化到 [-1.0, 1.0]
        audio_arr = torch.from_numpy(audio_arr)
        wav_arr, wav_sr, text_cstr = M.predict(audio_arr, sr)
        rsp = {"trace_id": "",
               "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),
               "sample_rate": wav_sr,
               "audio_text": str(text_cstr),
               "status": "0",
               "msg": "success."}
        print(rsp['audio_text'])
        rsp = json.dumps(rsp)
        return rsp


    app.run(host="0.0.0.0", port=6006)


