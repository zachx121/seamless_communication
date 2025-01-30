import torch
from pathlib import Path
from fairseq2.data import SequenceData
from fairseq2.data.audio import WaveformToFbankConverter
import logging
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


class DefualtArgs:
    # 'arb,ben,cat,ces,cmn,cym,dan,deu,eng,est,fin,fra,hin,ind,ita,jpn,kan,kor,mlt,nld,pes,pol,por,ron,rus,slk,spa,swe,swh,tam,tel,tgl,tha,tur,ukr,urd,uzn,vie'
    tgt_lang = 'eng'
    duration_factor=1.0  # 这是时长控制，即x0.8是变快速、x1.2是变慢速
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
    unit_generation_beam_size=5
    unit_generation_max_len_a=25
    unit_generation_max_len_b=50
    unit_generation_ngram_blocking=False
    unit_generation_ngram_filtering=False
    vocoder_name='vocoder_pretssel'


class ExpressiveModel:
    def __init__(self, inp_args=DefualtArgs, sample_rate=16000):
        self.args=inp_args
        self.sample_rate = sample_rate
        add_gated_assets(self.args.gated_model_dir)
        unit_tokenizer = load_unity_unit_tokenizer(self.args.model_name)
        # 通用Translator框架 (文本->文本、音频->文本、文本->音频 etc.)
        self.translator = Translator(
                self.args.model_name,
                vocoder_name_or_card=None,
                device=device,
                dtype=dtype,
            )

        self.pretssel_generator = PretsselGenerator(
                self.args.vocoder_name,
                vocab_info=unit_tokenizer.vocab_info,
                device=device,
                dtype=dtype,
            )


        self.fbank_extractor = WaveformToFbankConverter(
                num_mel_bins=80,
                waveform_scale=2**15,
                channel_last=True,
                standardize=False,
                device=device,
                dtype=dtype,
            )
        _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(self.args.vocoder_name)
        self.gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
        self.gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

        self.text_generation_opts, self.unit_generation_opts = set_generation_opts(self.args)

    @staticmethod
    def remove_prosody_tokens_from_text(text: str) -> str:
        # filter out prosody tokens, there is only emphasis '*', and pause '='
        text = text.replace("*", "").replace("=", "")
        text = " ".join(text.split())
        return text

    def predict(self, wav, duration_factor=None, tgt_lang=None,src_lang=None):
        duration_factor = self.args.duration_factor if duration_factor is None else duration_factor
        tgt_lang = self.args.tgt_lang if tgt_lang is None else tgt_lang
        data = self.fbank_extractor(
            {
                "waveform": wav,
                "sample_rate": self.sample_rate,
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
        # 如果传了src_lang，进行一次asr
        asr_text=None
        if src_lang:
            asr_texts = self.translator.asr_predict(
                input=src,            # 输入数据，通常是音频数据（如Tensor或音频文件路径）
                tgt_lang=src_lang,    # 目标语言，通常是源语言，或者是为特定 ASR 任务需要的语言
                sample_rate=16000     # 音频采样率，常见的如16000
            )
            if asr_texts :
                asr_text = self.remove_prosody_tokens_from_text(str(asr_texts[0]))

        text_output, unit_output = self.translator.predict(
            input=src,
            task_str="s2st",
            tgt_lang=tgt_lang,
            text_generation_opts=self.text_generation_opts,
            unit_generation_opts=self.unit_generation_opts,
            unit_generation_ngram_filtering=self.args.unit_generation_ngram_filtering,
            duration_factor=duration_factor,
            prosody_encoder_input=src_gcmvn,  # 通过配置src_gcmvn应该可以实现指定音色，这是个字典，有一个key存到tensor
        )
        
        text_output, unit_output = self.translator.predict(
            input=src,
            task_str="s2st",
            tgt_lang=tgt_lang,
            text_generation_opts=self.text_generation_opts,
            unit_generation_opts=self.unit_generation_opts,
            unit_generation_ngram_filtering=self.args.unit_generation_ngram_filtering,
            duration_factor=duration_factor,
            prosody_encoder_input=src_gcmvn,  # 通过配置src_gcmvn应该可以实现指定音色，这是个字典，有一个key存到tensor
        )
        speech_output = self.pretssel_generator.predict(
            units=unit_output.units,
            tgt_lang=tgt_lang,
            prosody_encoder_input=src_gcmvn,
        )

        wav_arr = speech_output.audio_wavs[0][0].to(torch.float32).cpu().numpy()
        wav_sr = speech_output.sample_rate
        text = self.remove_prosody_tokens_from_text(str(text_output[0]))
        if asr_text :
            return wav_arr, wav_sr, text,asr_text
        else :
            return wav_arr, wav_sr, text

    def predict_file(self, in_file, duration_factor=None, tgt_lang=None):
        import torchaudio
        wav, sample_rate = torchaudio.load(in_file)
        # 原音频信息 torch.Size([2, 334420]) torch.float32 44100
        logging.debug("原音频信息", wav.shape, wav.dtype, sample_rate)
        wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)
        # wav[1] = wav[0]  # Debug 尝试做成单通道模拟双通道的模式？
        # 重采样音频信息 torch.Size([2, 121332]) torch.float32
        logging.debug("重采样音频信息", wav.shape, wav.dtype)
        wav = wav.transpose(0, 1)
        # 最终输入音频信息 torch.Size([121332, 2]) torch.float32
        logging.debug("最终输入音频信息", wav.shape, wav.dtype)
        return self.predict(wav, duration_factor=duration_factor, tgt_lang=tgt_lang)


