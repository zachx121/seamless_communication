# 启动的时候: conda deactivate && python expressive_server.py
# conda下有一个异常无法处理，OSError: libsndfile is not found! Since you are in a Conda environment, use `conda install -c conda-forge libsndfile==1.0.31` to install it
import time

import torch
import numpy as np
import logging
import librosa
import base64
import json
import utils_audio
from flask import Flask, request
logging.basicConfig(format='[%(asctime)s-%(levelname)s-%(process)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
from expressive_model import ExpressiveModel
import multiprocessing as mp

if not torch.cuda.is_available():
    logging.error(">>>>>CUDA Not Available<<<<"+"\n"*4)

app = Flask(__name__, static_folder="/home/zhoutong", static_url_path="")

class Param:
    trace_id: str = ""
    buffer: bytes = None  # 音频按base64编码的字节流，会使用base64.b64decode(p.buffer)解码
    buffer_dtype: str = "int16"
    sample_rate: float = 16000
    speed: float = 1.0  # 合成音频的语速 (e.g. 1.2加速 0.8减速)
    lang: str = None  # 合成音频的语言 (e.g. zh_cn/en_us/fr_fr/es_es)

    @property
    def dtype(self):
        dtype_map = {"int16": np.int16, "float32": np.float32}
        return dtype_map.get(self.buffer_dtype, None)

    # 模型接收的参数是“时长” 所以入参是"速度"就需要改一下
    @property
    def duration_factor(self):
        return round(1 / self.speed, 4)

    # 模型接收的语言参数名和通用的不一样，重新映射
    @property
    def tgt_lang(self):
        # eng/cmn/eng/deu/fra/ita/spa
        language_map = {"zh_cn": "cmn", "en_us": "eng", "fr_fr": "fra", "es_es": "spa"}
        assert self.lang in language_map, f"语言参数错误: lang='{self.lang}'，只接受 {'/'.join(language_map.keys())}"
        return language_map[self.lang]

    def __init__(self, info_dict):
        for key in self.__annotations__.keys():
            if key in info_dict:
                setattr(self, key, info_dict[key])

        # 暂时强制要求音频类型为16khz、int16，后端这里不做重采样
        assert self.dtype == np.int16 and self.sample_rate == 16000, "暂时强制要求音频类型为16khz、int16，后端这里不做重采样"


# POST | 如果是json
@app.route("/test_json", methods=['POST'])
def test_json():
    logging.debug(f"Start connection.. start-time: {time.time()}")
    p = Param(request.get_json())
    logging.debug(f"Start Multi-Process Prediction of tid='{p.trace_id}' start-time: {time.time()}")
    queue_in.put(p)
    result = queue_out.get()
    logging.debug(f"Finish Multi-Process Prediction of tid='{p.trace_id}' start-time: {time.time()}")
    return result


def model_process(q_in, q_out):
    M = ExpressiveModel()
    while True:
        # data是M.predict的参数字典，例如 {"wav":np.array, "duration":..}
        p:Param = q_in.get()
        if p is None:
            break
        audio_arr = np.frombuffer(base64.b64decode(p.buffer), dtype=p.dtype)
        # utils_audio.save_audio(audio_arr, 16000, f"./tmp_{p.trace_id}.wav")
        # M.predict要求是双通道float32的音频，而输入是单通道int16
        if len(audio_arr.shape) == 1:
            # audio_arr = np.vstack([audio_arr, audio_arr])
            audio_arr = np.expand_dims(audio_arr, axis=0)
        audio_arr = audio_arr.astype(np.float32) / 32768.0  # 归一化到 [-1.0, 1.0]
        audio_arr = torch.from_numpy(audio_arr.T)
        logging.debug(f"    Start Predict of tid='{p.trace_id}'")
        wav_arr, wav_sr, text_cstr = M.predict(wav=audio_arr,
                                               duration_factor=p.duration_factor,
                                               tgt_lang=p.tgt_lang)
        logging.debug(f"    Finish Predict of tid='{p.trace_id}'")
        logging.debug(f"    Start resample&int16 of tid='{p.trace_id}'")
        wav_16khz = librosa.resample(wav_arr, orig_sr=wav_sr, target_sr=16000)
        wav_int16 = (np.clip(wav_16khz, -1.0, 1.0) * 32767).astype(np.int16)
        logging.debug(f"    Finish resample&int16 of tid='{p.trace_id}' end-time: {time.time()}")
        rsp = {"trace_id": p.trace_id,
               # "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),  # float32 & 24khz
               "audio_buffer_int16": base64.b64encode(wav_int16.tobytes()).decode(),  # int16 & 16khz
               "sample_rate": 16000,
               "audio_text": str(text_cstr),
               "status": "0",
               "msg": "success."}
        logging.info(f"    audio_text of tid='{p.trace_id}': '{rsp['audio_text']}' (sr={rsp['sample_rate']})")
        rsp = json.dumps(rsp)
        q_out.put(rsp)


if __name__ == '__main__':
    PROCESS_NUM = 4  # 3090 24GB
    mp.set_start_method("spawn")
    queue_in = mp.Queue()
    queue_out = mp.Queue()
    process_list = []
    for _ in range(PROCESS_NUM):
        p = mp.Process(target=model_process, args=(queue_in, queue_out))
        process_list.append(p)
        p.start()

    app.run(host="0.0.0.0", port=6006)

    # 给队列发送结束信号
    for _ in range(4):
        queue_in.put(None)
        queue_out.put(None)
    # 等待子进程完成并退出
    for p in process_list:
        p.join()
