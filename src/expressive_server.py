
import torch
import numpy as np
import logging
import base64
import json
from flask import Flask, request
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
from expressive_model import ExpressiveModel

if __name__ == '__main__':
    app = Flask(__name__, static_folder="/home/zhoutong", static_url_path="")
    M = ExpressiveModel()


    # POST | 如果是json
    @app.route("/test_json", methods=['POST'])
    def test_json():
        if request.method != "POST":
            return None

        info = request.get_json()

        audio_buffer, audio_sr = base64.b64decode(info['buffer']), info['sample_rate']
        assert audio_sr == 16000  # 暂时强制要求采样率，后端这里不做重采样
        dtype = np.int16  # np.float32 or np.int16
        audio_arr = np.frombuffer(audio_buffer, dtype=dtype)
        # M.predict要求是双通道float32的音频，而输入是单通道int16
        if len(audio_arr.shape) == 1:
            # audio_arr = np.vstack([audio_arr, audio_arr])
            audio_arr = np.expand_dims(audio_arr, axis=0)
        audio_arr = audio_arr.astype(np.float32) / 32768.0  # 归一化到 [-1.0, 1.0]
        audio_arr = torch.from_numpy(audio_arr.T)
        wav_arr, wav_sr, text_cstr = M.predict(audio_arr,
                                               duration_factor=info.get('duration_factor', None),
                                               tgt_lang=info.get('tgt_lang', None))
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


