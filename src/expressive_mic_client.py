import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)
import sys

import pyaudio
import requests
import json
import base64
import audioop
import time
import numpy as np
import scipy

# 默认都用int16的音频流
CHANNEL=1
SAMPLE_RATE=16000
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNEL
# Function to make the POST request with audio buffer
def send_audio(buffer):
    url = "http://192.168.0.1:6006/test_json"
    url = "https://u212392-8904-1248e7f4.bjb1.seetacloud.com:8443/" + "test_json"
    data = {"buffer": base64.b64encode(buffer).decode(),
            "sample_rate": SAMPLE_RATE,
            "duration_factor": 1.2,
            "tgt_lang": "eng"}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url, data=json.dumps(data), headers=headers, timeout=5)
    if response.status_code != 200:
        logging.error(f"Request Error with code: {response.status_code}")
        with open("tmp.text", "w") as fw:
            fw.writelines(json.dumps(data))
        logging.error(f"send data saved in 'tmp.text'")
    else:
        rsp = json.loads(response.text)
        rsp_audio_arr = np.frombuffer(base64.b64decode(rsp['audio_buffer']), dtype=np.float32)
        fp = f"rsp_audio_{int(time.time())}.wav"
        scipy.io.wavfile.write(fp, rsp['sample_rate'], rsp_audio_arr)
        logging.info(f"同传完毕，音频写入文件'{fp}'，文本为: '{rsp['audio_text']}'")

# Callback function for audio input
TIME_TAG = -1
BUFFER_CACHE = b""
SAMPLE_SEC, PAUSE_SEC, RMS_HOLD = 0.05, 0.5, 500  # 停顿检测的时长、音量; 控制在0.05s执行一次call_back 注意采样频率快了rms_hold得降低
IS_APPENDING = False
def callback(in_data, frame_count, time_info, status):
    global TIME_TAG, BUFFER_CACHE, RMS_HOLD, IS_APPENDING
    rms = audioop.rms(in_data, 2)
    if rms >= RMS_HOLD:
        IS_APPENDING = True
        # 音量超过阈值时直接追加buffer和确定开始时间戳
        TIME_TAG = TIME_TAG if TIME_TAG != -1 else int(time.time())
        print(f"\rappending at {time.time():.0f}, rms: {rms}")
        BUFFER_CACHE += in_data
        # todo. 过长怎么办，例如追加后检查buffer时长超过五秒？
        #       - 找到离末尾最近的一个最小音量（低于15分位数？）时间点？
    else:
        # 音量不够大时会持续更新buffer指向最新的"空白音in_data"
        # 如果上一个segment还是在追加，那就把当前的in_data追加进去发请求，然后再更新buffer
        # 判断上一个seg的状态
        if IS_APPENDING:
            BUFFER_CACHE += in_data
            print(f"Save And Send... TIME_TAG:{TIME_TAG} buffer_len({len(BUFFER_CACHE)}):{len(BUFFER_CACHE)/BYTES_PER_SEC:.2f}s")
            scipy.io.wavfile.write(f"req_audio_{TIME_TAG}.wav", SAMPLE_RATE, np.frombuffer(BUFFER_CACHE, dtype=np.int16))
            send_audio(BUFFER_CACHE)
            IS_APPENDING = False
            TIME_TAG = -1
        # 指向空白音
        BUFFER_CACHE = in_data
    return None, pyaudio.paContinue

# 检查buffer末尾，倒查是否有持续达0.2s的停顿，音频输入采样是0.05s一个segment所以应该不会出现AB两句话中间检测不到
def check_pause(audio_buffer, pause_sec, rms_hold, bytes_sec):
    if len(audio_buffer) > int(pause_sec*bytes_sec):
        # todo 这里是直接按采样buffer的末尾往回查0.2s停顿，是不是该从采样buffer最后一个小于rms_hold的0.05s片段开始往回查0.2s停顿？
        rms = audioop.rms(audio_buffer[-int(pause_sec*bytes_sec):], 2)
        if rms <= rms_hold:
            return True
    # 例如检测0.1s停顿，但音频长度不够0.1s，默认返回False表示无停顿需要继续追加buffer
    # 长度够了但是最后0.1s
    return False

START_APPEND = False
def callback_v2(in_data, frame_count, time_info, status):
    global TIME_TAG, START_APPEND, BUFFER_CACHE, RMS_HOLD
    rms = audioop.rms(in_data, 2)
    if rms >= RMS_HOLD and not START_APPEND:
        TIME_TAG = int(time.time())
        logging.debug(f"Start Append (rms: {rms}>={RMS_HOLD} time: {TIME_TAG}).")
        START_APPEND = True

    if START_APPEND:
        BUFFER_CACHE += in_data
        # 检查buffer末尾，倒查是否有持续达0.2s的停顿，音频输入采样是0.05s一个segment所以应该不会出现AB两句话中间检测不到
        cond_pause = check_pause(BUFFER_CACHE, pause_sec=PAUSE_SEC, rms_hold=RMS_HOLD, bytes_sec=BYTES_PER_SEC)
        if cond_pause:
            logging.debug(f"Save And Send... TIME_TAG:{TIME_TAG} buffer_len({len(BUFFER_CACHE)}):{len(BUFFER_CACHE)/BYTES_PER_SEC:.2f}s")
            scipy.io.wavfile.write(f"req_audio_{TIME_TAG}.wav", SAMPLE_RATE, np.frombuffer(BUFFER_CACHE, dtype=np.int16))
            send_audio(BUFFER_CACHE)
            # 重置
            BUFFER_CACHE = b""
            START_APPEND = False
    else:
        BUFFER_CACHE = in_data
    return None, pyaudio.paContinue

# Initialize PyAudio
audio = pyaudio.PyAudio()
# todo. 这里通过控制frames_per_buffer实现0.25s执行一次callback，潜在的问题是如果语速太快，AB两句话中间间隔时间不足0.25s可能算出来rms音量就没有停顿了
# 更合理的做法还是callback里持续追加一个buffer，然后对那个buffer做空白时长判断吧？
stream = audio.open(format=pyaudio.paInt16,
                    channels=CHANNEL,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=int(BYTES_PER_SEC*SAMPLE_SEC),  # 控制在0.05s执行一次call_back 注意采样频率快了rms_hold得降低
                    stream_callback=callback_v2)

print("Recording...")

# Start the stream
stream.start_stream()

# Keep the script running
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(e)
    # Stop stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sys.exit(1)
finally:
    # Stop stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

