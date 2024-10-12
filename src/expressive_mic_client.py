# ffmpeg -i a.mp4 -ar 16000 a.wav
import logging
logging.basicConfig(format='[%(asctime)s-%(levelname)s-CLIENT]: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
import sys
import wave
import pyaudio
import requests
import json
import base64
import audioop
import time
import numpy as np
import scipy

####### 尝试用耳机录音、电脑公放 注意这个修改是全局生效，不是只对这个程序生效######
import sounddevice as sd
print(sd.query_devices())
# 日志如下>标示为默认输入设备，<表示默认的输出设备
# > 0 zt的AirPods Pro - Find My, Core Audio (1 in, 0 out)
# < 1 zt的AirPods Pro - Find My, Core Audio (0 in, 2 out)
#   2 MacBook Pro Microphone, Core Audio (1 in, 0 out)
#   3 MacBook Pro Speakers, Core Audio (0 in, 2 out)
print(sd.default.device)  # 一个列表两个元素，依次表示使用的输入和输出
idx = [idx for idx, i in enumerate(sd.query_devices()) if 'MacBook Pro Speakers' in i['name']][0]
sd.default.device[1] = idx  # 使用mbp的speaker

# 默认都用int16的音频流
CHANNEL=1
SAMPLE_RATE=16000
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNEL
TIME_TAG = -1
BUFFER_CACHE = b""
# 停顿检测的时长、音量; 控制在0.05s执行一次call_back 注意采样频率快了rms_hold得降低
# 有些视频有背景音乐，现在还没有组合人声提取功能需要提高音量阈值(todo. 人声提取模块)
SAMPLE_SEC, PAUSE_SEC, RMS_HOLD = 0.05, 1.0, 700
START_APPEND = False


def send_audio(buffer, direct_play=False):
    url = "http://192.168.0.1:6006/test_json"
    url = "https://u212392-9161-bdb8f242.bjb1.seetacloud.com:8443/" + "test_json"
    url = "http://region-45.autodl.pro:40332/" + "test_json"
    data = {"buffer": base64.b64encode(buffer).decode(),
            "buffer_dtype": "int16",
            "sample_rate": SAMPLE_RATE,
            "speed": 0.8,
            "lang": "en_us"}  # cmn/eng/deu/fra/ita/spa
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url, data=json.dumps(data), headers=headers, timeout=10)
    if response.status_code != 200:
        logging.error(f"Request Error with code: {response.status_code}")
        with open("tmp.text", "w") as fw:
            fw.writelines(json.dumps(data))
        logging.error(f"send data saved in 'tmp.text'")
    else:
        rsp = json.loads(response.text)
        rsp_audio_arr = np.frombuffer(base64.b64decode(rsp['audio_buffer_int16']), dtype=np.int16)
        if direct_play:
            logging.info(f"同传完毕，直接播放，文本为: '{rsp['audio_text']}'")
            sd.play(rsp_audio_arr, samplerate=rsp['sample_rate'])
        global TIME_TAG
        fp = f"rsp_audio_{TIME_TAG}.wav"
        scipy.io.wavfile.write(fp, rsp['sample_rate'], rsp_audio_arr)
        # scipy.io.wavfile.write(f"rsp_{TIME_TAG}_24khz.wav", 24000, np.frombuffer(base64.b64decode(rsp['audio_buffer']), dtype=np.float32))
        # scipy.io.wavfile.write(f"rsp_{TIME_TAG}_16khz.wav", 16000, np.frombuffer(base64.b64decode(rsp['audio_buffer_int16']), dtype=np.int16))
        logging.info(f"同传完毕，音频写入文件'{fp}'，文本为: '{rsp['audio_text']}'")


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


def callback_v2(in_data, frame_count, time_info, status):
    global TIME_TAG, START_APPEND, BUFFER_CACHE, RMS_HOLD
    rms = audioop.rms(in_data, 2)
    if rms >= RMS_HOLD and not START_APPEND:
        TIME_TAG = int(time.time())
        logging.debug(f"Start Append (rms: {rms}>={RMS_HOLD} time: {TIME_TAG}).")
        START_APPEND = True
    else:
        if rms < RMS_HOLD:
            logging.debug(f"Low Volume (rms: {rms}<{RMS_HOLD} time: {TIME_TAG})")
        else:
            logging.debug(f"Keep appending...")

    if START_APPEND:
        BUFFER_CACHE += in_data
        # 检查buffer末尾，倒查是否有持续达0.2s的停顿，音频输入采样是0.05s一个segment所以应该不会出现AB两句话中间检测不到
        cond_pause = check_pause(BUFFER_CACHE, pause_sec=PAUSE_SEC, rms_hold=RMS_HOLD, bytes_sec=BYTES_PER_SEC)
        if cond_pause:
            logging.debug(f"Save And Send... TIME_TAG:{TIME_TAG} buffer_len({len(BUFFER_CACHE)}):{len(BUFFER_CACHE)/BYTES_PER_SEC:.2f}s")
            scipy.io.wavfile.write(f"req_audio_{TIME_TAG}.wav", SAMPLE_RATE, np.frombuffer(BUFFER_CACHE, dtype=np.int16))
            send_audio(BUFFER_CACHE, direct_play=True)
            # 重置
            BUFFER_CACHE = b""
            START_APPEND = False
    else:
        BUFFER_CACHE = in_data
    return None, pyaudio.paContinue


def audio_from_mic():
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


# 还没测通
def audio_from_file(fp):
    wf = wave.open(fp, 'rb')
    def _callback(in_data, frame_count, time_info, status):
        # logging.debug(f"[callback] {in_data} {frame_count} {time_info} {status}")
        data = wf.readframes(frame_count)
        callback_v2(data, frame_count, time_info, status)
        # 需要伪造一个数据return出去，不然检测到长度小于frame_count这个stream就自动中断了
        silent_data = np.zeros(frame_count * wf.getnchannels(), dtype=np.int16).tobytes()
        # logging.debug(f"[callback] read data size: {len(data)} silent data size: {len(silent_data)}")
        return silent_data, pyaudio.paContinue

    # Create an instance of PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=SAMPLE_RATE,
                        output=True,
                        frames_per_buffer=int(BYTES_PER_SEC * SAMPLE_SEC),
                        stream_callback=_callback)
    print("Playing audio from file...")
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Close PyAudio
    audio.terminate()
    print("Playback finished.")


if __name__ == '__main__':
    # audio_from_file("/Users/zhou/Downloads/a1544131919-1-16.wav")
    audio_from_mic()
