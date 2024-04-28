import pyaudio
import scipy
import numpy as np
import audioop

SAMPLE_RATE = 16000  # 采样频率
SAMPLE_WIDTH = 2  # 标准的16位PCM音频中，每个样本占用2个字节
CHANNELS = 1  # 音频通道数
CLEAR_GAP = 1  # 每隔多久没有收到新数据就认为要清空语音buffer
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

def play_audio_buffer(audio_buffer, sr, channels=1):
    p = pyaudio.PyAudio()
    # 打开一个音频流
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    output=True)
    # 播放音频
    stream.write(audio_buffer)
    # 结束后关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()

play_audio = play_audio_buffer

def save_audio_buffer(audio_buffer, sr, fp, dtype=np.float32):
    scipy.io.wavfile.write(fp, sr, np.frombuffer(audio_buffer, dtype=dtype))

def save_audio(audio, sr, fp):
    scipy.io.wavfile.write(fp, sr, audio)
    
# 计算音量，默认每0.5s一个计算gap
def cal_rms(inp_buffer, delta=0.5, sr=SAMPLE_RATE, sw=SAMPLE_WIDTH, c=CHANNELS):
    bps = sr*sw*c
    total_time = len(inp_buffer) / bps
    volume = []
    ts = []
    for i in range(0, int(total_time / delta)):
        s = int(i * delta * bps)
        e = int((i + 1) * delta * bps)
        y = audioop.rms(inp_buffer[s:e], sw)
        volume.append(y)
        ts.append(i*delta)
    return volume, ts


# wave写的wav文件dtype应该是用np.int16来解析
def play_audio_buffer_with_volume(audio_buffer, sr, channels=1, dtype=np.int16):
    p = pyaudio.PyAudio()

    # 打开一个音频流
    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=sr,
                    output=True)

    # 定义音频块大小
    BYTES_PER_SEC = sr * channels * 2
    CHUNK = int(1*BYTES_PER_SEC)

    # 转换音频缓冲区到 NumPy 数组
    audio_np_array = np.frombuffer(audio_buffer, dtype=dtype)

    # 播放音频
    for i in range(0, len(audio_np_array), CHUNK):
        chunk = audio_np_array[i:i+CHUNK]
        # 将 NumPy 数组转换为字节，写入音频流
        stream.write(chunk.astype(dtype).tobytes())

        # 计算并打印音量
        rms = audioop.rms(chunk.astype(dtype).tobytes(), 2)  # Here width = 2 because we're considering int16
        print("Volume:", rms)

    # 结束后关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()
