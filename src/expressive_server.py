# 启动的时候: conda deactivate && python expressive_server.py
# conda下有一个异常无法处理，OSError: libsndfile is not found! Since you are in a Conda environment, use `conda install -c conda-forge libsndfile==1.0.31` to install it
import time
import sys
import torch
import numpy as np
import logging
from logging.handlers import TimedRotatingFileHandler
import os

import librosa
import base64
import json
import utils_audio
from flask import Flask, request

from expressive_model import ExpressiveModel
import multiprocessing as mp
import pika
import pyloudnorm as pyln
import librosa

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
app = Flask(__name__, static_folder="/home/zhoutong", static_url_path="")


class Param:
    trace_id: str = ""
    buffer: bytes = None  # 音频按base64编码的字节流，会使用base64.b64decode(p.buffer)解码
    buffer_dtype: str = "int16"
    sample_rate: float = 16000
    speed: float = 1.0  # 合成音频的语速 (e.g. 1.2加速 0.8减速)
    lang: str = None  # 合成音频的语言 (e.g. zh_cn/en_us/fr_fr/es_es)
    result_queue_name: str = None  # 请求结果返回的queue
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


queue_service_seamless_request='queue_service_seamless_request'
# # POST | 如果是json
# @app.route("/test_json", methods=['POST'])
# def test_json():
#     logging.debug(f"Start connection.. start-time: {time.time()}")
#     p = Param(request.get_json())
#     logging.debug(f"Start Multi-Process Prediction of tid='{p.trace_id}' start-time: {time.time()}")
#     queue_in.put(p)
#     result = queue_out.get()
#     logging.debug(f"Finish Multi-Process Prediction of tid='{p.trace_id}' start-time: {time.time()}")
#     return result


def connect_to_rabbitmq():
    # RabbitMQ 连接信息
    rabbitmq_config = {
        "address": "120.24.144.127",
        "ports": [5672, 5673, 5674],
        "username": "admin",
        "password": "aibeeo",
        "virtual_host": "device-public"
    }

    # 连接到 RabbitMQ
    credentials = pika.PlainCredentials(rabbitmq_config["username"], rabbitmq_config["password"])
    parameters = pika.ConnectionParameters(
        host=rabbitmq_config["address"],
        port=rabbitmq_config["ports"][0],  # 默认使用第一个端口
        virtual_host=rabbitmq_config["virtual_host"],
        credentials=credentials
    )

    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        logger.info("Connected to RabbitMQ successfully.")
        # 全局消息属性
        global PROPERTIES
        PROPERTIES = pika.BasicProperties(content_type='application/json')  # 设置 content_type 为 JSON
        return connection, channel
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {repr(e)}")


def adjust_loudness(audio_arr, target_lufs=-23.0):
    """
    调整音频的响度（LUFS）到目标响度（LUFS）。
    :param audio_arr: 输入音频（[-1.0, 1.0] 范围的 numpy 数组）
    :param target_lufs: 目标 LUFS（默认为 -23 LUFS）
    :return: 调整后的音频
    """
    # 计算当前音频的 LUFS
    meter = pyln.Meter(16000)  # 采样率是 16kHz
    current_lufs = meter.integrated_loudness(audio_arr)
    logger.info(f"当前音频 LUFS: {current_lufs} LUFS")

    # 计算增益
    gain = target_lufs - current_lufs
    logger.info(f"需要应用的增益: {gain} dB")

    # 根据增益调整音频
    adjusted_audio = pyln.normalize.loudness(audio_arr, current_lufs, target_lufs)
    
    return adjusted_audio
        

def model_process():
    # 直接重新建立连接，因为tran耗时比较久，这时候断开了
    connection, channel = connect_to_rabbitmq()

    M = ExpressiveModel()
    while True:
        try:
            try:
                if channel is None or channel.is_closed:
                    connection, channel = connect_to_rabbitmq()
            except Exception:
                # 如果 抛出异常，直接重连
                logger.info(f"mq connect error,reconnect")
                connection, channel = connect_to_rabbitmq()
        
            method_frame, header_frame, body = channel.basic_get(queue=queue_service_seamless_request, auto_ack=True)
            if body is None:
                time.sleep(0.1)  # 如果没有消息，休眠一段时间
                continue  # 如果没有消息，等待下一次

            # 解析任务消息
            task = json.loads(body)
            p = Param(task)

            audio_arr = np.frombuffer(base64.b64decode(p.buffer), dtype=p.dtype)
            # utils_audio.save_audio(audio_arr, 16000, f"./tmp_{p.trace_id}.wav")
            # M.predict要求是双通道float32的音频，而输入是单通道int16
            if len(audio_arr.shape) == 1:
                # audio_arr = np.vstack([audio_arr, audio_arr])
                audio_arr = np.expand_dims(audio_arr, axis=0)
                
            # 转化到浮点，并归一化到 [-1.0, 1.0]
            audio_arr = audio_arr.astype(np.float32) / 32768.0  
            
            audio_arr = torch.from_numpy(audio_arr.T)
            logger.debug(f"    Start Predict of tid='{p.trace_id}'")
            wav_arr, wav_sr, text_cstr = M.predict(wav=audio_arr,
                                                   duration_factor=p.duration_factor,
                                                   tgt_lang=p.tgt_lang)
       
            logger.debug(f"    Finish Predict of tid='{p.trace_id}'")
            logger.debug(f"    Start resample&int16 of tid='{p.trace_id}'")
            wav_16khz = librosa.resample(wav_arr, orig_sr=wav_sr, target_sr=16000)
            # 增强20%
            wav_16khz = wav_16khz * 1.4
            wav_int16 = (np.clip(wav_16khz, -1.0, 1.0) * 32767).astype(np.int16)
            logger.debug(f"    Finish resample&int16 of tid='{p.trace_id}' end-time: {time.time()}")
            # rsp = {
            #     "trace_id": p.trace_id,
            #        # "audio_buffer": base64.b64encode(wav_arr.tobytes()).decode(),  # float32 & 24khz
            #        "audio_buffer_int16": base64.b64encode(wav_int16.tobytes()).decode(),  # int16 & 16khz
            #        "sample_rate": 16000,
            #        "audio_text": str(text_cstr),
            #        "status": "0",
            #        "msg": "success."}
            # 创建响应结构
            rsp = {
                "code": 0,  # 请求状态，通常为整数类型
                "msg": "success",  # 错误信息
                "result": {
                    "trace_id": p.trace_id,  # 将 trace_id 放入结果中
                    "audio_buffer_int16": base64.b64encode(wav_int16.tobytes()).decode(),   # base64 编码后的音频数据
                    "sample_rate": 16000,  # 采样率
                    "audio_text": str(text_cstr),  # 音频转换成的文本
                }
            }
            # 修正if语句
            if str(text_cstr) == 'Oh, my God.' or str(text_cstr) == "I'm sorry." or str(text_cstr) == "Come on, come on.":
                logger.info(f"error audio remove,tid='{p.trace_id}' audio_text='{rsp['result']['audio_text']}' result_queue_name='{p.result_queue_name}'")
                continue

               

            logger.info(f"success audio of tid='{p.trace_id}' audio_test='{rsp['result']['audio_text']}'  asr_text='{rsp['result']['asr_text']}' result_queue_name='{p.result_queue_name}' ")

            send_result_with_retry(channel, p.result_queue_name, rsp)
        
        except pika.exceptions.AMQPConnectionError:
            logger.error("Connection to RabbitMQ lost, attempting to reconnect...")
            connection, channel = connect_to_rabbitmq()
        except Exception as e:
            logger.error(f"Error in task: {e}", exc_info=True)
            result = {"code": 1, "msg": "Model Training failed.", "result": task.get('speaker', 'unknown')}
            # 发送结果到队列
            send_result_with_retry(channel, p.result_queue_name, result)


def send_result_with_retry(channel, queue, result):
    for attempt in range(5):
        try:
            channel.basic_publish(
                exchange='',
                routing_key=queue,
                body=json.dumps(result),
                properties=PROPERTIES  # Make message persistent
            )
            break  # 发送成功，退出循环
        except pika.exceptions.ChannelClosed:
            logger.error("Channel closed, reconnecting...")
            connection, channel = connect_to_rabbitmq()  # 重新连接
        except Exception as e:
            logger.error(f"Failed to send result: {e}")
            time.sleep(2)  # 等待后重试


if __name__ == '__main__':
    try:
        # 获取实例编号（从启动参数中获取）
        if len(sys.argv) > 1:
            instance_id = sys.argv[1]  # 启动时传入的实例编号
        else:
            instance_id = "default"  # 如果未传入参数，使用默认值

        # 配置日志
        handler_dir = "./logs"
        os.makedirs(handler_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(handler_dir, f"consumer-{instance_id}.log"),  # 根据实例编号生成日志文件名
            when="midnight",  # 按天分隔（午夜生成新日志文件）
            interval=1,  # 每 1 天分隔一次
            backupCount=7,  # 最多保留最近 7 天的日志文件
            encoding="utf-8"  # 设置编码，避免中文日志乱码
        )
        file_handler.suffix = "%Y-%m-%d"  # 设置日志文件后缀格式，例如 server.log.2025-01-09
        file_handler.setFormatter(logging.Formatter(
            fmt='[%(asctime)s-%(levelname)s]: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        # 将文件处理器添加到日志记录器中
        logger.addHandler(file_handler)

        model_process()
    except KeyboardInterrupt:
        logger.info("Consumer stopped.")
