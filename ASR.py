import speech_recognition as sr
from scipy.io import wavfile
# from deepspeech import Model
import deepspeech
import asr_xunfei
import json


def asr_api(path, api):
    r = sr.Recognizer()
    test = sr.AudioFile(path)
    with test as source:
        audio = r.record(source)
    if api == 'google':
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "Transcirbe error"
        except sr.RequestError as err:
            print("Google error; {0}".format(err))
            return "RequestError"

    if api == 'deepspeech':
        fs, data = wavfile.read(path)
        model_path = r'D:\output_graph.pb'  # 已下载的模型地址（正确的模型文件中有以.pb结尾的文件）
        ars = deepspeech.Model(model_path, 1024)
        translate_txt = ars.stt(data)
        return translate_txt

    if api == 'xunfei':
        appid = "71987808"
        secret_key = "eb294d155c59115352772cfde7755617"
        asr = asr_xunfei.RequestApi(appid=appid, secret_key=secret_key, upload_file_path=path)
        try:
            result = asr.all_api_request()
            result = result["data"]
            result = json.loads(result)
            result = result[0]
            result = result['onebest']
            result.replace("'", "")
            return result

        except json.decoder.JSONDecodeError:
            print("file transcribe error \n")
            result = path
            return result