import ASR
from datetime import datetime

path = r'E:\PythonProject\timit\dr1-fvmh0\sa1.wav'
a = datetime.now()
result = ASR.asr_api(path, 'google')
b = datetime.now()
print(b - a)


