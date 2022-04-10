import os
import soundfile
import librosa
import numpy as np
import ASR


def low_filter(ft_matrix, threshold):
    ft_filter = np.zeros(shape=(len(ft_matrix), len(ft_matrix[0])), dtype=float)
    for i in range(len(ft_matrix)):
        for j in range(len(ft_matrix[0])):
            if ft_matrix[i][j] < threshold:
                ft_filter[i][j] = 0
            else:
                ft_filter[i][j] = ft_matrix[i][j]
    # ft_matrix[ft_matrix < threshold] = 0
    return ft_filter
#
#
# def normalize(data):
#     normalized = data.ravel() * 1.0 / np.amax(np.abs(data.ravel()))
#     magnitude = np.abs(normalized)
#     return magnitude
#
#
# def calculate_MSE(audio1, audio2):
#     # Normalize
#     # n_audio1 = normalize(audio1)
#     # n_audio2 = normalize(audio2)
#
#     audio_len = min(len(audio1), len(audio2))
#     n_audio1 = audio1[:audio_len]
#     n_audio2 = audio2[:audio_len]
#
#     # Diff
#     diff = n_audio1 - n_audio2
#     abs_diff = np.abs(diff)
#     overall_change = sum(abs_diff)
#     average_change = overall_change / len(audio1)
#     return average_change
#
#


def find_phon(source_path_phn, phon, hope_length, win_length):
    """
    得到需要攻击的音素的边界
    STFT中帧的index与Pcm数据的index关系：
    第n帧的pcm数据点范围 = [win_length * (n - 1) - hop_length, win_length * n - hop_length]
    :return: (pcm数据点)stft转换后帧的起始和截止的边界[strat,end]
    """
    process_index = []
    '''加载并处理phn文件'''
    with open(source_path_phn) as f:
        phn_data = f.readlines()
        for i in range(0, len(phn_data)):
            phn_data[i] = phn_data[i].strip()
            phn_data[i] = phn_data[i].split()
    '''找到想要处理的phn对应的pcm数据下表'''
    for j in range(0, len(phn_data)):
        for i in range(0, len(phon)):
            if phn_data[j][2] == phon[i]:
                temp_list = [int(phn_data[j][0]), int(phn_data[j][1])]
                process_index.append(temp_list)
    '''将pcm下表转换为帧的下标'''
    for i_ in range(0, len(process_index)):
        process_index[i_][0] = (process_index[i_][0] + hope_length) // win_length
        process_index[i_][1] = (process_index[i_][1] + hope_length) // win_length + 1
    return process_index


def process_audio(source_path, n_fft):
    x, sr = soundfile.read(source_path)
    ft = librosa.stft(x, n_fft=n_fft)
    pha = np.exp(1j * np.angle(ft))
    ft_abs = np.abs(ft)
    return x, ft_abs, pha, sr
#


SOURCE_PATH_PHN = r'E:\PythonProject\timit\dr1-fvmh0\sa1.phn'
SOURCE_PATH_WAV = r'E:\PythonProject\timit\dr1-fvmh0\sa1.wav'
PHN = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw',
       'ux', 'er', 'ax', 'ix', 'arx', 'ax-h']  # 20个元音音素
n_fft = 512
threshold = 2.7

x, ft_abs, pha, sr = process_audio(SOURCE_PATH_WAV, n_fft=512)
process_index = find_phon(SOURCE_PATH_PHN, PHN, hope_length=n_fft // 4, win_length=n_fft)
for i in range(0, len(process_index)):
    strat_index = process_index[i][0]
    end_index = process_index[i][1]
    ft_abs[:, strat_index - 1:end_index - 1] = \
        low_filter(ft_abs[:, strat_index - 1:end_index - 1], threshold)
'''重建滤波后的音频'''
ft = ft_abs * pha
y_hat = librosa.istft(ft, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft)
temp_wirte_path = r'temp.wav'
soundfile.write(temp_wirte_path, y_hat, samplerate=sr)
trans_result = ASR.asr_api(temp_wirte_path, 'google')
print(trans_result)
# ft_abs_filter = low_filter(ft_abs, threshold=0.5)
# ft_filter = ft_abs_filter * pha
# y_hat = librosa.istft(ft_filter)
# print(calculate_MSE(x, y_hat))
