import numpy as np
import soundfile
import librosa
import numpy as np
import ASR
from jiwer import wer
import time


class Env(object):

    def __init__(self, phon, source_path_wav, source_path_phn):
        self.phon = phon
        self.source_path_wav = source_path_wav
        self.source_path_phn = source_path_phn

        flag = 0
        if flag == 0:
            self.source_result = ASR.asr_api(self.source_path_wav, 'google')
            self.temp_source_result = self.source_result
            flag = 1
            print("----source result :{}".format(self.source_result))
        else:
            self.source_result = self.temp_source_result
        # self.done = False
        # self.r = 0
        self.bound_high = 1
        self.bound_low = -1
        self.s_dim = 1
        self.a_dim = 1
        '''Init FFT param'''
        self.n_fft = 512
        self.win_length = self.n_fft
        self.hope_length = self.win_length // 4

    def process_audio(self, source_path):
        x, sr = soundfile.read(source_path)
        ft = librosa.stft(x, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hope_length)
        pha = np.exp(1j * np.angle(ft))
        ft_abs = np.abs(ft)
        return ft_abs, pha, sr

    def find_phon(self):
        """
        得到需要攻击的音素的边界
        STFT中帧的index与Pcm数据的index关系：
        第n帧的pcm数据点范围 = [win_length * (n - 1) - hop_length, win_length * n - hop_length]
        :return: (pcm数据点)stft转换后帧的起始和截止的边界[strat,end]
        """
        process_index = []
        '''加载并处理phn文件'''
        with open(self.source_path_phn) as f:
            phn_data = f.readlines()
            for i in range(0, len(phn_data)):
                phn_data[i] = phn_data[i].strip()
                phn_data[i] = phn_data[i].split()
        '''找到想要处理的phn对应的pcm数据下表'''
        for j in range(0, len(phn_data)):
            for i in range(0, len(self.phon)):
                if phn_data[j][2] == self.phon[i]:
                    temp_list = [int(phn_data[j][0]), int(phn_data[j][1])]
                    process_index.append(temp_list)
        '''将pcm下表转换为帧的下标'''
        for i_ in range(0, len(process_index)):
            process_index[i_][0] = (process_index[i_][0] + self.hope_length) // self.win_length
            process_index[i_][1] = (process_index[i_][1] + self.hope_length) // self.win_length + 1
        return process_index

    def low_filter(self, ft_matrix, threshold):
        ft_filter = np.zeros(shape=(len(ft_matrix), len(ft_matrix[0])), dtype=float)
        for i in range(len(ft_matrix)):
            for j in range(len(ft_matrix[0])):
                if ft_matrix[i][j] < threshold:
                    ft_filter[i][j] = 0
                else:
                    ft_filter[i][j] = ft_matrix[i][j]
        # ft_matrix[ft_matrix < threshold] = 0
        return ft_filter

    # def normalize(self, data):
    #     normalized = data.ravel() * 1.0 / np.amax(np.abs(data.ravel()))
    #     magnitude = np.abs(normalized)
    #     return magnitude

    def calculate_MSE(self, audio1, audio2):
        # Normalize
        # n_audio1 = self.normalize(audio1)
        # n_audio2 = self.normalize(audio2)

        audio_len = min(len(audio1), len(audio2))
        n_audio1 = audio1[:audio_len]
        n_audio2 = audio2[:audio_len]

        # Diff
        diff = n_audio1 - n_audio2
        abs_diff = np.abs(diff)
        overall_change = sum(abs_diff)
        average_change = overall_change / len(audio1)
        return average_change

    def calculate_reward(self, source_result, processed_result, source_path, phn_hat, threshold):
        if source_result == "RequestError":
            r = 0
            return r
        else:
            wer_value = wer(source_result, processed_result)

        global_ft_abs, source_pha, sr = self.process_audio(source_path)
        global_ft_abs_filter = self.low_filter(global_ft_abs, threshold)
        global_ft = global_ft_abs_filter * source_pha
        global_ft_hat = librosa.istft(global_ft, n_fft=self.n_fft, hop_length=self.hope_length, win_length=self.win_length)

        source_hat, _ = soundfile.read(source_path)

        MSE1 = self.calculate_MSE(source_hat, phn_hat)
        MSE2 = self.calculate_MSE(source_hat, global_ft_hat)

        if MSE2 == 0.0:
            return 0
        else:
            MSE_ratio = MSE1 / MSE2
            r = (wer_value - MSE_ratio) * 100 - abs(threshold) * 7  # 5太小10太大
            return r

    def step(self, s, a):
        """
        :input: 动作a
        计算当前状态s加上动作a后的下一状态s_;
        用这个s_进行一次滤波，并转录结果，判断是否攻击成功;
        攻击成功：奖励r=1，结束标志done=True；
        攻击成功：奖励r=0，结束标志done=Flase；
        :return: s_,r,done;
        """
        done = False
        r = 0
        s_ = s + a
        threshold = s_[0]
        ft_abs, pha, sr = self.process_audio(self.source_path_wav)
        process_index = self.find_phon()
        '''滤波'''
        for i in range(0, len(process_index)):
            strat_index = process_index[i][0]
            end_index = process_index[i][1]
            ft_abs[:, strat_index - 1:end_index - 1] = \
                self.low_filter(ft_abs[:, strat_index - 1:end_index - 1], threshold)
        '''重建滤波后的音频'''
        ft = ft_abs * pha
        y_hat = librosa.istft(ft, n_fft=self.n_fft, hop_length=self.hope_length, win_length=self.win_length)
        temp_wirte_path = r'temp.wav'
        soundfile.write(temp_wirte_path, y_hat, samplerate=sr)
        t0 = time.time()
        trans_result = ASR.asr_api(temp_wirte_path, 'google')
        t1 = time.time()
        r = self.calculate_reward(self.source_result, trans_result, self.source_path_wav, phn_hat=y_hat, threshold=threshold)
        # wer_result = wer(trans_result, self.source_result)
        # if trans_result != self.source_result:
        #     done = True
        #     r = 1
        # if 0 < wer_result <= 0.1:
        #     r = 0.3
        # if 0.1 < wer_result <= 0.2:
        #     r = 0.6
        # if 0.2 < wer_result <= 0.3:
        #     print(wer_result)
        #     r = 1
        #     done = True
        return s_, r, done, t1 - t0

    def reset(self):
        """
        初始化状态
        :return: 状态s
        """
        s = np.random.uniform(0, 5)
        return np.array([s], dtype=float)

    def action_space_high(self):
        return self.bound_high

    def action_space_low(self):
        return self.bound_low

    def get_s_dim(self):
        return self.s_dim

    def get_a_dim(self):
        return self.a_dim
