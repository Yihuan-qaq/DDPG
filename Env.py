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

        '''Init FFT param'''
        self.n_fft = 512
        self.win_length = self.n_fft
        self.hope_length = self.win_length // 4

        self.FLAG_EMPTY = []  # 记录没有出现的音素位置
        self.FLAG_VALUE = -100  # 标记音素没有出现位置的值
        self.bound_high = 0.2
        self.bound_low = -0.2
        self.STATE_HIGH_BOUND = 10
        self.STATE_LOW_BOUND = 0
        self.process_index_dict = self.find_phon()
        self.s_dim = len(self.process_index_dict)
        self.a_dim = self.s_dim

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
        EXPEND_FRAME = 2  # 扩展下帧数
        process_index_dict = dict([(p, []) for p in self.phon])
        '''加载并处理phn文件'''
        with open(self.source_path_phn) as f:
            phn_data = f.readlines()
            for i in range(0, len(phn_data)):
                phn_data[i] = phn_data[i].strip()
                phn_data[i] = phn_data[i].split()
        '''找到想要处理的phn对应的pcm数据下表'''
        for j in range(0, len(phn_data)):
            for key in process_index_dict.keys():
                if phn_data[j][2] == key:
                    '''将pcm下表转换为帧的下标'''
                    phn_data[j][0] = int(phn_data[j][0]) * 4 // self.win_length
                    phn_data[j][1] = int(phn_data[j][1]) * 4 // self.win_length + 1
                    """加入到字典中去"""
                    temp_list = [phn_data[j][0], phn_data[j][1]]
                    process_index_dict[key].append(temp_list)
        """寻找没有出现的音素的位置"""
        for key, i_ in zip(process_index_dict.keys(), range(len(process_index_dict))):
            index_lists = process_index_dict[key]
            if len(index_lists) == 0:
                self.FLAG_EMPTY.append(i_)
        return process_index_dict

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

    def calculate_reward(self, source_result, processed_result, source_path, phn_hat, threshold, s, a):
        """ 对那些超出阈值范围的状态进行大力度惩罚 """
        s = s[0]
        a = a[0]
        threshold_reward = np.zeros(shape=(len(s)), dtype=float)
        for i in range(len(s)):
            if threshold[i] == self.FLAG_VALUE:
                threshold_reward[i] = 0
            elif s[i] <= self.STATE_LOW_BOUND:
                if a[i] <= 0:
                    threshold_reward[i] = np.abs(threshold[i]) * 50
                else:
                    threshold_reward[i] = np.abs(threshold[i]) * 10
            elif s[i] >= self.STATE_HIGH_BOUND and a[i] >= 0:
                threshold_reward[i] = threshold[i] * 5
            else:
                threshold_reward[i] = threshold[i]

        """ 计算MSE分数"""
        if source_result == "RequestError":
            r = 0
            return r
        else:
            wer_value = wer(source_result, processed_result)

        global_ft_abs, source_pha, sr = self.process_audio(source_path)
        global_ft_abs_filter = self.low_filter(global_ft_abs, max(threshold_reward))
        global_ft = global_ft_abs_filter * source_pha
        global_ft_hat = librosa.istft(global_ft, n_fft=self.n_fft, hop_length=self.hope_length, win_length=self.win_length)

        source_hat, _ = soundfile.read(source_path)

        MSE1 = self.calculate_MSE(source_hat, phn_hat)
        MSE2 = self.calculate_MSE(source_hat, global_ft_hat)

        if MSE2 == 0.0:
            return -100
        else:
            MSE_ratio = MSE1 / MSE2

        """计算总的reward"""
        # total_threshold =
        mean_threshold = np.sum(threshold_reward) / (len(threshold_reward) - len(self.FLAG_EMPTY))
        r = wer_value * 100 - MSE_ratio * 70 - mean_threshold * 60
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
        """限制阈值范围"""
        threshold[threshold > self.STATE_HIGH_BOUND] = self.STATE_HIGH_BOUND
        threshold[threshold < self.STATE_LOW_BOUND] = self.STATE_LOW_BOUND
        for i_ in range(len(self.FLAG_EMPTY)):
            threshold[self.FLAG_EMPTY[i_]] = self.FLAG_VALUE
        """处理音频"""
        ft_abs, pha, sr = self.process_audio(self.source_path_wav)
        process_index_dict = self.process_index_dict
        '''滤波'''
        for key, i in zip(process_index_dict.keys(), range(len(process_index_dict))):
            index_lists = process_index_dict[key]
            if len(index_lists) == 0:
                continue
            else:
                for index in index_lists:
                    start_index = index[0]
                    end_index = index[1]
                    ft_abs[:, start_index - 1:end_index - 1] = \
                        self.low_filter(ft_abs[:, start_index - 1:end_index - 1], threshold[i])
        '''重建滤波后的音频'''
        ft = ft_abs * pha
        y_hat = librosa.istft(ft, n_fft=self.n_fft, hop_length=self.hope_length, win_length=self.win_length)
        temp_wirte_path = r'temp.wav'
        soundfile.write(temp_wirte_path, y_hat, samplerate=sr)
        t0 = time.time()
        trans_result = ASR.asr_api(temp_wirte_path, 'google')
        t1 = time.time()
        r = self.calculate_reward(self.source_result, trans_result, self.source_path_wav, phn_hat=y_hat,
                                  threshold=threshold, s=s, a=a)
        return s_, r, done, t1 - t0

    def reset(self):
        """
        初始化状态
        :return: 状态s
        """
        Max = 0.5  # 随机生成小数的最大值
        s = np.random.rand(self.s_dim) * Max
        return np.array([s], dtype=float)

    def action_space_high(self):
        return self.bound_high

    def action_space_low(self):
        return self.bound_low

    def get_s_dim(self):
        return self.s_dim

    def get_a_dim(self):
        return self.a_dim

    def get_FLAG_EMPTY(self):
        return self.FLAG_EMPTY
