import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert, savgol_filter
from scipy.stats import zscore
import glob
from tqdm import tqdm
import random
import pywt
import pandas as pd

class SeparatedADSBPreprocessor:
    """
    分离的ADS-B信号预处理器
    对瞬态和稳态信号采用不同的处理策略
    """

    def __init__(self, fs=1e6, snr_threshold=10.0):
        self.fs = fs
        self.snr_threshold = snr_threshold

    def estimate_snr(self, signal_data):
        """估计信号的信噪比"""
        analytic_signal = hilbert(signal_data)
        envelope = np.abs(analytic_signal)

        sorted_idx = np.argsort(envelope)
        noise_idx = sorted_idx[:int(0.2 * len(sorted_idx))]
        signal_idx = sorted_idx[int(0.2 * len(sorted_idx)):]

        noise_power = np.mean(np.power(signal_data[noise_idx], 2))
        signal_power = np.mean(np.power(signal_data[signal_idx], 2))

        if noise_power < 1e-10:
            noise_power = 1e-10

        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def detect_transient_steady_regions(self, signal_data, method='adaptive'):
        """
        检测瞬态和稳态区域

        参数:
        signal_data: 输入信号
        method: 检测方法 ('adaptive', 'energy_based', 'gradient_based')

        返回:
        transient_mask: 瞬态区域的布尔掩码
        steady_mask: 稳态区域的布尔掩码
        """
        if method == 'adaptive':
            return self._adaptive_detection(signal_data)
        elif method == 'energy_based':
            return self._energy_based_detection(signal_data)
        elif method == 'gradient_based':
            return self._gradient_based_detection(signal_data)
        else:
            raise ValueError(f"未知的检测方法: {method}")

    def _adaptive_detection(self, signal_data):
        """自适应检测方法，结合多种特征"""
        # 计算信号包络
        analytic_signal = hilbert(signal_data)
        envelope = np.abs(analytic_signal)

        # 平滑包络
        window_length = min(21, len(envelope)//10*2+1)
        if window_length >= 3:
            smooth_envelope = savgol_filter(envelope, window_length, 3)
        else:
            smooth_envelope = envelope

        # 计算多种特征
        # 1. 包络梯度
        envelope_gradient = np.gradient(smooth_envelope)

        # 2. 能量变化率
        energy = signal_data ** 2
        energy_gradient = np.gradient(energy)

        # 3. 瞬时频率变化
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        freq_gradient = np.gradient(instantaneous_phase)

        # 计算阈值
        grad_threshold = np.mean(np.abs(envelope_gradient)) + 1.5 * np.std(np.abs(envelope_gradient))
        energy_threshold = np.mean(np.abs(energy_gradient)) + 1.5 * np.std(np.abs(energy_gradient))
        freq_threshold = np.mean(np.abs(freq_gradient)) + 1.5 * np.std(np.abs(freq_gradient))

        # 综合判断瞬态区域
        transient_candidates = (
            (np.abs(envelope_gradient) > grad_threshold) |
            (np.abs(energy_gradient) > energy_threshold) |
            (np.abs(freq_gradient) > freq_threshold)
        )

        # 形态学操作连接相邻的瞬态点
        from scipy import ndimage
        transient_mask = ndimage.binary_dilation(transient_candidates, structure=np.ones(15))

        # 确保有足够的瞬态和稳态区域
        if np.sum(transient_mask) < 0.1 * len(signal_data):
            # 前30%作为瞬态
            transient_mask = np.zeros_like(signal_data, dtype=bool)
            transient_mask[:int(0.3 * len(signal_data))] = True
        elif np.sum(transient_mask) > 0.7 * len(signal_data):
            # 如果瞬态太多，只保留前50%
            transient_indices = np.where(transient_mask)[0]
            keep_indices = transient_indices[:int(0.5 * len(signal_data))]
            transient_mask = np.zeros_like(signal_data, dtype=bool)
            transient_mask[keep_indices] = True

        steady_mask = ~transient_mask
        return transient_mask, steady_mask

    def _energy_based_detection(self, signal_data):
        """基于能量的检测方法"""
        # 计算滑动窗口能量
        window_size = len(signal_data) // 20  # 5%的窗口大小
        energy = signal_data ** 2

        # 计算滑动平均能量
        moving_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')

        # 能量变化率
        energy_change = np.abs(np.gradient(moving_energy))

        # 阈值检测
        threshold = np.mean(energy_change) + 2 * np.std(energy_change)
        transient_mask = energy_change > threshold

        # 前30%强制为瞬态（ADS-B前导信号特性）
        forced_transient_length = int(0.3 * len(signal_data))
        transient_mask[:forced_transient_length] = True

        steady_mask = ~transient_mask
        return transient_mask, steady_mask

    def _gradient_based_detection(self, signal_data):
        """基于梯度的检测方法"""
        # 计算信号梯度
        signal_gradient = np.gradient(signal_data)

        # 平滑梯度
        window_length = min(15, len(signal_gradient)//10*2+1)
        if window_length >= 3:
            smooth_gradient = savgol_filter(signal_gradient, window_length, 2)
        else:
            smooth_gradient = signal_gradient

        # 梯度阈值
        grad_threshold = np.mean(np.abs(smooth_gradient)) + 2 * np.std(np.abs(smooth_gradient))

        # 检测高梯度区域
        transient_mask = np.abs(smooth_gradient) > grad_threshold

        # 形态学操作
        from scipy import ndimage
        transient_mask = ndimage.binary_dilation(transient_mask, structure=np.ones(10))

        # 确保前20%为瞬态
        forced_transient_length = int(0.2 * len(signal_data))
        transient_mask[:forced_transient_length] = True

        steady_mask = ~transient_mask
        return transient_mask, steady_mask

    def process_transient_signal(self, transient_signal):
        """
        专门处理瞬态信号
        重点关注动态特性和变化检测
        """
        processed_signal = transient_signal.copy()
        processing_info = {}

        # 1. 去除异常值（瞬态信号中的尖峰）
        z_scores = np.abs(zscore(processed_signal))
        outlier_mask = z_scores > 3.0
        if np.any(outlier_mask):
            # 用局部中值替换异常值
            for i in np.where(outlier_mask)[0]:
                start = max(0, i-2)
                end = min(len(processed_signal), i+3)
                local_data = processed_signal[start:end]
                processed_signal[i] = np.median(local_data[~outlier_mask[start:end]])

        # 2. 自适应滤波（保留高频变化）
        # 使用较高的截止频率保留瞬态特性
        spectrum = np.abs(np.fft.rfft(processed_signal))
        freqs = np.fft.rfftfreq(len(processed_signal), 1/self.fs)

        # 找到主要频率成分
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]

        # 设计滤波器（保留更多高频）
        nyq = 0.5 * self.fs
        low = max(0.01, (peak_freq * 0.1) / nyq)
        high = min(0.8, (peak_freq * 5) / nyq)  # 允许更高的频率

        if high > low + 0.01:
            b, a = butter(3, [low, high], btype='band')  # 低阶滤波器
            processed_signal = filtfilt(b, a, processed_signal)

        # 3. 保持信号的动态范围（轻度平滑）
        if len(processed_signal) > 5:
            window_length = min(7, len(processed_signal)//5*2+1)
            if window_length >= 3:
                processed_signal = savgol_filter(processed_signal, window_length, 2)

        # 4. 增强瞬态特征
        # 计算瞬态特征强度
        signal_energy = np.sum(processed_signal ** 2)
        signal_variation = np.var(processed_signal)
        max_gradient = np.max(np.abs(np.gradient(processed_signal)))

        processing_info.update({
            'transient_energy': signal_energy,
            'transient_variation': signal_variation,
            'max_gradient': max_gradient,
            'dynamic_range': np.max(processed_signal) - np.min(processed_signal)
        })

        # 5. 归一化（保持动态特性）
        if np.std(processed_signal) > 0:
            processed_signal = (processed_signal - np.mean(processed_signal)) / np.std(processed_signal)

        return processed_signal, processing_info

    def process_steady_signal(self, steady_signal):
        """
        专门处理稳态信号
        重点关注频域特性和稳定性分析
        """
        processed_signal = steady_signal.copy()
        processing_info = {}

        # 1. 强力去噪（稳态信号噪声容忍度低）
        # 小波去噪
        if len(processed_signal) > 16:
            wavelet = 'db4'
            level = min(pywt.dwt_max_level(len(processed_signal), wavelet), 4)
            coeffs = pywt.wavedec(processed_signal, wavelet, level=level)

            # 更强的阈值处理
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(processed_signal))) * 1.5  # 更强的阈值

            denoised_coeffs = []
            for i, coeff in enumerate(coeffs):
                if i > 0:  # 细节系数
                    denoised_coeff = pywt.threshold(coeff, threshold, mode='soft')
                    denoised_coeffs.append(denoised_coeff)
                else:  # 近似系数
                    denoised_coeffs.append(coeff)

            processed_signal = pywt.waverec(denoised_coeffs, wavelet)

            # 确保长度一致
            if len(processed_signal) > len(steady_signal):
                processed_signal = processed_signal[:len(steady_signal)]

        # 2. 频域滤波（重点保留低频和特征频率）
        spectrum = np.abs(np.fft.rfft(processed_signal))
        freqs = np.fft.rfftfreq(len(processed_signal), 1/self.fs)

        # 找到主要频率成分
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]

        # 设计更窄的带通滤波器
        nyq = 0.5 * self.fs
        bandwidth = peak_freq * 0.3  # 更窄的带宽
        low = max(0.01, (peak_freq - bandwidth) / nyq)
        high = min(0.6, (peak_freq + bandwidth) / nyq)

        if high > low + 0.01:
            b, a = butter(5, [low, high], btype='band')  # 高阶滤波器
            processed_signal = filtfilt(b, a, processed_signal)

        # 3. 强力平滑（稳态信号应该平滑）
        if len(processed_signal) > 9:
            window_length = min(15, len(processed_signal)//3*2+1)
            if window_length >= 3:
                processed_signal = savgol_filter(processed_signal, window_length, 3)

        # 4. 稳定性分析
        # 计算稳态特征
        signal_stability = 1.0 / (1.0 + np.var(processed_signal))  # 稳定性指标
        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * spectrum) / np.sum(spectrum)) if np.sum(spectrum) > 0 else 0

        processing_info.update({
            'steady_stability': signal_stability,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'mean_amplitude': np.mean(np.abs(processed_signal)),
            'rms_amplitude': np.sqrt(np.mean(processed_signal ** 2))
        })

        # 5. 归一化（保持稳定特性）
        if np.std(processed_signal) > 0:
            processed_signal = (processed_signal - np.mean(processed_signal)) / np.std(processed_signal)
        else:
            # 如果信号完全稳定，给予小的随机扰动以避免数值问题
            processed_signal = processed_signal + np.random.normal(0, 1e-6, len(processed_signal))

        return processed_signal, processing_info

    def process_complete_signal(self, signal_data, detection_method='adaptive'):
        """
        完整的信号处理流程
        分离瞬态和稳态后分别处理
        """
        # 1. 基础预处理
        clean_signal = self.basic_preprocessing(signal_data)

        # 2. 检测瞬态和稳态区域
        transient_mask, steady_mask = self.detect_transient_steady_regions(
            clean_signal, method=detection_method
        )

        # 3. 分离信号
        transient_signal = clean_signal[transient_mask]
        steady_signal = clean_signal[steady_mask]

        # 4. 分别处理
        processed_transient, transient_info = self.process_transient_signal(transient_signal)
        processed_steady, steady_info = self.process_steady_signal(steady_signal)

        # 5. 组合处理信息
        processing_info = {
            'original_length': len(signal_data),
            'transient_length': len(transient_signal),
            'steady_length': len(steady_signal),
            'transient_ratio': len(transient_signal) / len(signal_data),
            **transient_info,
            **steady_info
        }

        return processed_transient, processed_steady, transient_mask, steady_mask, processing_info

    def basic_preprocessing(self, signal_data):
        """基础预处理"""
        processed_signal = signal_data.copy()

        # 1. 去除直流分量
        processed_signal = processed_signal - np.mean(processed_signal)

        # 2. 基础去噪
        if len(processed_signal) > 1:
            processed_signal = np.convolve(processed_signal, [0.25, 0.5, 0.25], mode='same')

        return processed_signal

# 使用示例和集成到现有流程的类
class ADSBSignalProcessor:
    """
    ADS-B信号处理器，集成分离的预处理功能
    """

    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.preprocessor = SeparatedADSBPreprocessor()

        os.makedirs(output_path, exist_ok=True)

    def process_individual(self, individual_id, route):
        """
        处理单个个体的所有信号，使用分离的预处理方法
        """
        individual_path = os.path.join(self.data_path, f"{route}_data", individual_id)

        if not os.path.exists(individual_path):
            print(f"路径不存在: {individual_path}")
            return None

        # 创建输出目录结构
        output_individual_path = os.path.join(self.output_path, f"{route}_data", individual_id)
        transient_output_path = os.path.join(output_individual_path, "transient")
        steady_output_path = os.path.join(output_individual_path, "steady")

        os.makedirs(transient_output_path, exist_ok=True)
        os.makedirs(steady_output_path, exist_ok=True)

        # 获取所有信号文件
        signal_files = sorted(glob.glob(os.path.join(individual_path, "signal_*.txt")))

        if not signal_files:
            print(f"在 {individual_path} 中没有找到信号文件")
            return None

        all_processing_info = []

        # 处理每个信号文件
        for file_path in tqdm(signal_files, desc=f"处理 {individual_id} {route}路"):
            file_name = os.path.basename(file_path)
            signal_idx = int(file_name.split('_')[1].split('.')[0])

            # 读取信号数据
            signal_data = np.loadtxt(file_path)

            # 分离处理信号
            processed_transient, processed_steady, transient_mask, steady_mask, process_info = \
                self.preprocessor.process_complete_signal(signal_data)

            # 保存处理后的瞬态和稳态信号
            transient_file_path = os.path.join(transient_output_path, file_name)
            steady_file_path = os.path.join(steady_output_path, file_name)

            np.savetxt(transient_file_path, processed_transient, fmt='%.10f')
            np.savetxt(steady_file_path, processed_steady, fmt='%.10f')

            # 收集处理信息
            process_info.update({
                'individual_id': individual_id,
                'route': route,
                'signal_idx': signal_idx
            })

            all_processing_info.append(process_info)

        # 保存处理信息
        if all_processing_info:
            pd.DataFrame(all_processing_info).to_csv(
                os.path.join(output_individual_path, "separated_processing_info.csv"),
                index=False
            )

        return len(signal_files)

    def batch_process(self, args, routes=None):
        """批量处理多个个体"""
        if routes is None:
            routes = ['I', 'Q']

        for route in routes:
                route_path = os.path.join(self.data_path, f"{route}_data")

                if not os.path.exists(route_path):
                    print(f"路径不存在: {route_path}")
                    continue

                all_individuals = [d for d in os.listdir(route_path)
                                if os.path.isdir(os.path.join(route_path, d))]

                selected_individuals = all_individuals[:args.items]

                print(f"\n处理 {route}路，选择了 {len(selected_individuals)} 个个体")

                for individual_id in selected_individuals:
                    processed_count = self.process_individual(individual_id, route)
                    if processed_count:
                        print(f"完成 {individual_id}: {processed_count} 个信号")

def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description='分离的ADS-B信号预处理')
    parser.add_argument('--raw_data', type=str, required=True, help='原始数据目录')
    parser.add_argument('--processed_data', type=str, required=True, help='处理后的数据目录')
    parser.add_argument('--routes', type=str, nargs='+', choices=['I', 'Q'], help='要处理的路径列表')
    parser.add_argument('--items', type=int, help='要处理的个体數量')

    args = parser.parse_args()

    # 创建处理器
    processor = ADSBSignalProcessor(args.raw_data, args.processed_data)

    # 批量处理
    processor.batch_process(args,
        routes=args.routes
    )

    print(f"处理完成！结果保存在 {args.processed_data}")
    print("输出结构：")
    print("  - {route}_data/")
    print("    - {individual_id}/")
    print("      - transient/  # 瞬态信号")
    print("      - steady/     # 稳态信号")
    print("      - separated_processing_info.csv  # 处理信息")

if __name__ == "__main__":
    main()
