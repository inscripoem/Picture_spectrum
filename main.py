import cv2
import numpy as np
import soundfile as sf
import time

t1 = int(round(time.time() * 1000))                 #起始时间戳

img = ""    #图片路径
hz = 44100  #采样频率 hz
q = 10000   #起始频率 hz
s = 50      #频谱行距 ms
z = 10      #频谱列距 hz
            #结束频率 = 起始频率 + 图片宽度 * 频谱列距

def sin_wave(A, f, fs, phi, t):                     #生成正弦波
    Ts = 1 / fs
    n = t / Ts / 1000
    n = np.arange(n)
    y = A * np.sin(2 * np.pi * f * n * Ts + phi * (np.pi / 180))
    return y

def picture_spectrum(img ,hz ,q ,s ,z):             #生成频谱图
    img_cv = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    size = img_cv.shape
    hi = size[0]
    wi = size[1]
    
    # 预分配输出：初始零波 + hi 行
    duration_samples = int(s / (1/hz) / 1000)
    wave = np.zeros((hi + 1) * duration_samples)
    
    for ho in range(hi):
        data = np.zeros(duration_samples)  # 每行重置
        for wo in range(wi):
            a = img_cv[ho][wo] / 256
            f = wo * z + q
            data = data + sin_wave(a, f, hz, a, s)
        data = data / wi
        wave[(ho + 1) * duration_samples : (ho + 2) * duration_samples] = data
    
    # 反转以匹配原始堆叠顺序
    wave = wave[::-1].copy()
    return wave

wav = picture_spectrum(img ,hz ,q ,s ,z)
txt = str(s)+'ms_'+str(z)+'hz_'+str(t1)+'.wav'      #文件命名
sf.write(txt, wav, hz, subtype='PCM_16')            #保存wav文件

t2 = int(round(time.time() * 1000))                 #结束时间戳
t0 = '本次转换用时：' + str((t2 - t1)/1000) + 's'
print(t0)
