import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def get_melspectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    '''
    오디오 파일 경로를 입력받아 mel-spectrogram을 반환합니다.
    '''
    y, sr = librosa.load(audio_path, sr=sr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

    log_S = librosa.power_to_db(S, ref=np.max) #추출된 mel-spectrogram 데이터인 S를 dB 스케일로 변환하는 코드입니다. 
    #이 코드에서 사용된 power_to_db() 함수는 입력 데이터를 파워 스펙트럼에서 dB 스펙트럼으로 변환하는 함수입니다.
    return log_S

def plot_melspectrogram(mels, sr=22050, hop_length=512, y_axis='mel', x_axis='time'):
    '''
    mel-spectrogram을 시각화합니다.
    '''
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mels, x_axis=x_axis, y_axis=y_axis, sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()

def save_melspectrogram_image(mels, sr=22050, hop_length=512, y_axis='mel', x_axis='time', filename='melspectrogram.png'):
    '''
    mel-spectrogram 이미지를 저장합니다.
    '''
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mels, x_axis=x_axis, y_axis=y_axis, sr=sr, hop_length=hop_length)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
