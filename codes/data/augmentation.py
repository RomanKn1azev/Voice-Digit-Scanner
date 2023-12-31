import librosa
import numpy as np


def change_volume(audio, level):
    """
    Функция позволяет изменить громкость аудио путем увеличения или уменьшения
    значения аудио сигнала на определенный уровень.

    Аргументы 
    - audio - это аудио сигнал, представленный числовыми значениями
    - level - уровень изменения громкости.

    Функция умножает значение audio на level и возвращает результат.
    """
    return audio * float(level)

def change_tonalities(audio, sr, n_steps):
    """
    Функция изменяет тон звука аудиофайла на указанное количество полутонов.

    Аргументы
        - audio:  Аудиоданные в формате временного сигнала.
        - sr: Частота дискретизации аудио в герцах.
        - n_steps: Количество полутонов, на которое нужно изменить тон звука аудиофайла.

    Возвращаемое значение
        - Аудиоданные с измененной тональностью.
    """
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=float(n_steps))

def change_time_stretch(audio, rate):
    return librosa.effects.time_stretch(audio, rate=float(rate))

def pink_noise(length, alpha=1):
    return np.random.randn(length) / np.sqrt(np.arange(1, length + 1)) ** alpha

def brown_noise(length, alpha=0.5):
    return np.random.randn(length) / np.arange(1, length + 1) ** alpha

def add_noise(audio, type, level):
    if type == 'white':
        noise = np.random.randn(len(audio))
    elif type == 'pink':
        noise = pink_noise(len(audio))
    elif type == 'brown':
        noise = brown_noise(len(audio))
    else:
        raise ValueError(f'Invalid noise type {type}')
    
    augmented_audio = audio + float(level) * noise
    
    return augmented_audio
