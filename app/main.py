import sys
import os
import pyaudio
import wave
import time
import threading
import datetime


from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QComboBox, QDialog
from PyQt5.QtCore import Qt
from pydub import AudioSegment
from pydub.playback import play


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.FORMAT = pyaudio.paInt16
        self.DIRECTION = "records_micro"
        self.CHANNELS = 1
        self.RATE = 22050
        self.INPUT = True
        self.FRAMES = 1024


        self.file_path = ""
        self.input_device_index = 0
        self.audio_segment = None
        self.recording = False
        

        self.setWindowTitle("Voice Digit Scanner")
        self.setGeometry(300, 300, 800, 600)

        self.load_button = QPushButton("Загрузить файл", self)
        self.load_button.setGeometry(100, 500, 200, 30)
        self.load_button.clicked.connect(self.load_file)
        
        self.model_button = QPushButton("Выбрать модель", self)
        self.model_button.setGeometry(100, 540, 200, 30)
        self.model_button.clicked.connect(self.select_model)

        self.record_button = QPushButton("Старт запись", self)
        self.record_button.setGeometry(300, 500, 200, 30)
        self.record_button.clicked.connect(self.press_record)

        self.stop_button = QPushButton("Выбрать микрофон", self)
        self.stop_button.setGeometry(300, 540, 200, 30)
        self.stop_button.clicked.connect(self.select_device)

        self.play_button = QPushButton("Проиграть", self)
        self.play_button.setGeometry(500, 500, 200, 30)
        self.play_button.clicked.connect(self.play_audio)

        self.recognize_button = QPushButton("Распознать", self)
        self.recognize_button.setGeometry(500, 540, 200, 30)
        self.recognize_button.clicked.connect(self.recognize)

        self.status_label = QLabel(self)
        self.status_label.setGeometry(100, 450, 200, 30)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.time_label = QLabel("00:00:00", self)
        self.time_label.setGeometry(300, 450, 200, 30)
        self.time_label.setAlignment(Qt.AlignCenter)


    def load_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "Выберите WAV-файл")
        if self.file_path:
            self.audio_segment = AudioSegment.from_file(self.file_path)
            self.status_label.setText(os.path.basename(self.file_path))

    def select_model(self):
        # TODO: Add code to select model
        ...

    
    def select_device(self):
        class DeviceDialog(QDialog):
            def __init__(self, devices, parent=None):
                super(DeviceDialog, self).__init__(parent)
                self.devices = devices
                self.selected_device = None
                
                layout = QVBoxLayout()
                
                self.device_combo = QComboBox()
                self.device_combo.addItems(self.devices)
                layout.addWidget(self.device_combo)
                
                self.save_button = QPushButton('Сохранить')
                self.save_button.clicked.connect(self.save_device)
                layout.addWidget(self.save_button)
                
                self.setLayout(layout)

            def save_device(self):
                self.selected_device = self.device_combo.currentText()
                self.accept()

        pa = pyaudio.PyAudio()

        devices = [f"{pa.get_device_info_by_index(i)['name']}" for i in range(pa.get_device_count())]

        pa.terminate()

        dialog = DeviceDialog(devices, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_device = dialog.selected_device
            self.input_device_index = devices.index(selected_device)
        

    def press_record(self):
        if self.recording:
            self.recording = False
            self.record_button.setText("Старт запись")
        else:
            self.recording = True
            self.record_button.setText("Стоп запись")
            threading.Thread(target=self.record).start()

    def record(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            input_device_index=self.input_device_index,
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=self.INPUT,
            frames_per_buffer=self.FRAMES
        )
        
        frames = []
        start = time.time()

        while self.recording:
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

            passed_time = time.time() - start

            seconds = passed_time % 60
            mins = passed_time // 60
            hours = mins // 60

            self.time_label.setText(f"{int(hours):02d}:{int(mins):02d}:{int(seconds):02d}")

        stream.stop_stream()
        stream.close()
        pa.terminate()

        current_time = datetime.datetime.now().strftime('%H_%M_%S')
        sound_file = wave.open(f"{self.DIRECTION}/{current_time}.wav", 'wb')
        sound_file.setnchannels(self.CHANNELS)
        sound_file.setsampwidth(pa.get_sample_size(self.FORMAT))
        sound_file.setframerate(self.RATE)
        sound_file.writeframes(b''.join(frames))
        sound_file.close()

        self.file_path = f"{self.DIRECTION}/{current_time}.wav"
        self.audio_segment = AudioSegment.from_file(self.file_path)
        self.status_label.setText(os.path.basename(self.file_path))

    def play_audio(self):
        if self.audio_segment:
            play(self.audio_segment)

    def recognize(self):
        if self.audio_segment:
            # TODO: Add code to recognize audio
            ...


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()