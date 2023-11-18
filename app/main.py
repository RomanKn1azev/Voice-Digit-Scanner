import sys


from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from pydub import AudioSegment
from pydub.playback import play


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Recognition")
        self.setGeometry(100, 100, 400, 300)

        self.file_path = ""
        self.audio_segment = None

        self.load_button = QPushButton("Load File", self)
        self.load_button.setGeometry(10, 10, 100, 30)
        self.load_button.clicked.connect(self.load_file)
        
        self.model_button = QPushButton("Select Model", self)
        self.model_button.setGeometry(10, 50, 100, 30)
        self.model_button.clicked.connect(self.select_model)

        self.record_button = QPushButton("Record", self)
        self.record_button.setGeometry(10, 90, 100, 30)
        self.record_button.clicked.connect(self.record)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setGeometry(10, 130, 100, 30)
        self.stop_button.clicked.connect(self.stop)

        self.play_button = QPushButton("Play", self)
        self.play_button.setGeometry(10, 170, 100, 30)
        self.play_button.clicked.connect(self.play_audio)

        self.recognize_button = QPushButton("Recognize", self)
        self.recognize_button.setGeometry(10, 210, 100, 30)
        self.recognize_button.clicked.connect(self.recognize)

        self.status_label = QLabel(self)
        self.status_label.setGeometry(120, 10, 200, 30)
        self.status_label.setAlignment(Qt.AlignCenter)

    def load_file(self):
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, "Select WAV File")
        if self.file_path:
            self.audio_segment = AudioSegment.from_file(self.file_path)
            self.status_label.setText("File Loaded")

    def select_model(self):
        # TODO: Add code to select model
        ...

    def record(self):
        # TODO: Add code to record audio from microphone
        ...

    def stop(self):
        # TODO: Add code to stop recording and load audio
        ...

    def play_audio(self):
        if self.audio_segment:
            play(self.audio_segment)

    def recognize(self):
        if self.audio_segment:
            # TODO: Add code to recognize audio
            ...

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())