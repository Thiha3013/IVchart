import sys
import subprocess

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QLineEdit,
    QLabel,
)

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IV Chart")
        
        vlayout = QVBoxLayout()

        titlelayout = QHBoxLayout()
        titlelayout.addWidget(QLabel("<h1>IV Chart</h1>"))
  
        hlayout1 = QHBoxLayout()  
        hlayout1.addWidget(QLabel("Ticker: "))
        self.textbox = QLineEdit()
        hlayout1.addWidget(self.textbox, 1)
        self.btn = QPushButton("confirm")
        hlayout1.addWidget(self.btn, 2)
    
        vlayout.addLayout(titlelayout)
        vlayout.addLayout(hlayout1)

        self.setLayout(vlayout)

        self.btn.clicked.connect(self.run_IV)


    def run_IV(self):
        text = self.textbox.text()
        try:
            subprocess.run(["python", "yf.py", text], check=True)
        except subprocess.CalledProcessError:
            print("Error running the script")
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())