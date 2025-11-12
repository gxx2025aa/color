# ui/main_window.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QGroupBox, QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import torch
from model.manns_stylegan import MANNsStyleGAN
from utils.image_processing import ImagePreprocessor

class AdvertisingDesignSystem(QMainWindow):
    """Main interface of advertising design color matching system"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image = None
        self.processed_image = None
        self.preprocessor = ImagePreprocessor()
        
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Advertising design color matching and pattern coloring system")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central component
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left image display area
        left_panel = self.create_image_panel()
        main_layout.addWidget(left_panel, 3)
        
        # Right control panel
        right_panel = self.create_control_panel()
        main_layout.addWidget(right_panel, 1)
        
        central_widget.setLayout(main_layout)
    
    def create_image_panel(self):
        """Create Image Display Panel"""
        panel = QGroupBox("image display")
        layout = QVBoxLayout()
        
        # Original image display
        self.original_label = QLabel("original image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(400, 300)
        self.original_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(QLabel("original image:"))
        layout.addWidget(self.original_label)
        
        # Processed image display
        self.processed_label = QLabel("Processed image")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(400, 300)
        self.processed_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(QLabel("Coloring Results:"))
        layout.addWidget(self.processed_label)
        
        panel.setLayout(layout)
        return panel
    
    def create_control_panel(self):
        """Create Control Panel"""
        panel = QGroupBox("control panel")
        layout = QVBoxLayout()
        
        # File operation group
        file_group = QGroupBox("File operation")
        file_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_image)
        file_layout.addWidget(self.save_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Image processing group
        process_group = QGroupBox("image processing")
        process_layout = QVBoxLayout()
        
        self.color_btn = QPushButton("Intelligent shading")
        self.color_btn.clicked.connect(self.colorize_image)
        process_layout.addWidget(self.color_btn)
        
        # Brightness adjustment
        process_layout.addWidget(QLabel("Brightness adjustment:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        process_layout.addWidget(self.brightness_slider)
        
        # Contrast adjustment
        process_layout.addWidget(QLabel("Contrast adjustment:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-50, 50)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        process_layout.addWidget(self.contrast_slider)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # Advanced function group
        advanced_group = QGroupBox("Advanced Features")
        advanced_layout = QVBoxLayout()
        
        self.enhance_btn = QPushButton("image enhancement")
        self.enhance_btn.clicked.connect(self.enhance_image)
        advanced_layout.addWidget(self.enhance_btn)
        
        self.filter_btn = QPushButton("Application filter")
        self.filter_btn.clicked.connect(self.apply_filter)
        advanced_layout.addWidget(self.filter_btn)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        panel.setLayout(layout)
        return panel
    
    def load_model(self):
        """Load pre training model"""
        try:
            self.model = MANNsStyleGAN()
            # Pre training weights should be loaded here
            # self.model.load_state_dict(torch.load('weights/manns_stylegan.pth'))
            self.model.eval()
            QMessageBox.information(self, "success", "Model loaded successfully！")
        except Exception as e:
            QMessageBox.warning(self, "error", f"Model loading failed: {str(e)}")
    
    def load_image(self):
        """Load image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "", "image file (*.png *.jpg *.jpeg)"
        )
        
        if file_path:
            self.current_image = Image.open(file_path)
            pixmap = QPixmap(file_path)
            self.original_label.setPixmap(
                pixmap.scaled(self.original_label.width(), 
                            self.original_label.height(),
                            Qt.KeepAspectRatio)
            )
    
    def colorize_image(self):
        """Intelligent shading of images"""
        if self.current_image is None:
            QMessageBox.warning(self, "warning", "Please load the image first！")
            return
        
        if self.model is None:
            QMessageBox.warning(self, "warning", "Model not loaded！")
            return
        
        try:
            # Image preprocessing
            image_tensor = self.preprocessor.transform(self.current_image).unsqueeze(0)
            
            # Generate grayscale image
            grayscale_tensor = torch.mean(image_tensor, dim=1, keepdim=True).repeat(1, 3, 1, 1)
            
            # Model reasoning
            with torch.no_grad():
                colored_image, _, _ = self.model(grayscale_tensor)
            
            # Post processing
            colored_image = (colored_image.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            
            # Convert to QPixmap display
            colored_np = (colored_image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            height, width, channel = colored_np.shape
            bytes_per_line = 3 * width
            
            q_image = QImage(colored_np.data, width, height, bytes_per_line, 
                           QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.processed_label.setPixmap(
                pixmap.scaled(self.processed_label.width(),
                            self.processed_label.height(),
                            Qt.KeepAspectRatio)
            )
            
            self.processed_image = colored_image
            
        except Exception as e:
            QMessageBox.warning(self, "error", f"Shading failed: {str(e)}")
    
    def adjust_brightness(self, value):
        """adjust brightness"""
        if self.processed_image is not None:
            brightness = value / 100.0
            adjusted = self.preprocessor.adjust_brightness_contrast(
                self.processed_image, brightness=brightness
            )
            self.update_processed_display(adjusted)
    
    def adjust_contrast(self, value):
        """Adjust contrast"""
        if self.processed_image is not None:
            contrast = value / 100.0
            adjusted = self.preprocessor.adjust_brightness_contrast(
                self.processed_image, contrast=contrast
            )
            self.update_processed_display(adjusted)
    
    def update_processed_display(self, image_tensor):
        """Update the image display after processing"""
        image_np = (image_tensor.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        self.processed_label.setPixmap(
            pixmap.scaled(self.processed_label.width(),
                        self.processed_label.height(),
                        Qt.KeepAspectRatio)
        )
    
    def save_image(self):
        """Save the processed image"""
        if self.processed_image is None:
            QMessageBox.warning(self, "warning", "No images to save！")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG image (*.png);;JPEGimage (*.jpg)"
        )
        
        if file_path:
            try:
                image_to_save = (self.processed_image.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(image_to_save).save(file_path)
                QMessageBox.information(self, "success", "Image saved successfully！")
            except Exception as e:
                QMessageBox.warning(self, "error", f"Save failed: {str(e)}")
    
    def enhance_image(self):
        """Image enhancement"""
        QMessageBox.information(self, "prompt", "Image enhancement function under development...")
    
    def apply_filter(self):
        """Application filter function"""
        QMessageBox.information(self, "prompt", "The filter function is under development...")

def main():
    app = QApplication(sys.argv)
    window = AdvertisingDesignSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()