# MTCNN Face Authentication - Attendance System

A real-time face recognition-based attendance system that automatically marks student attendance using MTCNN face detection and deep learning facial encoding.

**Made by: Arpit Sharma**

---

## 📋 Project Overview

MTCNN Face Authentication is an AI-powered attendance management system that uses facial recognition technology to automate the attendance marking process. The system detects student faces through a webcam or IP camera, matches them against registered students, and automatically records attendance without manual intervention.

### Key Features

- **Real-time Face Detection**: Uses MTCNN (Multi-task Cascaded Convolutional Networks) for robust face detection
- **Facial Recognition**: Utilizes InceptionResnetV1 deep learning model for accurate face encoding and matching
- **Web-Based Interface**: Built with Django and HTML5 for easy browser access
- **Automatic Attendance Marking**: Recognizes students and marks attendance in real-time
- **Student Management**: Register, authorize, and manage student records
- **Camera Configuration**: Support for multiple cameras including IP cameras
- **Attendance Tracking**: View and filter attendance records by date and student
- **Admin Dashboard**: Secure admin panel for system management

---

## 🛠️ Technology Stack

### Backend
- **Django**: Web framework for Python
- **Python 3.x**: Programming language

### Face Recognition & Detection
- **MTCNN** (Multi-task Cascaded Convolutional Networks): Face detection
- **InceptionResnetV1**: Deep learning model for facial encoding
- **PyTorch**: Deep learning framework
- **FaceNet PyTorch**: Pre-trained facial recognition model

### Computer Vision
- **OpenCV**: Video capture and image processing
- **NumPy**: Numerical computing

### Frontend
- **HTML5**: Markup for web pages
- **CSS3**: Styling and responsive design
- **JavaScript**: Browser-based camera access and frame processing
- **Bootstrap**: UI framework

### Database
- **SQLite**: Lightweight database for development
- **Django ORM**: Object-relational mapping

### Audio
- **Pygame**: Sound playback for attendance confirmation

---

## 🚀 How It Works

### 1. **Student Registration**
- Students register by entering their details (name, email, phone, class)
- A photo is captured through the webcam and stored in the system
- The system generates facial encodings for recognition

### 2. **Face Authorization**
- Admin authorizes registered students before they can be recognized
- Only authorized faces are used for attendance matching

### 3. **Real-Time Recognition**
- When "Mark Attendance" is clicked, the camera starts streaming
- Each frame is sent to the server for processing every 500ms
- The system compares detected faces against authorized student faces
- MTCNN detects faces and InceptionResnetV1 creates facial encodings
- Euclidean distance is calculated to find the best match

### 4. **Attendance Recording**
- When a recognized student's face is detected:
  - A check-in time is recorded (first recognition of the day)
  - After 60 seconds, they can check out
  - A success sound plays to confirm attendance
  - Camera automatically stops after marking attendance

### 5. **View Attendance**
- Admins can view attendance records filtered by date and student name
- Attendance displays check-in and check-out times

### 6. **Camera Configuration**
- Support for local webcams (index 0, 1, etc.)
- Support for IP cameras (RTSP URLs)
- Configurable face matching threshold

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Real-Time-Face-Based-Attendance-System-with-Django-main
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**
   ```bash
   python manage.py migrate
   ```

5. **Create superuser (admin account)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Open your browser and go to `http://127.0.0.1:8000/`
   - Admin panel: `http://127.0.0.1:8000/admin/`

---

## 📱 Usage

### For Students
1. Go to "Register Students"
2. Enter your details and take a selfie
3. Submit registration
4. Wait for admin authorization

### For Admin
1. Log in to the admin dashboard
2. View registered students
3. Authorize students for recognition
4. Configure camera settings
5. View attendance records

### Marking Attendance
1. Click "Mark Attendance"
2. Click "Start Camera"
3. Position your face in the camera
4. Once recognized, attendance is automatically marked
5. Camera stops after successful recognition
6. Click "Start Camera" again for the next student

---

## 🔧 Configuration

### Camera Configuration
- Navigate to "Camera Configuration"
- Add camera name, source (0 for webcam, or IP camera URL)
- Set matching threshold (0.6 is recommended)

### Sound File
- Place `suc.wav` in the `app1/` directory for attendance confirmation sound

---

## 📁 Project Structure

```
project/
├── manage.py
├── requirements.txt
├── db.sqlite3
├── README.md
├── app1/
│   ├── models.py          # Database models
│   ├── views.py           # View logic and API endpoints
│   ├── urls.py            # URL routing
│   ├── admin.py           # Admin configuration
│   ├── forms.py           # Django forms
│   ├── migrations/        # Database migrations
│   └── suc.wav            # Success sound file
├── Project101/            # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── templates/             # HTML templates
│   ├── home.html
│   ├── capture_student.html
│   ├── capture_and_recognize.html
│   ├── student_list.html
│   ├── student_detail.html
│   ├── student_authorize.html
│   ├── student_attendance_list.html
│   ├── camera_config_form.html
│   ├── camera_config_list.html
│   ├── camera_config_delete.html
│   ├── login.html
│   ├── error.html
│   └── success.html
└── media/
    └── students/          # Student photos directory
```

---

## 🔐 Security Features

- **Admin Authentication**: Secure login required for admin functions
- **Authorization System**: Only authorized students are recognized
- **CSRF Protection**: Django's built-in CSRF token protection
- **Database Security**: SQLite with proper ORM usage

---

## 📊 Attendance Algorithm

1. **Face Detection**: MTCNN detects all faces in the frame
2. **Face Encoding**: InceptionResnetV1 converts faces to 512-dimensional vectors
3. **Face Matching**: Euclidean distance is calculated between:
   - Detected face encoding
   - Known student face encodings
4. **Threshold Comparison**: If minimum distance < configured threshold (default 0.6):
   - Student is recognized
   - Attendance is marked
5. **Duplicate Prevention**: System checks if student already marked attendance today

---

## 🎯 Performance Considerations

- Frame processing interval: 500ms (configurable)
- Face matching threshold: 0.6 (configurable in camera settings)
- Automatic camera stop after successful recognition prevents duplicate entries
- Check-out requires minimum 60 seconds after check-in

---

## 🐛 Troubleshooting

### Camera Not Accessible
- Check browser permissions for camera access
- Ensure `getUserMedia` is available in your browser
- Try a different camera index in configuration

### Face Not Recognized
- Ensure student is authorized in admin panel
- Check face matching threshold (try lowering it)
- Ensure good lighting and clear face visibility
- Verify student photo quality at registration

### Sound Not Playing
- Confirm `suc.wav` file exists in `app1/` directory
- Check browser audio settings
- Ensure pygame mixer initialized successfully

---

## 📝 Future Enhancements

- Liveness detection to prevent photo spoofing
- Email notifications for attendance records
- Mobile app integration
- Multi-face recognition in single frame
- Advanced reporting and analytics
- Integration with external attendance systems

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 👤 Author

**Arpit Sharma**

For questions or support, please contact the project maintainer.

---

## 🙏 Acknowledgments

- MTCNN for robust face detection
- FaceNet for facial recognition technology
- Django community for excellent web framework
- OpenCV for computer vision capabilities

---

**Last Updated**: April 27, 2026
