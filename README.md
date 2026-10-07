# Face Recognition Demo

A lightweight local webcam face-recognition project built with OpenCV. It enrolls people, trains a local LBPH model, recognizes faces from the webcam, and keeps a log of recognized identities in the `data/` folder.

## Features

- Enroll new people with live webcam samples
- Train a local recognition model from saved face images
- Recognize faces in real time from the webcam
- List enrolled people and sample counts
- Remove a person and their saved samples
- Save recognized results to `data/recognized_faces.txt`

## Project structure

```text
.
├── face_recognition_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── tests/
│   └── test_face_recognition_app.py
├── data/
│   ├── faces/
│   ├── labels.json
│   ├── trainer.yml
│   └── recognized_faces.txt
└── .venv/
```

## Requirements

- Windows, Linux, or macOS
- Python 3.10+
- Working webcam

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Usage

### 1. Enroll a person

```powershell
python face_recognition_app.py enroll Chetan
```

This captures multiple face samples from the webcam and stores them in `data/faces/<name>/`.

### 2. Train the model

```powershell
python face_recognition_app.py train
```

This trains the local LBPH recognizer using the saved samples.

### 3. Recognize faces

```powershell
python face_recognition_app.py recognize
```

The app shows recognized names on the webcam feed and appends entries to `data/recognized_faces.txt`.

### 4. List saved people

```powershell
python face_recognition_app.py list
```

### 5. Remove a person

```powershell
python face_recognition_app.py remove Chetan
```

Press `Q` to close any webcam window.

## Notes

- The app stores training data and logs locally on your machine.
- It is intended for educational and local demo use.
- It is not a secure identity system and should not be used for sensitive authentication.

## Important limitation

This is a learning/demo project, not a secure replacement for Windows Hello or enterprise biometric authentication. Face matching can be affected by lighting, camera angle, masks, and other variations, and should not be treated as a trusted security control.
