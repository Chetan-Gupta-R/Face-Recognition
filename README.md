# Local Face Recognition

A small, local webcam face-recognition demo. It stores face samples and its model in `data/` on this computer; it does not upload them anywhere.

## What you need

- A working webcam
- Python 3.10 or later

## Setup

In PowerShell, from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Use it

1. Enroll your face. Slowly turn your head while samples are being saved:

   ```powershell
   python face_recognition_app.py enroll Chetan
   ```

2. Train the local model:

   ```powershell
   python face_recognition_app.py train
   ```

3. Start recognition:

   ```powershell
   python face_recognition_app.py recognize
   ```

Press `Q` to close a camera window.

## Important limitation

This is an educational recognition demo, not a secure replacement for Windows Hello. Webcam image matching can be fooled and must not protect sensitive data. Windows Hello requires supported hardware, usually an IR camera, and its own secure sign-in system.
