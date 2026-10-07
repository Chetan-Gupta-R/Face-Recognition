# Face Recognition Demo

A local face-recognition demo built with Python and OpenCV. Use the browser dashboard to enroll people, train an LBPH model, view live recognition results, and review a local recognition log.

## Features

- Browser dashboard with live camera preview and face-detection overlays
- Enroll people from the browser or command line
- Train a local OpenCV LBPH recognizer
- View enrolled profiles, sample counts, model status, and recent recognition events
- Remove a profile and its saved samples
- Store face samples, model files, and recognition history under `data/`

## Requirements

- Python 3.10 or newer
- A webcam
- Windows, macOS, or Linux

The browser dashboard uses the browser's camera permission and the Python standard-library HTTP server. OpenCV and NumPy are installed from `requirements.txt`.

## Setup

From the project folder, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Browser Dashboard

Start the local server:

```powershell
python web_app.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765), then:

1. Select **Connect camera** and grant camera permission in your browser.
2. Enter a name and select **Enroll face**. Keep your face visible while the dashboard captures 30 samples.
3. Select **Train model** after enrolling or changing people.
4. Select **Start recognition** to see live match labels and update the local activity log.

The browser preview stays in the browser. For recognition and enrollment, reduced-size camera frames are sent to the Python server on this device at `127.0.0.1`; no camera data is sent to a remote service. The server accepts browser mutations only from its local dashboard origin. Stop it with `Ctrl+C` in the terminal.

The model is marked stale after face data changes. Train again before recognition. The dashboard is intended for a single local user and does not provide authentication for network access.

## Command-Line Use

The CLI is available through `face_recognition_app.py`:

```powershell
python face_recognition_app.py enroll Chetan
python face_recognition_app.py train
python face_recognition_app.py recognize
python face_recognition_app.py list
python face_recognition_app.py remove Chetan
```

Enrollment and recognition use an OpenCV camera window in CLI mode; press `Q` to close it. `app.py` contains the core implementation, and `face_recognition_app.py` is the CLI compatibility entry point.

Optional interfaces are also available: `python frontend.py` opens the desktop window, and `python gui_app.py` opens a terminal menu.

## Data Files

```text
data/
├── faces/<person-name>/     # Captured face samples
├── labels.json              # Names associated with model labels
├── trainer.yml              # Trained LBPH model
└── recognized_faces.txt     # Timestamped recognition events
```

The face samples and recognition log contain personal data. Keep them local, and remove them when they are no longer needed.

## Tests

Run the unit tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

## Limitations

This is an educational demo, not a secure identity or authentication system. Face matching can be affected by lighting, camera angle, masks, and other conditions; do not use it to protect sensitive information or make consequential decisions.
