# FastLPR — Edge-Optimized License Plate Recognition

**A GPU-free Automatic License Plate Recognition (ALPR) system built to run on mini-PCs and edge devices**, with native support for Persian/Iranian plates.    
<img width="800" height="378" alt="ScreenRecording2026-07-10160428online-video-cutter com-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/7a1e1101-2c52-410c-82c0-9e966ee5fa07" />

---

## Table of Contents

* [Why FastLPR](https://www.google.com/search?q=%23why-fastlpr)
* [Architecture](https://www.google.com/search?q=%23architecture)
* [Persian Plate Handling](https://www.google.com/search?q=%23persian-plate-handling)
* [Tech Stack](https://www.google.com/search?q=%23tech-stack)
* [Project Structure](https://www.google.com/search?q=%23project-structure)
* [Getting Started](https://www.google.com/search?q=%23getting-started)
* [Dashboard](https://www.google.com/search?q=%23dashboard)
* [API Reference](https://www.google.com/search?q=%23api-reference)
* [Performance](https://www.google.com/search?q=%23performance)
* [Known Issues](https://www.google.com/search?q=%23known-issues)
* [License](https://www.google.com/search?q=%23license)

---

## Why FastLPR

Most open-source ALPR pipelines assume a GPU. FastLPR doesn't — every design decision was made to keep accuracy high while running entirely on CPU, on hardware as modest as a fanless mini-PC:

* **No vehicle-detection step.** Classic ALPR pipelines detect the car first, then the plate inside it. For a dedicated plate reader, that first step is pure overhead — FastLPR's detector targets the plate region directly.
* **LPRNet for OCR**, instead of a general-purpose text recognizer — small enough to run comfortably on CPU.
* **ONNX Runtime** for both the detector and the recognizer, to get the most out of whatever CPU it's given.
* **Motion-gated standby.** The heavy detector only runs when something is actually moving in frame; otherwise the pipeline idles.
* **Best-frame selection, not first-frame.** Rather than OCR-ing whatever frame happens to land on a processing tick, a scoring step picks the sharpest, best-framed candidate crop out of a short rolling window before it's sent to the recognizer.
* **Display never blocks on inference.** Frame capture and AI processing run on separate threads, so the live feed stays smooth regardless of how long a detection cycle takes.

## Architecture

FastLPR is split into three independent layers:

### 1. Capture & Display

A dedicated thread continuously pulls frames from the camera/video source and pushes them straight to the live feed. It does no AI work, so it can never be slowed down by the rest of the pipeline — the dashboard's video stream runs at the source's native frame rate even while a detection is being processed in the background.

### 2. Detection & Recognition (the processing thread)

Runs independently of the display thread, reading the latest available frame rather than competing for the camera:

1. **Motion gate** — skip everything if there's no relevant movement.
2. **Plate detection** — an ONNX-exported YOLO model locates the plate region directly (no vehicle-detection pass).
3. **Tracking** — ByteTrack (via `supervision`) follows the plate across frames so the same vehicle isn't re-processed from scratch every cycle.
4. **Frame scoring** — candidate crops of the tracked plate are scored on sharpness, aspect ratio, pixel area, and detector confidence; only the best crop is forwarded.
5. **OCR** — LPRNet reads the characters off the winning crop and returns a plate string plus a confidence score.

When a detection lands, the bounding box is also pushed to the display thread for a short hold window (~1s) so it's visible on the live feed without needing to wait for the next processing cycle.

### 3. Traffic State Machine & Logging

#### Enter/Exit Detection

A hypothetical line divides the camera frame into two halves — an entry zone and an exit zone. The center point of each detected plate bounding box is tracked across frames. When a vehicle's plate center crosses the dividing line from the entry side to the exit side, it is recorded as an exit; crossing in the opposite direction is recorded as an entry.

Every recognized plate is then checked against the whitelist and the table of currently-active (in-progress) sessions, then routed into one of a few outcomes:

* Not whitelisted → access denied, logged with the reason.
* Entering, no active session → new session opened.
* Entering, but a session is already open for that plate → the stale session is closed with a warning and a fresh one is opened (handles a vehicle re-triggering entry without ever registering an exit).
* Exiting, with a matching active session → session closed, duration calculated, written to history.
* Exiting, with no active session → logged with a warning rather than silently dropped (handles a whitelisted vehicle whose entry was missed).

All of this is exposed to operators through a FastAPI backend and a single-file dashboard — see [API Reference](https://www.google.com/search?q=%23api-reference) and [Dashboard](https://www.google.com/search?q=%23dashboard).

## Persian Plate Handling

Iranian plates mix digits with a Persian letter and, occasionally, a category marker (e.g. ceremonial or accessibility plates) — none of which a Latin-only OCR model handles natively. FastLPR's recognizer outputs each plate as a space-separated sequence of tokens (digits and Latin letter-names, e.g. `1 2 Saad 1 2 3 4 5`), and a shared mapping table converts that into the correct Persian glyphs everywhere a plate is displayed or entered — dashboard tables, the whitelist form, and manual session entry all use the same mapping, so a plate typed in by an operator and a plate read by the camera end up in the same canonical format.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Plate detection | YOLO, exported to ONNX, served via ONNX Runtime |
| Tracking | ByteTrack (`supervision`) |
| OCR | LPRNet, exported to ONNX, served via ONNX Runtime |
| Backend | FastAPI, Uvicorn |
| Database | SQLite |
| Frontend | Single-file HTML + Tailwind CSS (CDN) + vanilla JS — no build step |
| Core | Python 3.x, OpenCV, NumPy |

## Project Structure

```
fastLPR/
├── api.py                 # FastAPI app and HTTP routes
├── pipeline.py            # Threaded capture/display + detection/recognition pipeline
├── detector.py            # YOLO (ONNX) plate-ROI detector
├── recognizer.py          # LPRNet (ONNX) OCR
├── frame_reader.py        # Threaded video/camera frame reader
├── index.html             # Dashboard (RTL, Persian)
├── weights/               # ONNX model weights for the detector and recognizer
├── data/
│   ├── database.py         # GateDatabase — SQLite access layer
│   ├── samples/             # Sample video(s) for testing without a live camera
│   └── stored_images/       # Saved plate crops, served at /data/stored_images (created at runtime)
├── utils/
│   ├── motion.py            # Motion-based standby trigger
│   ├── storage.py           # Saves cropped plate images to disk
│   ├── plate_batcher.py     # Multi-frame crop scoring & best-candidate selection
│   └── visualizer.py        # Drawing helpers (trigger line, detection overlay)
└── requirements.txt         # pip dependencies

```

## Getting Started

### Prerequisites

* Python 3.14+
* A CPU-only machine is fine — no CUDA/GPU setup required.
* Exported ONNX weights for both the detector and the recognizer, placed in `weights/`.

### Installation

```bash
git clone https://github.com/LikeaBubble/fastLPR.git
cd fastLPR

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

```

### Configuration

* **Video source** — set in dashboard or `pipeline.py` (`self.video_source`), or change it at runtime via `POST /system/source` with `{"source": ...}` (camera index, RTSP/HTTP URL, or a video file path).
* **CORS** — already enabled (`CORSMiddleware`, all origins) in `api.py`, so the dashboard can call the API from any origin out of the box. Tighten `allow_origins` once you know the actual domain you're deploying to.

### Running

```bash
uvicorn api:app --host 0.0.0.0 --port 8000

```

Then open `index.html` directly in a browser (or serve it however you prefer — it talks to the API over plain HTTP, no build step needed).

## Dashboard

A single-file, right-to-left Persian dashboard ships with the project — no separate frontend build, no framework:

* **Live Monitor** — live MJPEG feed, start/stop controls, currently-active sessions, and a manual "register entry" form for operator overrides.
* **Whitelist** — add/remove authorized plates through a segmented input (digits / letter / digits) that mirrors the real plate layout instead of free-text entry.
* **Logs & Reports** — today's traffic or full history, both paginated.
* **Manual control** — open/close sessions.

Saved plate-crop images (entry/exit evidence) are served directly by the API at `/data/stored_images/...`, so the dashboard can show thumbnails without any extra setup.

## API Reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/system/status` | Pipeline running state and current video source |
| `POST` | `/system/source` | Change the video source |
| `POST` | `/system/start` | Start the capture/processing pipeline |
| `POST` | `/system/stop` | Stop the pipeline |
| `GET` | `/whitelist` | List authorized plates |
| `POST` | `/whitelist/add` | Add a plate |
| `DELETE` | `/whitelist/remove/{plate}` | Remove a plate |
| `GET` | `/logs/today` | Today's traffic history, paginated |
| `GET` | `/logs` | Full traffic history, paginated |
| `GET` | `/logs/sessions` | Vehicles currently inside |
| `POST` | `/logs/sessions/add` | Manually register an entry |
| `DELETE` | `/logs/sessions/delete/{plate}` | Manually clear an active session |
| `POST` | `/logs/history/add` | Manually insert a complete history record |
| `GET` | `/video_feed` | MJPEG live stream |
| `GET` | `/data/stored_images/{filename}` | Static file serving for saved plate-crop images |

## Performance

FastLPR is designed to prioritize **per-plate accuracy** over raw pipeline FPS.

To guarantee the clearest possible read, the system intentionally employs a "wait-for-close-up" strategy. As a vehicle approaches, the plate becomes larger and sharper. By pacing the cycle, the pipeline captures the highest-quality crops rather than the first available ones:

* The detector runs every 8th frame (`SKIP_FRAMES = 7`) and is gated by motion.
* For each tracked plate, the system waits to collect **10 candidate crops**.
* The **best 5 crops** (ranked by sharpness, size, and aspect ratio) are sent as a single batch to LPRNet for OCR and voting.

### Raw Speeds vs. Full-Cycle Latency

Raw model inference speeds differ significantly from the practical "cycle time" (the duration from a plate's first detection to its logged final string):

* **Detection (640x640):** ~10 FPS on baseline hardware.
* **Recognizer (94x24):** ~25 FPS on baseline hardware (batched).
* **Full Cycle Time:** ~4 seconds on baseline hardware. This is dominated by the wait time to gather 10 valid frames during the vehicle's approach.

### Hardware Estimates (Inference-Bound Scaling)

Because ONNX Runtime on these models is largely single-thread bound, performance scales predictably with single-core CPU speeds.

*Note: While the Raspberry Pi 5 provides solid CPU throughput for this workload, the **Radxa Zero 3W** remains our explicitly favored target for dedicated edge hardware due to its superior compute efficiency in minimalist deployments.*

| Device Class | Example CPU | GB6 SC* | Est. Detection | Est. Recognizer | Est. Full Cycle |
| --- | --- | --- | --- | --- | --- |
| Thin Laptop | Intel Core i7-1355U | 2368 | ~24.7 FPS | ~61.7 FPS | ~1.6 s |
| Office Laptop | Intel Core i5-1135G7 | 1587 | ~16.5 FPS | ~41.3 FPS | ~2.4 s |
| Edge Mini-PC | Intel Processor N100 | ~1024 | ~10.7 FPS | ~26.7 FPS | ~3.7 s |
| **Baseline** | **Intel Core i7-4702MQ** | **960** | **~10.0 FPS** | **~25.0 FPS** | **~4.0 s** |
| SBC (Popular) | Raspberry Pi 5 (4C/4T) | ~774 | ~8.1 FPS | ~20.2 FPS | ~5.0 s |
| Legacy Mini-PC | Intel Celeron J4125 | ~343 | ~3.6 FPS | ~8.9 FPS | ~11.2 s |

**GB6 SC = Geekbench 6 single-core score. Use this ratio to estimate full cycle times on unlisted hardware.*

## Known Issues

* **Small detection training set.** The plate detector was trained on roughly 500 images, which limits its robustness across varied lighting, angles, and plate conditions. Detection accuracy and generalization will improve significantly with a larger, more diverse dataset.
* **Enter/exit mechanism edge cases.** The line-crossing logic works well under normal conditions but can misclassify direction in specific situations — for example, when a vehicle stalls near the dividing line, reverses, or when two vehicles are in frame simultaneously and their tracked centers become ambiguous.

## License

The following open-source resources were used in building FastLPR:

* **[IR-LPR](https://github.com/mut-deep/IR-LPR)** — dataset used for training both the plate detector and the character recognizer, covering Iranian/Persian license plates.
* **[LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)** — the LPRNet architecture used as the basis for the OCR model.
* **[FastAPI](https://fastapi.tiangolo.com/)** — the web framework powering the backend API.
* **[OpenCV (cv2)](https://opencv.org/)** — used throughout the pipeline for frame capture, image processing, and visualization.
* **[SQLite](https://www.sqlite.org/)** — embedded database for session and traffic history storage.
