# Mobile backend services

The companion [ClimateAdapt Mobile](https://github.com/TSgunawardena/climateadapt-mobile) Android APK checks faces and number plates and applies reviewed covers on the phone. Run this repository's prediction, places and photo services for connected analysis. Keep the V3 runtime data and project environment configured as described in the [main README](../README.md).

## Current Android workflow

1. Selecting a camera or gallery photo starts detection on Android using bundled YuNet face and open-lpr YOLOX-tiny plate models through ONNX Runtime.
2. The app opens a covered preview. Users inspect the complete photo, adjust missed or incorrect covers and confirm.
3. Android exports a fresh JPEG with permanent solid covers without copying source EXIF metadata.
4. Tapping Analyse sends the reviewed replacement to Script 55 and then to Gemma. Privacy detection and applying covers do not send the original to the laptop.
5. Optional GPS is used for separate V3 screening. A photo observation and a location screening result remain different evidence.

The privacy stage works offline with the bundled models. Gemma interpretation and geographic screening still need the laptop services. A missing or failed native privacy module does not automatically fall back to laptop processing or send an unchecked original.

See [on-device privacy](https://github.com/TSgunawardena/climateadapt-mobile/blob/main/docs/ON-DEVICE-PRIVACY.md) for native sources, model preparation, release-build keep rules and validation limits.

## Required services

| Component | Entry point | Default port |
| --- | --- | --- |
| Android privacy module | Included in the current mobile APK | None |
| V3 prediction API | `scripts/30_run_prediction_api.py` | `8000` |
| Places and geographic screening | `scripts/54_run_mobile_bff.py` | `8081` |
| Gemma photo analysis | `scripts/55_run_photo_bff.py` | `8080` |
| Ollama | `gemma3:4b` | `11434` |

Script 56 and a separate Python privacy environment are not required by the current Android APK. They remain available for older clients, as described below.

## First-time backend setup

Use the backend's existing `.venv`. Install the root dependencies and photo dependency file if they are not already installed. Do not reinstall a working environment merely to update documentation.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-photo.txt
```

If the environment does not exist, create it first using the Python setup in the main README.

Place analysis needs the separately supplied V3 raster runtime described in the main README. Gemma photo analysis needs Ollama with the vision model installed.

```powershell
ollama pull gemma3:4b
```

This is a different model from `climateadapt-planner:4b`. The mobile photo and places routes do not need the planning model or Gitea publication workflow.

## Start the current services

From the backend project root, run each command in a separate terminal. Keep Ollama running.

```powershell
.\.venv\Scripts\python.exe .\scripts\30_run_prediction_api.py
```

```powershell
.\.venv\Scripts\python.exe .\scripts\54_run_mobile_bff.py
```

```powershell
.\.venv\Scripts\python.exe .\scripts\55_run_photo_bff.py
```

Open the installed Android APK directly. Metro and Expo Go are not needed. Expo Go cannot execute the custom native privacy module. For JavaScript development, use a compatible native development build from the mobile repository; rebuild after changing native code or bundled assets. The custom privacy module currently has no iOS or web implementation.

## Health and phone connections

| Service | Laptop health check | Ready condition | Phone service address example |
| --- | --- | --- | --- |
| Prediction | `http://127.0.0.1:8000/health` | Healthy runtime | Phone uses the places service |
| Places | `http://127.0.0.1:8081/api/health` | `area_ready` is true | `http://192.168.1.20:8081` |
| Photo | `http://127.0.0.1:8080/api/health` | `vision_ready` is true | `http://192.168.1.20:8080` |

Replace the example IP with the laptop's current Wi-Fi IPv4 address. On the phone, `127.0.0.1` refers to the phone. For the local setup, connect both devices to the same network and allow the required development ports through the laptop firewall.

Configure the photo address in app Settings and the places address in Explore's connection settings. The current Android APK needs no privacy-service address or privacy HTTP health check. Check the laptop address after a restart or network change.

A successful health check does not prove that an image request will finish within its timeout. Run a real photo analysis and location query. If analysis times out, inspect Script 55 and Ollama logs. Initial model loading can add delay, but a timeout alone does not establish its cause.

## Privacy and interpretation limits

Detection can miss faces and plates or cover the wrong detail. The models do not identify people, read plate text or automatically cover other identifiers such as names, house numbers and distinctive clothing. Zero detections still require review.

The app keeps temporary working files rather than a photo archive. Its analysis waiting animation is decorative and does not apply privacy masks. The reviewed replacement goes to the laptop only when analysis is requested. The original gallery photograph remains under the user's control.

Source-image binding prevents accidental cover/image mismatches inside the app. It is not authentication or an independent proof to Script 55 that review occurred. The laptop services use local HTTP without user authentication, so reviewed uploads are not transport-encrypted. Do not expose these development APIs directly to the public internet.

Photo findings are preliminary observations, not measurements of water depth or flood probability. V3 location results are uncalibrated susceptibility screening. Neither establishes that a location will flood or is safe from future flooding.

## Current APK smoke checks

- Stop Script 56 and verify that selecting a photo still opens privacy review.
- Turn Wi-Fi and GPS off and verify detection and applying covers still work.
- Inspect the complete photo and correct missed or incorrect covers before confirming.
- Restore the network, run Scripts 30, 54 and 55, and make a real photo-analysis and location-screening request.
- Check that the reviewed photo remains available after an analysis failure.
- Record the tested APK build and source revision separately. Report only checks actually performed.

These are functional checks, not detector-accuracy, accessibility or comprehensive security benchmarks. Mobile contract and privacy tests are maintained in the mobile repository.

## Legacy laptop privacy helper

The instructions in this section apply only to older clients using the laptop privacy API. Keep these files for compatibility rather than treating Script 56 as a current Android dependency.

```text
scripts/
  56_run_privacy_bff.py
  privacy_core.py
  prepare_privacy_models.py
  requirements-privacy.txt
  test_privacy_bff.py
```

### Legacy setup and launch

First-time setup from the backend root

```powershell
python -m venv .venv-privacy
.\.venv-privacy\Scripts\python.exe -m pip install -r .\scripts\requirements-privacy.txt
.\.venv-privacy\Scripts\python.exe .\scripts\prepare_privacy_models.py
```

The helper downloads pinned model files, verifies SHA-256 checksums and stores them in `scripts/privacy-models`. It runs inference on the laptop CPU. These files are separate from the university V3 runtime package.

Start the legacy helper in its own terminal

```powershell
.\.venv-privacy\Scripts\python.exe .\scripts\56_run_privacy_bff.py
```

Its health endpoint is `http://127.0.0.1:8083/api/health`. A legacy client uses the laptop's address on port 8083 and expects `ready: true` with contract `privacy-v1`.

Unlike the current Android path, this legacy workflow sends the original prepared JPEG to `POST /api/privacy/detect` on the laptop. After review, `POST /api/privacy/apply` verifies the source digest and returns a flattened covered JPEG. That replacement can then go to Script 55. The helper handles images in request memory without creating an archive, but the original still crosses the network in this architecture.

### Legacy models and tests

| Target | Model | Source | Published licence |
| --- | --- | --- | --- |
| Faces | YuNet 2023mar | [OpenCV Zoo](https://huggingface.co/opencv/opencv_zoo/tree/d4938dfc9d4ec5d098bfa33e98b3f3345a236586/models/face_detection_yunet) | MIT |
| Plates | YOLOX-tiny, 640 input | [open-lpr-models](https://huggingface.co/faisalthaheem/open-lpr-models/tree/aed0df35ec2891c5a90437041ebeadb592cc516b/onnx) | Apache 2.0 as declared by the publisher |

Exact legacy revisions and checksums are in `scripts/prepare_privacy_models.py`. The Android downloader and checksums are maintained in the mobile repository. Retain the upstream model terms when redistributing weights.

Legacy unit and API tests use injected detections and do not need downloaded models or running services.

```powershell
.\.venv-privacy\Scripts\python.exe -m unittest discover -s .\scripts -p test_privacy_bff.py -v
```

They check image limits, cover bounds, flattened output, metadata removal, source binding, explicit review, request validation and unavailable-detector handling. They do not measure detector accuracy or execute Android native code.

## Git exclusions

Keep local environments, downloaded models, private photos and credentials out of source control.

```gitignore
.venv-privacy/
scripts/privacy-models/
```
