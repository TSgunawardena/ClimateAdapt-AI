# Mobile backend services

The Android Expo app is maintained in [climateadapt-mobile](https://github.com/TSgunawardena/climateadapt-mobile). Run its Python services from this backend repository. Keep the V3 runtime data and existing project environment configured as described in the main README.

## Files

```text
scripts/
  30_run_prediction_api.py
  54_run_mobile_bff.py
  55_run_photo_bff.py
  56_run_privacy_bff.py
  privacy_core.py
  prepare_privacy_models.py
  requirements-photo.txt
  requirements-privacy.txt
  test_privacy_bff.py
```

Scripts 54 and 55 are the working mobile integration scripts from the existing local backend. Keep those tested copies when committing this integration. The privacy update adds Script 56 and its helpers alongside them. The original integration sources are also available in the mobile repository's `backend-addon` directory.

## Existing photo and places environment

For a new installation, use the backend's existing `.venv`, install its root requirements and then the small photo dependency file. Do not reinstall a working environment merely to update documentation.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\scripts\requirements-photo.txt
```

Place analysis needs the separately supplied V3 raster runtime described in the main README. Gemma photo analysis needs Ollama with the vision model installed.

```powershell
ollama pull gemma3:4b
```

This is a different model from `climateadapt-planner:4b`. The mobile photo workflow does not need to invoke the planning model.

## Privacy environment

First-time setup only, from the backend root

```powershell
python -m venv .venv-privacy
.\.venv-privacy\Scripts\python.exe -m pip install -r .\scripts\requirements-privacy.txt
.\.venv-privacy\Scripts\python.exe .\scripts\prepare_privacy_models.py
```

Model preparation downloads pinned versions and verifies SHA-256 checksums. Files are saved in `scripts/privacy-models`. The service runs inference locally on the CPU and does not require CUDA. Internet access is needed for installation and model downloads, not for subsequent privacy inference.

The privacy models are separate from the existing university V3 runtime package. That package should not be assumed to contain them.

## Start services

Use a separate terminal for each command and keep Ollama running. Existing installations can go straight to these commands.

```powershell
.\.venv\Scripts\python.exe .\scripts\30_run_prediction_api.py
```

```powershell
.\.venv\Scripts\python.exe .\scripts\54_run_mobile_bff.py
```

```powershell
.\.venv\Scripts\python.exe .\scripts\55_run_photo_bff.py
```

```powershell
.\.venv-privacy\Scripts\python.exe .\scripts\56_run_privacy_bff.py
```

Start Expo from the separate mobile project, not this backend repository.

```powershell
npx expo start --go --lan --port 8082
```

## Health and phone connections

| Service | Laptop health check | Ready condition | Phone service address example |
| --- | --- | --- | --- |
| Prediction | `http://127.0.0.1:8000/health` | Healthy runtime | Phone uses the places service |
| Places | `http://127.0.0.1:8081/api/health` | `area_ready` is true | `http://192.168.1.20:8081` |
| Photo | `http://127.0.0.1:8080/api/health` | `vision_ready` is true | `http://192.168.1.20:8080` |
| Privacy | `http://127.0.0.1:8083/api/health` | `ready` is true and contract is `privacy-v1` | `http://192.168.1.20:8083` |

Replace the example IP with the laptop's current Wi-Fi IPv4 address. On a phone, `127.0.0.1` refers to the phone. The phone and laptop need network reachability, and the firewall must allow the development services. Configure photo and privacy addresses in app Settings and the places address in Explore's connection settings.

Do not expose these unauthenticated development APIs directly to the public internet.

## Privacy workflow

1. Taking or selecting a photo automatically sends its prepared JPEG to `POST /api/privacy/detect` on the laptop.
2. The app displays detected faces and plates under solid black covers. Users must inspect the whole image and can adjust covers for missed or incorrect detections.
3. After confirmation, `POST /api/privacy/apply` checks the source image digest and exports a fresh JPEG with permanent covers and no EXIF metadata.
4. The reviewed JPEG replaces the app's working upload image. Tapping Analyse sends this replacement to Script 55 and then to Gemma.
5. Optional GPS goes through separate location screening. Notes and GPS are not sent to the privacy detector.

The helper does not create a photo archive. Images are handled in request memory; the app keeps temporary working copies. This does not remove the original from the user's gallery. The app's analysis waiting animation is decorative and does not apply privacy masks.

The detectors target faces and number plates. They do not automatically cover every possible identifying detail, such as names, house numbers or distinctive clothing. A result containing no detections still requires review. Failure or cancellation must not silently send an unchecked original to analysis.

The source digest binds the covers to their input image. It is not authentication or an independent proof to Script 55 that privacy review occurred. The review gate is part of the mobile client.

## Models and licences

| Target | Model | Source | Published licence |
| --- | --- | --- | --- |
| Faces | YuNet 2023mar | [OpenCV Zoo](https://huggingface.co/opencv/opencv_zoo/tree/d4938dfc9d4ec5d098bfa33e98b3f3345a236586/models/face_detection_yunet) | MIT |
| Plates | YOLOX-tiny, 640 input | [open-lpr-models](https://huggingface.co/faisalthaheem/open-lpr-models/tree/aed0df35ec2891c5a90437041ebeadb592cc516b/onnx) | Apache 2.0 as declared by the publisher |

Exact model revisions and checksums are in `prepare_privacy_models.py`. No OCR model or text recognition is used. Keep downloaded models out of Git and retain their upstream terms when redistributing them. Test representative local plates and image conditions rather than claiming accuracy from a synthetic test image.

## Verification

Privacy unit and API tests use injected detector results and do not need downloaded models or running services.

```powershell
.\.venv-privacy\Scripts\python.exe -m unittest discover -s .\scripts -p test_privacy_bff.py -v
```

The tests check image limits, cover bounds, flattened output, metadata removal, source binding, explicit review, request validation and unavailable-detector handling. They do not measure detection accuracy.

On the phone, check a suitable synthetic image with two faces, two plates and visible standing water. Confirm automatic preview opening, correct coverage, manual corrections, zero detections, cancellation and privacy-service failure. Check that the water and drain remain visible to the photo model. Record missed detections and false covers honestly.

## Git exclusions

Keep these out of source control

```gitignore
.venv-privacy/
scripts/privacy-models/
```

Commit source, dependency files and tests. Do not commit virtual environments, downloaded models, private photos or local credentials.
