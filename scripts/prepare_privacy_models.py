"""One-time, pinned model download. No photos are handled by this script."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

MODEL_DIR = Path(__file__).resolve().parent / 'privacy-models'
MODELS = {
    'face_detection_yunet_2023mar.onnx': (
        'https://huggingface.co/opencv/opencv_zoo/resolve/d4938dfc9d4ec5d098bfa33e98b3f3345a236586/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'plate_yolox_tiny_640.onnx': (
        'https://huggingface.co/faisalthaheem/open-lpr-models/resolve/aed0df35ec2891c5a90437041ebeadb592cc516b/onnx/plate_yolox_tiny_640.onnx',
        'e5350bb2152eb44fa129b2438d1e39bbbf3c27accfe51cd893ce6e17d028d0b9'),
}

def verify(path, expected):
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODELS.items():
        target = MODEL_DIR / name
        if verify(target, expected):
            print(name, 'verified')
            continue
        temporary = target.with_suffix('.part')
        try:
            print('Downloading', name)
            with urlopen(url, timeout=120) as response, temporary.open('wb') as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            if not verify(temporary, expected):
                raise RuntimeError('Model checksum mismatch: ' + name)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(name, 'downloaded and verified')
    print('Privacy models ready. Start 56_run_privacy_bff.py next.')

if __name__ == '__main__':
    main()
