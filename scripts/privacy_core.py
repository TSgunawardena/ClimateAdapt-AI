"""Image-only privacy processing. No recognition, OCR, network requests or archives."""
import base64
import hashlib
import io
import math
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 1600 * 1600
MAX_REGIONS = 100
CONTRACT = 'privacy-v1'

def open_photo(encoded):
    if not isinstance(encoded, str) or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError('The privacy photo is missing or too large.')
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ValueError('The privacy photo is missing or too large.')
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != 'JPEG' or source.width * source.height > MAX_PIXELS:
                raise ValueError('Use a prepared JPEG no larger than 1600 by 1600 pixels.')
            if getattr(source, 'n_frames', 1) != 1:
                raise ValueError('Use a still JPEG photograph.')
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.load()
        return image, hashlib.sha256(raw).hexdigest()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError('The photo could not be decoded.') from error

def validate_regions(regions):
    if not isinstance(regions, list) or len(regions) > MAX_REGIONS:
        raise ValueError('Use at most 100 privacy covers.')
    clean = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError('Invalid privacy cover.')
        values = [region.get(key) for key in ('x', 'y', 'width', 'height')]
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
            raise ValueError('Invalid privacy cover coordinates.')
        x, y, width, height = values
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
            raise ValueError('A privacy cover is outside the photo.')
        kind = region.get('kind', 'manual')
        if kind not in ('face', 'plate', 'manual'):
            raise ValueError('Invalid privacy cover type.')
        clean.append(dict(x=x, y=y, width=width, height=height, kind=kind))
    return clean

def apply_covers(image, regions):
    """Flatten solid covers into fresh pixels, then encode without source metadata."""
    regions = validate_regions(regions)
    output = Image.new('RGB', image.size)
    output.paste(image)
    draw = ImageDraw.Draw(output)
    for region in regions:
        x, y, w, h = [region[key] for key in ('x', 'y', 'width', 'height')]
        left, top = math.floor(x * image.width), math.floor(y * image.height)
        right, bottom = math.ceil((x + w) * image.width), math.ceil((y + h) * image.height)
        draw.rectangle((left, top, min(image.width - 1, right), min(image.height - 1, bottom)), fill=(0, 0, 0))
    buffer = io.BytesIO()
    output.save(buffer, format='JPEG', quality=92, subsampling=0)
    data = buffer.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('The masked image is too large. Choose a smaller photo.')
    return dict(contract=CONTRACT, image_base64=base64.b64encode(data).decode('ascii'),
                width=output.width, height=output.height, bytes=len(data), mask_count=len(regions),
                sha256=hashlib.sha256(data).hexdigest(), mime_type='image/jpeg')

def region_from_pixels(box, size, kind):
    """Expand detections before preview, so the user sees the actual cover extent."""
    width, height = size
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    margin_x, margin_y = max(8, w * .2), max(8, h * .2)
    left, top = max(0, x - margin_x), max(0, y - margin_y)
    right, bottom = min(width, x + w + margin_x), min(height, y + h + margin_y)
    if right <= left or bottom <= top:
        return None
    return dict(x=left / width, y=top / height, width=(right - left) / width,
                height=(bottom - top) / height, kind=kind)

class PrivacyDetector:
    """Pinned YuNet face detection and YOLOX-tiny plate detection on CPU only."""
    def __init__(self, directory):
        import cv2
        import onnxruntime as ort
        from prepare_privacy_models import MODELS, verify
        for name, (_, digest) in MODELS.items():
            if not verify(directory / name, digest):
                raise RuntimeError('Missing or invalid privacy model. Run prepare_privacy_models.py.')
        self.cv = cv2
        cv2.setNumThreads(2)
        self.face = cv2.FaceDetectorYN_create(str(directory / 'face_detection_yunet_2023mar.onnx'), '',
                                            (640, 640), .65, .3, 5000)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.plate = ort.InferenceSession(str(directory / 'plate_yolox_tiny_640.onnx'),
                                         sess_options=options, providers=['CPUExecutionProvider'])
        self.lock = threading.Lock()

    def _plates(self, image):
        # The pinned export consumes RGB float32 pixels in 0..255, centred letterbox.
        scale = min(640 / image.width, 640 / image.height)
        resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        px, py = round((640 - resized.width) / 2), round((640 - resized.height) / 2)
        canvas = Image.new('RGB', (640, 640), (114, 114, 114))
        canvas.paste(resized, (px, py))
        tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None]
        rows = self.plate.run(None, {self.plate.get_inputs()[0].name: np.ascontiguousarray(tensor)})[0]
        if rows.shape != (1, 8400, 6):
            raise RuntimeError('Unexpected plate model output. Privacy detection was stopped.')
        rows = rows[0]
        scores = rows[:, 4] * rows[:, 5]
        candidates, confidences = [], []
        for row, score in zip(rows[scores >= .25], scores[scores >= .25]):
            cx, cy, w, h = map(float, row[:4])
            left, top = max(0, (cx - w / 2 - px) / scale), max(0, (cy - h / 2 - py) / scale)
            right, bottom = min(image.width, (cx + w / 2 - px) / scale), min(image.height, (cy + h / 2 - py) / scale)
            if right > left and bottom > top:
                candidates.append([left, top, right - left, bottom - top])
                confidences.append(float(score))
        indices = self.cv.dnn.NMSBoxes(candidates, confidences, .25, .4)
        return [candidates[int(i)] for i in np.asarray(indices).reshape(-1)]

    def detect(self, image):
        with self.lock:
            regions = []
            scale = min(1, 1280 / max(image.size))
            small = image.resize((round(image.width * scale), round(image.height * scale)))
            pixels = np.asarray(small)[:, :, ::-1].copy()
            self.face.setInputSize(small.size)
            _, faces = self.face.detect(pixels)
            if faces is not None:
                for face in faces:
                    region = region_from_pixels(face[:4] / scale, image.size, 'face')
                    if region:
                        regions.append(region)
            plates = self._plates(image)
            # Four overlapping crops preserve small plates in wider field photographs.
            if max(image.size) > 900:
                cw, ch = min(800, image.width), min(800, image.height)
                for x, y in sorted(set((x, y) for x in (0, image.width - cw) for y in (0, image.height - ch))):
                    crop = image.crop((x, y, x + cw, y + ch))
                    plates.extend([[b[0] + x, b[1] + y, b[2], b[3]] for b in self._plates(crop)])
            for box in plates:
                region = region_from_pixels(box, image.size, 'plate')
                if region and not any(region['kind'] == old['kind'] and
                                      abs(region['x'] - old['x']) < .02 and abs(region['y'] - old['y']) < .02
                                      for old in regions):
                    regions.append(region)
            if len(regions) > MAX_REGIONS:
                raise ValueError('Too many private details. Choose a simpler photo.')
            return validate_regions(regions)
