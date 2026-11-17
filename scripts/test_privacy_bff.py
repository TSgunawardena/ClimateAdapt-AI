import base64
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from privacy_core import apply_covers, open_photo, validate_regions, region_from_pixels
spec = importlib.util.spec_from_file_location('privacy_bff', Path(__file__).with_name('56_run_privacy_bff.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fixture():
    image = Image.new('RGB', (80, 60), (220, 150, 100))
    exif = Image.Exif()
    exif[270] = 'Private source annotation'
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', exif=exif)
    raw = buffer.getvalue()
    return raw, base64.b64encode(raw).decode()

class Detector:
    def detect(self, image):
        return [dict(x=.25, y=.25, width=.5, height=.5, kind='plate')]

class PrivacyTests(unittest.TestCase):
    def test_black_pixels_are_flattened_and_source_metadata_removed(self):
        raw, encoded = fixture()
        image, digest = open_photo(encoded)
        self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
        output = apply_covers(image, Detector().detect(image))
        data = base64.b64decode(output['image_base64'])
        self.assertNotIn(b'Private source annotation', data)
        with Image.open(io.BytesIO(data)) as result:
            self.assertFalse(result.getexif())
            self.assertLess(max(result.getpixel((40, 30))), 5)
            self.assertGreater(result.getpixel((5, 5))[0], 200)
        self.assertGreater(image.getpixel((40, 30))[0], 200)

    def test_zero_regions_still_exports_fresh_jpeg_without_metadata(self):
        _, encoded = fixture()
        image, _ = open_photo(encoded)
        result = apply_covers(image, [])
        self.assertEqual(result['mask_count'], 0)
        self.assertNotIn(b'Exif', base64.b64decode(result['image_base64']))

    def test_region_validation_rejects_nonfinite_outside_and_large_lists(self):
        for regions in [[dict(x=float('nan'), y=0, width=1, height=1)],
                        [dict(x=.9, y=0, width=.2, height=1)], [dict(x=0, y=0, width=-1, height=1)], [{}] * 101]:
            with self.assertRaises(ValueError):
                validate_regions(regions)

    def test_detection_margin_is_clipped_to_photo(self):
        region = region_from_pixels((0, 0, 10, 10), (80, 60), 'face')
        self.assertEqual(region['x'], 0)
        self.assertGreater(region['width'], 10 / 80)
        validate_regions([region])

    def test_api_detect_apply_no_archive_and_source_binding(self):
        _, encoded = fixture()
        with TestClient(module.create_app(Detector())) as client:
            health = client.get('/api/health').json()
            self.assertTrue(health['ready'])
            self.assertEqual(health['photo_storage'], 'no_archive')
            detected = client.post('/api/privacy/detect', json={'image_base64': encoded})
            self.assertEqual(detected.status_code, 200)
            self.assertEqual(detected.headers['cache-control'], 'no-store')
            scan = detected.json()
            payload = dict(image_base64=encoded, reviewed=True, source_sha256=scan['source_sha256'], regions=scan['regions'])
            self.assertEqual(client.post('/api/privacy/apply', json=payload).status_code, 200)
            self.assertEqual(client.post('/api/privacy/apply', json={**payload, 'reviewed': False}).status_code, 422)
            self.assertEqual(client.post('/api/privacy/apply', json={**payload, 'source_sha256': 'wrong'}).status_code, 422)

    def test_missing_or_failed_detector_does_not_return_success(self):
        _, encoded = fixture()
        with TestClient(module.create_app()) as client:
            self.assertFalse(client.get('/api/health').json()['ready'])
            self.assertEqual(client.post('/api/privacy/detect', json={'image_base64': encoded}).status_code, 503)
        class Broken:
            def detect(self, image):
                raise RuntimeError('No inference')
        with TestClient(module.create_app(Broken())) as client:
            self.assertEqual(client.post('/api/privacy/detect', json={'image_base64': encoded}).status_code, 503)

    def test_bad_image_and_json_return_errors(self):
        with TestClient(module.create_app(Detector())) as client:
            self.assertEqual(client.post('/api/privacy/detect', json={'image_base64': 'not-an-image'}).status_code, 422)
            self.assertEqual(client.post('/api/privacy/detect', content=b'{', headers={'content-type': 'application/json'}).status_code, 400)

    def test_oversize_request_is_rejected_before_json_processing(self):
        with TestClient(module.create_app(Detector())) as client:
            response = client.post('/api/privacy/detect', content=b' ' * (module.MAX_REQUEST_BYTES + 1),
                                   headers={'content-type': 'application/json'})
            self.assertEqual(response.status_code, 413)

if __name__ == '__main__':
    unittest.main()
