"""Local CPU privacy helper on 8083. Existing photo and place services stay separate."""
import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from privacy_core import CONTRACT, MAX_IMAGE_BYTES, apply_covers, open_photo, validate_regions

MAX_REQUEST_BYTES = 8 * 1024 * 1024

def create_app(detector=None):
    app = FastAPI(title='ClimateAdapt local privacy helper')
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])
    lock = asyncio.Lock()

    @app.middleware('http')
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api/health')
    async def health():
        return dict(ok=detector is not None, contract=CONTRACT, mode='local_privacy',
                    ready=detector is not None, photo_storage='no_archive',
                    processing='local_cpu', max_image_bytes=MAX_IMAGE_BYTES)

    async def payload(request):
        if request.headers.get('content-type', '').split(';')[0] != 'application/json':
            raise HTTPException(415, 'Send a prepared JPEG as JSON base64.')
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_REQUEST_BYTES:
                raise HTTPException(413, 'This photo is too large for privacy checking.')
        try:
            value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, UnicodeError):
            raise HTTPException(400, 'Invalid privacy request.') from None

    @app.post('/api/privacy/detect')
    async def detect(request: Request):
        if detector is None:
            raise HTTPException(503, 'Privacy models are not ready. Run prepare_privacy_models.py and restart Script 56.')
        if lock.locked():
            raise HTTPException(429, 'Privacy checking is busy. Try again shortly.')
        value = await payload(request)
        async with lock:
            started = time.perf_counter()
            try:
                image, digest = await run_in_threadpool(open_photo, value.get('image_base64'))
                regions = await run_in_threadpool(detector.detect, image)
                regions = validate_regions(regions)
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
            except Exception:
                raise HTTPException(503, 'Automatic privacy detection failed. Your original has not been sent to Gemma.') from None
            return dict(contract=CONTRACT, source_sha256=digest, width=image.width, height=image.height,
                        regions=regions, elapsed_ms=round((time.perf_counter() - started) * 1000),
                        detectors=['YuNet 2023mar', 'open-lpr YOLOX-tiny'], review_required=True)

    @app.post('/api/privacy/apply')
    async def apply(request: Request):
        value = await payload(request)
        if value.get('reviewed') is not True:
            raise HTTPException(422, 'Review the whole photo before applying covers.')
        try:
            image, digest = await run_in_threadpool(open_photo, value.get('image_base64'))
            if value.get('source_sha256') != digest:
                raise ValueError('The photo changed. Run privacy checking again.')
            return await run_in_threadpool(apply_covers, image, value.get('regions'))
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    return app

if __name__ == '__main__':
    import uvicorn
    from privacy_core import PrivacyDetector
    try:
        detector = PrivacyDetector(Path(__file__).resolve().parent / 'privacy-models')
        print('Privacy models loaded on CPU. No OCR or face recognition is performed.')
    except Exception as error:
        print('Privacy models unavailable:', type(error).__name__)
        print('Run prepare_privacy_models.py, then restart this service.')
        detector = None
    uvicorn.run(create_app(detector), host='0.0.0.0', port=8083, access_log=False)
