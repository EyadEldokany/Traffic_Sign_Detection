import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO
import cv2

# ----- Paths & Config -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "traffic_sign_detection.pt"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Traffic Sign Detection (YOLO)")

# Static and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load model once at startup
@app.on_event("startup")
def load_model():
    global model
    model = YOLO(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result_url": None, "media_type": None, "error": None},
    )


@app.post("/detect", response_class=HTMLResponse)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
):
    # Save upload to outputs dir
    suffix = os.path.splitext(file.filename)[1].lower()
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(OUTPUT_DIR, temp_name)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Determine media type
    content_type = file.content_type or ""
    is_image = content_type.startswith("image/") or suffix in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]
    is_video = content_type.startswith("video/") or suffix in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    ]

    if is_image:
        # Inference on image
        results = model.predict(source=temp_path, conf=conf, imgsz=imgsz, verbose=False)
        annotated = results[0].plot()
        out_name = f"annotated_{os.path.splitext(temp_name)[0]}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, annotated)
        result_url = f"/outputs/{out_name}"
        media_type = "image"

    elif is_video:
        # Streaming inference on video
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        out_name = f"annotated_{os.path.splitext(temp_name)[0]}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        for result in model.predict(
            source=temp_path, conf=conf, imgsz=imgsz, stream=True, verbose=False
        ):
            frame = result.plot()
            writer.write(frame)

        cap.release()
        writer.release()
        result_url = f"/outputs/{out_name}"
        media_type = "video"

    else:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Unsupported file type. Please upload an image or a video.",
                "result_url": None,
                "media_type": None,
            },
        )

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result_url": result_url, "media_type": media_type, "error": None},
    )


# JSON API endpoint
@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...), conf: float = Form(0.25), imgsz: int = Form(640)):
    suffix = os.path.splitext(file.filename)[1].lower()
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(OUTPUT_DIR, temp_name)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    content_type = file.content_type or ""
    is_image = content_type.startswith("image/") or suffix in [
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    ]
    is_video = content_type.startswith("video/") or suffix in [
        ".mp4", ".avi", ".mov", ".mkv", ".webm"
    ]

    if is_image:
        results = model.predict(source=temp_path, conf=conf, imgsz=imgsz, verbose=False)
        annotated = results[0].plot()
        out_name = f"annotated_{os.path.splitext(temp_name)[0]}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, annotated)
        return {"type": "image", "result_url": f"/outputs/{out_name}"}

    if is_video:
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_name = f"annotated_{os.path.splitext(temp_name)[0]}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        for result in model.predict(source=temp_path, conf=conf, imgsz=imgsz, stream=True, verbose=False):
            frame = result.plot()
            writer.write(frame)

        cap.release()
        writer.release()
        return {"type": "video", "result_url": f"/outputs/{out_name}"}

    return {"error": "Unsupported file type"}
