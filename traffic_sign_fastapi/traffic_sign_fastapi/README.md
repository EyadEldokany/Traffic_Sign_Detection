
# Traffic Sign Detection · FastAPI + YOLO (Dark Mode)

A FastAPI web app with a dark, creative UI to upload an **image or video**, run a YOLO model, and return the **annotated** media.

## Quickstart

1. Ensure your YOLO model is available (we default to `/mnt/data/traffic_sign_detection.pt`).  
   To override, set env var `MODEL_PATH=/path/to/model.pt`.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Open http://localhost:8000 and try it.

## Notes

- Images are annotated in-memory and saved as `outputs/annotated_*.jpg`.
- Videos are processed by YOLO with `save=True` and placed under `outputs/<job_id>/`.
- You can tweak confidence and image size from the UI.
- Dark/Light theme toggle is built-in.
