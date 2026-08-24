from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import cv2

from generator import (
    DEFECTS,
    DEMO_OUTPUT_DIR,
    encode_jpeg,
    generate_defect,
    get_board,
    list_boards,
    original_image_path,
)


app = FastAPI(title="Smart PCBA Defect Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    board: str = Field(..., examples=["pcb_09"])
    component: str = Field(..., examples=["capacitor"])
    instance: int = Field(1, ge=1)
    defect: str = Field(..., examples=["move_25px"])


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/defects")
def defects():
    return list(DEFECTS.values())


@app.get("/api/boards")
def boards():
    return list_boards()


@app.get("/api/boards/{board_id}")
def board_detail(board_id: str):
    board = get_board(board_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Board not found")
    return board


@app.get("/api/boards/{board_id}/image")
def board_image(board_id: str, max_width: int = Query(1400, ge=200, le=4000)):
    try:
        path = original_image_path(board_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=404, detail="Could not read board image")
    return Response(
        content=encode_jpeg(image, max_width=max_width),
        media_type="image/jpeg",
    )


@app.post("/api/generate")
def generate(request: GenerateRequest):
    try:
        result = generate_defect(
            request.board,
            request.component,
            request.instance,
            request.defect,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "board": result["board"],
        "component": result["component"],
        "instance": result["instance"],
        "filename": result["filename"],
        "segmentationMethod": result["segmentation_method"],
        "maskScore": result["mask_score"],
        "defect": result["defect"],
        "originalBox": result["original_box"],
        "defectBox": result["defect_box"],
        "imageSize": result["image_size"],
        "imageUrl": f"/api/demo-images/{result['filename']}",
    }


@app.get("/api/demo-images/{filename}")
def demo_image(filename: str, max_width: int = Query(1600, ge=200, le=4000)):
    path = DEMO_OUTPUT_DIR / filename
    if (
        not path.exists()
        or path.suffix.lower() != ".png"
        or path.resolve().parent != DEMO_OUTPUT_DIR.resolve()
    ):
        raise HTTPException(status_code=404, detail="Generated image not found")

    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=404, detail="Could not read generated image")
    return Response(
        content=encode_jpeg(image, max_width=max_width),
        media_type="image/jpeg",
    )
