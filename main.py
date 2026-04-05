import asyncio, base64, io, os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "models/gemini-3-pro-image-preview"  # по умолчанию

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

async def _generate_one(prompt: str, ref_bytes: bytes | None, seed_offset: int, model: str) -> str | None:
    print(f"[{seed_offset}] Старт, модель: {model}")
    for attempt in range(3):  # 3 попытки
        try:
            parts = []
            if ref_bytes:
                parts.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/jpeg"))
            parts.append(types.Part.from_text(text=f"{prompt} [variation {seed_offset}]"))

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                )
            )
            print(f"[{seed_offset}] Готово!")
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    return base64.b64encode(part.inline_data.data).decode()
            return None

        except Exception as e:
            print(f"[{seed_offset}] Попытка {attempt+1} — ошибка: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 30 * (attempt + 1)  # 30, 60, 90 сек
                print(f"[{seed_offset}] Rate limit! Жду {wait} сек...")
                await asyncio.sleep(wait)
            else:
                return None
    return None

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    count: int = Form(default=4),
    model: str = Form(default="models/nano-banana-pro-preview"),
    reference: UploadFile | None = File(default=None)
):
    ref_bytes = await reference.read() if reference else None
    count = max(1, min(count, 6))
    
    tasks = [_generate_one(prompt, ref_bytes, i, model) for i in range(count)]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    images = []
    for r in results:
        if isinstance(r, str):
            images.append(r)
    
    return JSONResponse({"images": images})
@app.get("/models")
async def list_models():
    models = await asyncio.to_thread(client.models.list)
    return {"models": [m.name for m in models]}

