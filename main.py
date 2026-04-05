import asyncio, base64, json, os, time, uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

async def _generate_one(prompt: str, ref_parts: list, seed_offset: int, model: str) -> str | None:
    print(f"[{seed_offset}] Старт, модель: {model}")
    for attempt in range(3):
        try:
            parts = list(ref_parts)
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
                wait = 30 * (attempt + 1)
                print(f"[{seed_offset}] Rate limit! Жду {wait} сек...")
                await asyncio.sleep(wait)
            else:
                return None
    return None

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    count: int = Form(default=4),
    model: str = Form(default="models/gemini-2.0-flash-exp"),
    references: list[UploadFile] = File(default=[])
):
    ref_parts = []
    for ref in references:
        data = await ref.read()
        if data:
            mime = ref.content_type or "image/jpeg"
            ref_parts.append(types.Part.from_bytes(data=data, mime_type=mime))

    count = max(1, min(count, 6))
    tasks = [_generate_one(prompt, ref_parts, i, model) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    images = [r for r in results if isinstance(r, str)]
    return JSONResponse({"images": images})

@app.get("/models")
async def list_models():
    models = await asyncio.to_thread(client.models.list)
    image_models = []
    for m in models:
        name = m.name if hasattr(m, 'name') else str(m)
        image_models.append(name)
    return {"models": image_models}

# --- History API (file-based, simple) ---

@app.post("/history")
async def save_history(entry: dict):
    entry_id = entry.get("id", str(uuid.uuid4()))
    entry["id"] = entry_id
    entry["saved_at"] = time.time()
    filepath = HISTORY_DIR / f"{entry_id}.json"
    filepath.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "id": entry_id}

@app.get("/history")
async def get_history():
    entries = []
    for f in sorted(HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            entries.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return {"entries": entries}

@app.delete("/history/{entry_id}")
async def delete_history(entry_id: str):
    filepath = HISTORY_DIR / f"{entry_id}.json"
    if filepath.exists():
        filepath.unlink()
    return {"ok": True}
