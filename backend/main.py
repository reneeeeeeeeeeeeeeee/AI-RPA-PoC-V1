# ERP-Assistent Backend
# FastAPI Server - Chat direkt an CogAgent + Excel-Workflow

import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from orchestrator import Orchestrator
from excel_parser import ExcelParser
from executor import ERPExecutor

app = FastAPI(title="ERP-Assistent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR   = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

orchestrator = Orchestrator()
excel_parser = ExcelParser()
executor     = ERPExecutor()

# Pro WebSocket-Session: eigene CogAgent-Historie
session_history: dict[str, list] = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    session_history[session_id] = []

    try:
        while True:
            data     = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "chat":
                await handle_chat(
                    data.get("text", ""),
                    data.get("file_path"),
                    websocket,
                    session_id,
                )
            elif msg_type == "confirm_start":
                await handle_execute(data.get("job_id"), websocket, session_id)
            elif msg_type == "abort":
                executor.abort()
                await websocket.send_json({"type": "system", "text": "Aborted."})
            elif msg_type == "reset_history":
                session_history[session_id] = []
                await websocket.send_json({"type": "system", "text": "History cleared."})

    except WebSocketDisconnect:
        session_history.pop(session_id, None)


async def handle_chat(text: str, file_path: Optional[str], ws: WebSocket, session_id: str):

    # ── Excel-Upload-Flow ──────────────────────────────────────────────
    if file_path:
        await ws.send_json({"type": "status", "model": "excel-parser", "state": "active"})
        await ws.send_json({"type": "assistant", "text": "Reading file...", "model": "excel-parser"})

        analysis = excel_parser.analyze(file_path)
        missing  = [r for r in analysis["rows"] if not r.get("Kostenstelle")]

        if missing:
            row_nums = [r["_row"] for r in missing]
            await ws.send_json({
                "type": "assistant",
                "text": f"Rows {row_nums} have no cost center. Which one should I use as fallback?",
                "model": "orchestrator",
                "pending": {"type": "missing_kostenstelle", "rows": row_nums,
                            "analysis": analysis, "prompt": text}
            })
            await ws.send_json({"type": "status", "model": "all", "state": "idle"})
            return

        await _plan_and_confirm(text, analysis, ws)
        await ws.send_json({"type": "status", "model": "all", "state": "idle"})
        return

    # ── Pending-Kontext (Rückfrage Kostenstelle) ───────────────────────
    if orchestrator.pending_context:
        ctx = orchestrator.pending_context
        if ctx.get("type") == "missing_kostenstelle":
            for r in ctx["analysis"]["rows"]:
                if not r.get("Kostenstelle"):
                    r["Kostenstelle"] = text.strip()
            await _plan_and_confirm(ctx["prompt"], ctx["analysis"], ws)
            orchestrator.pending_context = None
            await ws.send_json({"type": "status", "model": "all", "state": "idle"})
            return

    # ── Orchestrator zerlegt Prompt in Schritte, CogAgent fuehrt aus ──
    if executor.cogagent_ready():
        await _orchestrated_execute(text, ws, session_id)
    else:
        await ws.send_json({"type": "status", "model": "orchestrator", "state": "active"})
        response = orchestrator.chat(text)
        await ws.send_json({"type": "assistant", "text": response, "model": "orchestrator"})
        await ws.send_json({"type": "status", "model": "all", "state": "idle"})


async def _orchestrated_execute(prompt: str, ws: WebSocket, session_id: str):
    """Zerlegt Prompt in Einzelschritte und fuehrt jeden separat aus."""
    import re

    # Schritte aus Prompt extrahieren
    # Trennzeichen: Zeilenumbruch, " then ", " and then ", oder Satzenende vor neuem Verb
    steps = _split_into_steps(prompt)

    if len(steps) <= 1:
        # Einzelner Schritt - direkt ausfuehren
        await _direct_cogagent(prompt, ws, session_id)
        return

    await ws.send_json({
        "type": "assistant",
        "text": "I will execute " + str(len(steps)) + " steps:\n" + "\n".join(str(i+1) + ". " + s for i,s in enumerate(steps)),
        "model": "orchestrator"
    })

    history = session_history.get(session_id, [])

    history = session_history.get(session_id, [])
    for i, step in enumerate(steps):
        await ws.send_json({"type": "status", "model": "cogagent", "state": "active"})
        await ws.send_json({"type": "thinking", "text": f"Step {i+1}/{len(steps)}: {step[:50]}..."})

        import asyncio, time
        result = await asyncio.to_thread(_cogagent_step, step, history)
        action = result.get("action", "unknown")

        if action in ("click", "ocr_click"):
            reply = f"Step {i+1}: Clicked."
        elif action == "type":
            reply = f"Step {i+1}: Typed '{result.get('text','')}'."
        elif action == "scroll":
            reply = f"Step {i+1}: Scrolled {result.get('direction','down')}."
        elif action == "key":
            reply = f"Step {i+1}: Pressed {result.get('key','')}."
        elif action == "done":
            reply = f"Step {i+1}: Done."
        else:
            reply = f"Step {i+1}: {result.get('raw','?')[:100]}"

        screenshot_b64 = result.get("screenshot_b64", "")
        await ws.send_json({
            "type": "cogagent_result",
            "text": reply,
            "action": action,
            "screenshot": screenshot_b64,
            "model": "cogagent",
        })

        history.append(result)
        session_history[session_id] = history

        # Kurze Pause zwischen Schritten
        await asyncio.sleep(0.5)

    await ws.send_json({"type": "status", "model": "all", "state": "idle"})
    await ws.send_json({"type": "assistant", "text": f"All {len(steps)} steps completed.", "model": "orchestrator"})


def _split_into_steps(prompt: str) -> list:
    """Zerlegt einen Prompt in einzelne Aktionsschritte."""
    import re

    # Format "1: step 2: step 3: step"
    numbered = re.split(r"\s*\d+[.:)]\s*", prompt)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) > 1:
        return numbered

    # Zeilenumbrueche
    lines_split = [l.strip() for l in prompt.split("\n") if l.strip()]
    if len(lines_split) > 1:
        return lines_split

    # Aktionsverben als Trenner
    import re as _re
    matches = list(_re.finditer(r"(?:^|(?<=\s))(?:click|type|scroll|press|select|choose|fill)\b", prompt, _re.IGNORECASE))
    if len(matches) > 1:
        steps = []
        for i, m in enumerate(matches):
            end_pos = matches[i+1].start() if i+1 < len(matches) else len(prompt)
            steps.append(prompt[m.start():end_pos].strip())
        return steps

    return [prompt]


async def _direct_cogagent(task: str, ws: WebSocket, session_id: str):
    """Schickt Task direkt an CogAgent, fuehrt Aktion aus, sendet Ergebnis."""
    history = session_history.get(session_id, [])

    await ws.send_json({"type": "status", "model": "cogagent", "state": "active"})
    await ws.send_json({"type": "thinking", "text": "Taking screenshot..."})

    result = await asyncio.to_thread(_cogagent_step, task, history)

    action  = result.get("action", "unknown")
    raw     = result.get("raw", "")
    coord   = result.get("coordinate")
    changed = result.get("screen_changed", False)

    # Antwort aufbauen
    if action == "click" and coord:
        reply = f"Clicked at {coord}.\n\n_{raw}_"
    elif action == "type":
        reply = f"Typed: **{result.get('text','')}**\n\n_{raw}_"
    elif action == "key":
        reply = f"Pressed key: **{result.get('key','')}**\n\n_{raw}_"
    elif action == "scroll":
        reply = f"Scrolled **{result.get('direction','down')}**.\n\n_{raw}_"
    elif action == "error":
        reply = f"CogAgent not loaded."
    else:
        reply = f"Response: {raw}"

    if changed:
        reply += "\n\nScreen changed after action."

    # Screenshot als Base64 mitschicken
    screenshot_b64 = result.get("screenshot_b64", "")

    session_history[session_id].append(result)

    await ws.send_json({
        "type":       "cogagent_result",
        "text":       reply,
        "action":     action,
        "screenshot": screenshot_b64,
        "model":      "cogagent",
    })
    await ws.send_json({"type": "status", "model": "all", "state": "idle"})


def _cogagent_step(task: str, history: list) -> dict:
    """Laeuft in Thread: Screenshot -> OCR/CogAgent -> Aktion ausfuehren."""
    import base64, io, re, time as _t
    import pyautogui as _pag
    import pytesseract as _tess
    _tess.pytesseract.tesseract_cmd = "tesseract"

    def ocr_click(target, img=None, confidence=30):
        """Findet Text per OCR und klickt. Gibt (ax,ay) oder None zurueck."""
        hits = executor._find_text_on_screen(target, img)
        if not hits:
            print(f"[Step] OCR: '{target}' nicht gefunden")
            return None
        ox, oy, conf = hits[0]
        if conf < confidence:
            print(f"[Step] OCR: '{target}' Konfidenz zu niedrig ({conf})")
            return None
        sw = getattr(executor, "_scaled_size",  (1920,1080))[0]
        sh = getattr(executor, "_scaled_size",  (1920,1080))[1]
        ww = getattr(executor, "_window_size",  (3858,2034))[0]
        wh = getattr(executor, "_window_size",  (3858,2034))[1]
        ox2 = getattr(executor, "_window_offset", (0,0))
        ax = int(ox * ww / sw) + ox2[0]
        ay = int(oy * wh / sh) + ox2[1]
        executor._focus_ie()
        _t.sleep(0.3)
        _pag.click(ax, ay)
        print(f"[Step] OCR-Click: '{target}' -> ({ax},{ay})")
        return (ax, ay)

    def screenshot_b64():
        img = executor._screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # ── TYPE ────────────────────────────────────────────────────────────────
    field_match  = re.search(r"In\s+([A-Za-z ]{3,30})\s+type\s+", task, re.IGNORECASE)
    colon_match  = re.search(r"field:\s*(.+)$", task, re.IGNORECASE)
    type_match   = re.search(r"\btype\b", task, re.IGNORECASE)

    if type_match:
        # Text nach "field:" oder nach letztem Doppelpunkt
        if colon_match:
            type_text = colon_match.group(1).strip().strip('"\'')
        else:
            # Alles nach "type"
            m = re.search(r"\btype\b\s+(?:in\s+)?(?:the\s+)?(?:empty\s+)?(?:field[:\s]+)?(.+)", task, re.IGNORECASE)
            type_text = m.group(1).strip().strip('"\'') if m else ""

        if type_text:
            img = executor._screenshot()
            # Feld suchen und klicken
            if field_match:
                field_name = field_match.group(1).strip()
                ocr_click(field_name, img)
                _t.sleep(0.3)
            _pag.hotkey("ctrl", "a")
            _t.sleep(0.1)
            _pag.typewrite(type_text, interval=0.05)
            print(f"[Step] TYPE: '{type_text}'")
            _t.sleep(0.3)
            return {"action":"type","text":type_text,"screen_changed":True,
                    "screenshot_b64":screenshot_b64(),"raw":f"TYPE '{type_text}'"}

    # ── DROPDOWN ────────────────────────────────────────────────────────────
    dropdown_match = re.search(r'choose\s+["\']?([A-Za-z][A-Za-z0-9 ]{1,30}?)["\']?\s*$', task, re.IGNORECASE)
    dropdown_field = re.search(r"(?:At|In)\s+([A-Za-z ]{3,30})\s+click", task, re.IGNORECASE)

    if dropdown_match:
        choice = dropdown_match.group(1).strip()
        img    = executor._screenshot()

        # Dropdown-Feld klicken
        if dropdown_field:
            field_name = dropdown_field.group(1).strip()
            # Versuche 1: OCR mit geringerem Kontrast (psm 6)
            import pytesseract as _tess2
            _tess2.pytesseract.tesseract_cmd = "tesseract"
            ww = getattr(executor, "_window_size",  (3858,2034))[0]
            wh = getattr(executor, "_window_size",  (3858,2034))[1]
            sw = getattr(executor, "_scaled_size",  (1920,1080))[0]
            sh = getattr(executor, "_scaled_size",  (1920,1080))[1]
            off = getattr(executor, "_window_offset", (0,0))

            data = _tess2.image_to_data(img, config="--psm 6", output_type=_tess2.Output.DICT)
            found_y = None
            # Ignoriere obere 15% des Bildes (URL-Bar, Tabs)
            min_y = int(img.size[1] * 0.15)
            for idx2, t in enumerate(data["text"]):
                first_word = field_name.split()[0]
                if (first_word.lower() in t.lower()
                        and data["conf"][idx2] > 20
                        and data["top"][idx2] > min_y):
                    found_y = data["top"][idx2] + data["height"][idx2]//2
                    print(f"[Step] Dropdown-Label '{t}' bei y={found_y} (scaled, min_y={min_y})")
                    break

            if found_y:
                # Dropdown-Box ist rechts vom Label, ca. bei x=42% der Fensterbreite
                ax = int(0.42 * sw * ww / sw) + off[0]
                ay = int(found_y * wh / sh) + off[1]
            else:
                # Fallback: bekannte relative Position fuer "Main currency"
                # Aus Screenshot: ~42% links, ~27% oben
                ax = int(0.42 * ww) + off[0]
                ay = int(0.27 * wh) + off[1]
                print(f"[Step] Dropdown fallback position: ({ax},{ay})")

            executor._focus_ie(); _t.sleep(0.3)
            _pag.click(ax, ay)
            print(f"[Step] Dropdown geklickt bei ({ax},{ay})")
            _t.sleep(1.0)  # Warten bis Dropdown offen

        # In Suchfeld tippen
        _pag.typewrite(choice, interval=0.06)
        _t.sleep(0.8)

        # Option per OCR finden und klicken
        img2 = executor._screenshot()
        result = ocr_click(choice, img2)
        if not result:
            _pag.press("enter")
            print(f"[Step] Dropdown Enter fuer '{choice}'")

        return {"action":"click","coordinate":list(result) if result else [0,0],
                "screen_changed":True,"screenshot_b64":screenshot_b64(),
                "raw":f"DROPDOWN '{choice}'"}

    # ── SCROLL + optional SAVE ───────────────────────────────────────────────
    scroll_match = re.search(r"scroll\s+(down|up|to\s+(?:the\s+)?(?:end|bottom|top))", task, re.IGNORECASE)

    if scroll_match:
        executor._focus_ie()
        _t.sleep(0.3)
        direction = scroll_match.group(1).lower()
        if "up" in direction or "top" in direction:
            _pag.hotkey("ctrl", "Home")
        else:
            _pag.hotkey("ctrl", "End")
        _t.sleep(0.8)
        print(f"[Step] SCROLL: {direction}")

        # Save klicken falls im Task
        if re.search(r"\bsave\b", task, re.IGNORECASE):
            img = executor._screenshot()
            data = _tess.image_to_data(img, output_type=_tess.Output.DICT)
            sw = getattr(executor, "_scaled_size", (1920,1080))[0]
            sh = getattr(executor, "_scaled_size", (1920,1080))[1]
            ww = getattr(executor, "_window_size",  (3858,2034))[0]
            wh = getattr(executor, "_window_size",  (3858,2034))[1]
            off = getattr(executor, "_window_offset", (0,0))
            for idx, t in enumerate(data["text"]):
                if t.upper() in ("SAVE", "SPEICHERN") and data["conf"][idx] > 40:
                    ax = int((data["left"][idx]+data["width"][idx]//2)*ww/sw)+off[0]
                    ay = int((data["top"][idx]+data["height"][idx]//2)*wh/sh)+off[1]
                    print(f"[Step] SAVE bei ({ax},{ay})")
                    _pag.click(ax, ay)
                    _t.sleep(0.5)
                    break

        return {"action":"scroll","direction":direction,"screen_changed":True,
                "screenshot_b64":screenshot_b64(),"raw":f"SCROLL {direction}"}

    # ── CLICK via OCR oder CogAgent ─────────────────────────────────────────
    img    = executor._screenshot()
    result = executor._ask_cogagent(img, task, history)

    if result.get("action") not in ("error", "unknown", "done"):
        executor._execute_action(result)
        _t.sleep(0.8)

    after = executor._screenshot()
    result["screen_changed"] = executor._screen_changed(img, after)
    buf = io.BytesIO()
    after.save(buf, format="PNG")
    result["screenshot_b64"] = base64.b64encode(buf.getvalue()).decode()
    return result




async def _plan_and_confirm(prompt: str, analysis: dict, ws: WebSocket):
    await ws.send_json({"type": "status", "model": "orchestrator", "state": "active"})
    plan   = orchestrator.plan(prompt, analysis)
    job_id = orchestrator.save_job(plan, analysis)

    rows    = analysis["rows"]
    n_high  = sum(1 for r in rows if str(r.get("Prioritaet","")).lower() == "hoch")
    summary = (
        f"{len(rows)} orders -> ERP\n"
        f"{n_high} x express approval\n"
        f"Results will be written back to Excel"
    )
    await ws.send_json({
        "type":    "confirm",
        "text":    f"Plan:\n\n{summary}\n\nStart?",
        "job_id":  job_id,
        "model":   "orchestrator",
    })


async def handle_execute(job_id: str, ws: WebSocket, session_id: str):
    job = orchestrator.get_job(job_id)
    if not job:
        await ws.send_json({"type": "assistant", "text": "Job not found.", "model": "orchestrator"})
        return

    await ws.send_json({"type": "status", "model": "cogagent", "state": "active"})
    await ws.send_json({"type": "assistant", "text": "Started. Processing row by row.", "model": "executor"})

    results = []
    errors  = []

    for i, row in enumerate(job["rows"]):
        if executor.aborted:
            break
        await ws.send_json({
            "type":    "progress",
            "current": i + 1,
            "total":   len(job["rows"]),
            "text":    f"Row {i+1} - {row.get('Lieferant','?')}...",
        })
        result = await asyncio.to_thread(executor.process_row, row, job["steps_template"])
        if result["success"]:
            results.append(result)
            await ws.send_json({
                "type": "progress_done",
                "row":  i + 1,
                "text": f"Row {i+1} - {row.get('Lieferant')} -> {result.get('beleg_nr','OK')}",
            })
        else:
            errors.append({"row": i + 1, "reason": result.get("error"), "data": row})
            await ws.send_json({
                "type": "progress_error",
                "row":  i + 1,
                "text": f"Row {i+1} - {result.get('error')}",
            })

    excel_parser.write_results(job["file_path"], results, errors)
    summary = f"{len(results)} successful, {len(errors)} failed."
    await ws.send_json({"type": "assistant", "text": f"Done. {summary}", "model": "orchestrator"})

    if errors:
        err_text = "\n".join(f"- Row {e['row']}: {e['reason']}" for e in errors)
        await ws.send_json({
            "type":  "assistant",
            "text":  f"Failed rows:\n{err_text}\n\nSend summary by email?",
            "model": "orchestrator",
        })
    await ws.send_json({"type": "status", "model": "all", "state": "idle"})


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())
    return {"file_path": str(dest), "filename": file.filename}


@app.get("/health")
def health():
    return {
        "status":   "ok",
        "cogagent": executor.cogagent_ready(),
        "llm":      orchestrator.llm_ready(),
    }


app.mount("/", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
