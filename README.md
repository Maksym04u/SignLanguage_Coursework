# Multilingual Sign Language Translator

Web platform for **gesture → text** (live webcam) and **text → gesture** (animated playback), with English and Ukrainian support. Includes an offline data-collection pipeline (OpenCV + MediaPipe) and a Keras GRU classifier for live recognition.

---

## Features

| Mode | Description |
|------|-------------|
| **Live Translation** | Webcam + MediaPipe in the browser → backend classifies gestures → builds a sentence |
| **Grammar correction** | Optional OpenAI pass on the assembled sentence (button **Correct**) |
| **Text to Gestures** | Type text → backend maps words/letters to recorded gestures → Canvas animation |
| **Lemmatization** | Optional OpenAI maps inflected forms to base glosses (`loves` → `love`, `любить` → `любити`) |

---

## Project layout

```
Sign-Language-Translator/
├── backend/                 # FastAPI API (auth, predict, text-to-sign, grammar)
│   ├── app/
│   └── alembic/             # DB migrations
├── frontend/                # React + Vite UI
├── dataset/
│   ├── labels.json          # Master catalog of all gesture classes
│   ├── model_labels.json    # Classes/order used by the trained model (inference)
│   └── gesture_lexicon.json # Lookup for text-to-gesture (+ static poses)
├── data/                    # Normalized training sequences (gitignored)
│   └── <data_dir>/<seq>/<frame>.npy
├── translation_data/        # Raw playback clips for text-to-gesture (gitignored)
│   └── <data_dir>/0.npy … 19.npy + meta.json
├── data_collection.py       # Record gestures (webcam, OpenCV)
├── model.py                 # Train Keras model → my_model.h5
├── my_functions.py          # Keypoint extraction (shared logic with frontend)
├── main.py                  # Legacy desktop live translator (optional)
└── requirements.txt         # Python dependencies (repo root)
```

**Two data formats (important):**

| Folder | Format | Used for |
|--------|--------|----------|
| `data/` | Normalized keypoints (wrist-centered) | Training + live `/predict` |
| `translation_data/` | Raw MediaPipe coords (`raw_v1`) | Text-to-gesture **motion** playback |

Both are written automatically during `data_collection.py` recording.

---

## Prerequisites

- **Python 3.10+** (TensorFlow 2.18 / Keras 3)
- **Node.js 18+** and npm
- **Webcam** (for data collection and live translation)
- **OpenAI API key** (optional but recommended for grammar + lemmatization)

---

## 1. Python environment (repo root)

```bash
cd Sign-Language-Translator

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
pip install openai
```

> `openai` is required for grammar correction and text-to-gesture lemmatization but is not pinned in `requirements.txt` yet.

---

## 2. Backend setup

### 2.1 Environment file

Create `backend/.env`:

```env
# Database (default: SQLite file in backend/)
DATABASE_URL=sqlite:///./sign_app.db

# Or PostgreSQL:
# DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/sign_translator

JWT_SECRET=change-me-to-a-long-random-string
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Optional — enables grammar + lemmatization
OPENAI_API_KEY=sk-...

# Optional — override model paths (defaults: repo root)
# SIGN_MODEL_PATH=../my_model.h5
# SIGN_LABELS_PATH=../dataset/model_labels.json
```

### 2.2 Database migrations

```bash
cd backend
alembic upgrade head
```

### 2.3 Trained model

Live recognition needs **`my_model.h5`** at the repo root (see [Train the model](#4-train-the-model) below).  
If missing, `/translate/predict` returns **503 Model assets missing**.

After training, these files must exist:

- `my_model.h5`
- `dataset/model_labels.json`

### 2.4 Run the API

From the **`backend/`** directory:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) → `{"ok": true}`  
API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

If the API runs on another host/port, edit `frontend/src/api.js`:

```js
export const API_BASE = "http://127.0.0.1:8000";
```

---

## 4. Using the web app

1. **Sign up / Log in** (JWT stored in `localStorage`).
2. Choose **Translation Language**: English or Ukrainian.
3. **Live Translation**
   - Click **Start** — browser loads MediaPipe (needs internet on first run).
   - Sign gestures; accepted tokens appear when confidence ≥ 96%.
   - **Add space** between words if needed.
   - **Correct** — sends the sentence for grammar fix (OpenAI).
4. **Text to Gestures**
   - Type a phrase → **Translate**.
   - Whole-word gestures are used when available; otherwise letters are spelled out.
   - With `OPENAI_API_KEY`, inflected words are lemmatized before lookup (`working` → `work`).
   - **Replay** / **Clear** as needed.

**Browser notes:** Live mode needs camera permission. MediaPipe WASM loads from CDN on first **Start**.

---

## 5. Gather gesture data

Run from the **repo root** (with venv active). You need a working webcam.

### 5.1 GUI (default)

```bash
python data_collection.py
```

Opens a setup window to enter new symbols (letters or words), language, then records.

### 5.2 CLI examples

```bash
# List all labels in dataset/labels.json
python data_collection.py --list

# Register new Ukrainian letters + record (skip confirmation)
python data_collection.py --no-gui --symbols "а,б,в" --language uk -y

# Register English word folders only (no camera)
python data_collection.py --setup-only --symbols "hello,love" --language en

# Re-record classes missing data
python data_collection.py --pick-existing --language en --missing-only -y

# Re-record specific class_ids
python data_collection.py --pick-existing --classes "en_word_hello,uk_letter_а" -y
```

### 5.3 Recording behaviour

- **30 sequences × 20 frames** per class (configurable in code: `DEFAULT_SEQUENCES`, `DEFAULT_FRAMES`).
- **Automatic pauses:** ~12 s before the first sequence of each sign, ~1 s between sequences.
- Press **`q`** in the camera window to stop.
- Each frame saves:
  - `data/<data_dir>/<sequence>/<frame>.npy` — normalized (for ML)
  - After each sequence: updates `translation_data/<data_dir>/` — raw 20-frame clip (`raw_v1`)

### 5.4 Verify raw playback (text-to-gesture)

```bash
python -m dataset.build_translation_data --status
```

---

## 6. Build lexicon (text-to-gesture lookup)

After recording (or updating) data under `data/`:

```bash
python -m dataset.build_gesture_lexicon
```

Writes / refreshes `dataset/gesture_lexicon.json` (representative poses + gloss keys).

The backend loads this file lazily on the first `/translate/text-to-sign` request.

---

## 7. Train the model

Training uses only classes with **complete** data under `data/`:

```bash
python model.py
```

Outputs:

| File | Purpose |
|------|---------|
| `my_model.h5` | Keras GRU classifier (live predict) |
| `dataset/model_labels.json` | Exact class list and order for inference |

Training skips incomplete classes automatically. Expect several minutes depending on dataset size and GPU/CPU.

**Optional:** export SavedModel format:

```bash
python convert.py
```

---

## 8. Legacy desktop app (optional)

Before the web UI, live translation ran as a local OpenCV window:

```bash
python main.py
```

Same keypoint pipeline as the web client, but inference runs locally. Useful for debugging without the backend.

---

## 9. End-to-end workflow (new gesture)

```text
1. python data_collection.py          # record data/ + translation_data/
2. python -m dataset.build_gesture_lexicon
3. python model.py                    # retrain if adding classes for LIVE
4. Restart backend (or wait for lazy reload of lexicon on next request)
5. Test in web UI (live + text-to-gesture)
```

For **text-to-gesture only** (no live class): steps 1–2 and 5 may be enough if the gloss already maps via lexicon letters/words.

---

## 10. Configuration reference

| Variable | Location | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | `backend/.env` | SQLite | User accounts |
| `JWT_SECRET` | `backend/.env` | (dev placeholder) | Sign JWT tokens |
| `OPENAI_API_KEY` | `backend/.env` | — | Grammar + lemmatization |
| `SIGN_MODEL_PATH` | env | `../my_model.h5` | Live model file |
| `SIGN_LABELS_PATH` | env | `../dataset/model_labels.json` | Label manifest |
| `API_BASE` | `frontend/src/api.js` | `http://127.0.0.1:8000` | Backend URL |

---

## 11. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `503 Model assets missing` | No `my_model.h5` | Run `python model.py` or set `SIGN_MODEL_PATH` |
| Live always low confidence | Bad/missing training data or lighting | Re-record; match collection conditions (mirror, distance) |
| Text-to-gesture has no motion | Missing `translation_data/` | Re-record class; check `--status` |
| Grammar / lemmas not working | No OpenAI key | Set `OPENAI_API_KEY` in `backend/.env`, restart uvicorn |
| `401` on translate routes | Not logged in / expired token | Log in again |
| MediaPipe fails on Start | No network first time | Allow CDN access; retry |
| CORS errors | Frontend not on `:5173` | Add origin in `backend/app/main.py` CORS middleware |
| DB errors on signup | Migrations not applied | `cd backend && alembic upgrade head` |

---

## 12. API overview (authenticated)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/signup` | Create account |
| `POST` | `/auth/login` | Get JWT |
| `GET` | `/auth/me` | Profile |
| `POST` | `/translate/predict` | Live gesture classification (20×126 keypoints) |
| `POST` | `/translate/sign-to-text` | Compose + grammar-correct tokens |
| `POST` | `/translate/text-to-sign` | Text → gesture frames JSON |

---

## 13. Further documentation

Thesis / technical deep-dives:

- [`docs/thesis/4.2-gesture-to-text.md`](docs/thesis/4.2-gesture-to-text.md) — live pipeline, buffer, confidence
- [`docs/thesis/4.3-text-to-gesture.md`](docs/thesis/4.3-text-to-gesture.md) — reverse translation, `translation_data`, playback
- [`docs/thesis/2.5-keras-model-architecture.md`](docs/thesis/2.5-keras-model-architecture.md) — model architecture
- [`docs/thesis/4-program-overview.md`](docs/thesis/4-program-overview.md) — full system overview


