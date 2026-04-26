## Local LanguageTool Setup

This backend is configured to use a local LanguageTool server by default:

- `GRAMMAR_BACKEND=local`
- `LANGUAGETOOL_SERVER_URL=http://localhost:8081`

### 1) Download and unzip LanguageTool

Download a recent snapshot and unzip it (see official docs).

### 2) Ensure Java is installed

LanguageTool server requires Java 8+.

### 3) Create `server.properties`

Inside the unzipped LanguageTool directory, create a `server.properties` file.
If you do not use fastText, it can be empty.

### 4) Start server

Run from the unzipped LanguageTool directory:

```powershell
java -cp languagetool-server.jar org.languagetool.server.HTTPServer --config server.properties --port 8081 --allow-origin
```

### 5) Test server

```powershell
curl -Method Post -Body "language=en-GB&text=I go to school yesterday" http://localhost:8081/v2/check
```

### 6) Start backend

Make sure `backend/.env` has:

```env
GRAMMAR_BACKEND=local
LANGUAGETOOL_SERVER_URL=http://localhost:8081
LANGUAGETOOL_TIMEOUT_SECONDS=8
GRAMMAR_PUBLIC_FALLBACK=true
```

Restart the backend after env changes.

## Notes

- If local server is down, backend can fallback to public API when
  `GRAMMAR_PUBLIC_FALLBACK=true`.
- Backend now logs grammar failures instead of silently returning `None`.
