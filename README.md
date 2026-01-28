# A simple FastAPI based server that handles movie trimming and subtitles merging

Installation:

1. Make sure ffmpeg is installed and is in the path
2. Project is tested with **python 3.14** but should work with earlier versions as well
3. Run `pip install -r requirements.txt`

Running:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 3600 --workers 2
```
