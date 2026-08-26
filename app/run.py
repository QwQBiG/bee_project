"""本地启动脚本：python app/run.py

启动后自动打开浏览器访问 http://127.0.0.1:8000
"""

import threading
import webbrowser
import uvicorn


def open_browser():
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)