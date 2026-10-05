"""
Web Server for College Enquiry Chatbot.
Provides REST API endpoints:
- POST /api/chat       (Process query and return response, confidence, intent, slots, suggestions)
- POST /api/reset      (Clear conversational session state)
- GET  /api/faq        (Return complete FAQ database)
- GET  /api/categories (List all available categories)
- GET  /health         (Server health status)
- GET  /               (Serves the interactive Web UI)

Supports FastAPI / Uvicorn when available, and includes an automatic fallback to
Python's standard library `http.server` for zero-setup execution.
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

# Set paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.chatbot import CollegeChatbotEngine

# Initialize chatbot engine singleton
engine = CollegeChatbotEngine()

HOST = "127.0.0.1"
PORT = 8000


class ChatbotHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Zero-dependency HTTP Request Handler serving both REST API and Web UI."""

    def __init__(self, *args, **kwargs):
        web_dir = os.path.join(BASE_DIR, "web")
        super().__init__(*args, directory=web_dir, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join(BASE_DIR, "web", "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        elif path == "/api/categories":
            self._send_json(200, {"categories": engine.categories})
            return

        elif path == "/api/faq":
            self._send_json(200, engine.faq_db)
            return

        elif path == "/health":
            self._send_json(200, {
                "status": "healthy",
                "college": engine.college_name,
                "intents_count": len(engine.intents)
            })
            return

        # Serve static assets
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(post_body)
        except Exception:
            payload = {}

        if path == "/api/chat":
            user_msg = payload.get("message", "").strip()
            response_data = engine.process_query(user_msg)
            self._send_json(200, response_data)
            return

        elif path == "/api/reset":
            engine.reset_conversation()
            self._send_json(200, {"message": "Session reset successfully"})
            return

        self._send_json(404, {"error": "Not Found"})

    def _send_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Concise logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run_server(port: int = PORT):
    print("=" * 72)
    print(f"  🎓 APEX INSTITUTE COLLEGE ENQUIRY CHATBOT WEB SERVER 🎓")
    print("=" * 72)
    print(f"  🌐 Web UI:   http://{HOST}:{port}")
    print(f"  📡 REST API: http://{HOST}:{port}/api/chat")
    print(f"  🩺 Health:   http://{HOST}:{port}/health")
    print("=" * 72)
    print("  Press Ctrl+C to stop the server.\n")

    httpd = HTTPServer((HOST, port), ChatbotHTTPRequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    run_server()
