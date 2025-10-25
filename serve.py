#!/usr/bin/env python3
"""
Simple HTTP server for Halloween scare system
Serves static files from ./static/ directory
"""

import os
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial


class StaticFileHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves from ./static/ directory"""

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        """Custom logging with colors"""
        print(f"\033[94m[HTTP]\033[0m {format % args}")


def get_local_ip():
    """Get local IP address"""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"


def main():
    """Start HTTP server"""

    # Configuration
    port = 8080
    host = "0.0.0.0"
    directory = "./static"

    # Ensure static directory exists
    if not os.path.exists(directory):
        print(f"❌ Error: {directory} directory not found!")
        print(f"   Please create it first")
        return

    # Get local IP
    local_ip = get_local_ip()

    # Create handler with custom directory
    handler = partial(StaticFileHandler, directory=directory)

    # Create server
    server = HTTPServer((host, port), handler)

    print("=" * 80)
    print("🎃 HALLOWEEN SCARE SYSTEM - HTTP SERVER 🎃")
    print("=" * 80)
    print(f"\n✅ Serving files from: {os.path.abspath(directory)}")
    print(f"\n📡 Server URLs:")
    print(f"   Local:   http://localhost:{port}")
    print(f"   Network: http://{local_ip}:{port}")
    print(f"\n👉 Open http://{local_ip}:{port} on your Chromebook")
    print(f"\n⌨️  Press Ctrl+C to stop the server\n")
    print("=" * 80)
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutting down server...")
        server.shutdown()
        print("✅ Server stopped")


if __name__ == "__main__":
    main()
