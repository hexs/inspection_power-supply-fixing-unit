# server.py
from __future__ import annotations
from typing import Any
import io

import numpy as np
import cv2
from flask import Flask, request, jsonify, abort, Response, send_file, render_template_string

app = Flask(__name__)


def sanitize_value(value: Any) -> Any:
    """
    Convert value to JSON-safe structure.
    """
    simple_types = (bool, int, float, str, type(None))

    if isinstance(value, simple_types):
        return value

    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]

    if isinstance(value, dict):
        return {str(k): sanitize_value(v) for k, v in value.items()}

    return str(type(value))


def resolve_path(root: Any, path: str, sep: str = "/") -> Any:
    """
    Traverse path like "camera/0/setting/CAP_PROP_FRAME_WIDTH"
    """
    if not path:
        return root

    parts = [p for p in path.split(sep) if p != ""]
    node = root

    for part in parts:
        if isinstance(node, dict):
            if part not in node:
                abort(404, description=f"Key '{part}' not found in dict.")
            node = node[part]
        elif isinstance(node, (list, tuple)):
            try:
                idx = int(part)
            except ValueError:
                abort(404, description=f"Index '{part}' is not a valid integer.")
            try:
                node = node[idx]
            except IndexError:
                abort(404, description=f"Index {idx} out of range.")
        else:
            abort(404, description=f"Cannot go deeper at '{part}'.")

    return node


@app.route("/")
def index():
    shared_state = app.config.get("shared_state")
    if not shared_state:
        abort(500, description="shared_state is not initialized.")

    cameras = (shared_state.get("camera") or {})

    def _to_int(x: str) -> int:
        try:
            return int(x)
        except ValueError:
            return 0

    camera_ids = sorted(cameras.keys(), key=_to_int)

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Camera Control</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background: #111; color: #eee; }
            button { cursor: pointer; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; font-size: 14px; }
            button:hover { background: #0056b3; }
            button.all-btn { background: #28a745; font-size: 16px; margin-bottom: 20px; }
            button.all-btn:hover { background: #1e7e34; }
            .camera-grid { display: flex; flex-wrap: wrap; gap: 16px; }
            .camera-card { border: 1px solid #444; border-radius: 8px; padding: 10px; background: #1e1e1e; }
            .camera-card img { max-width: 480px; height: auto; display: block; background: #000; margin-top: 10px; }
            .controls { margin-top: 10px; display: flex; gap: 10px; }
        </style>
        <script>
            function setApi(k, v) {
                fetch(`/api/set?k=${k}&v=${v}`)
                .then(r => r.json())
                .then(d => {
                    if(d.success) console.log("Set OK:", k, v);
                    else alert("Error setting value");
                })
                .catch(e => console.error(e));
            }

            function captureAll(ids) {
                ids.forEach(id => {
                    setApi(`camera/${id}/fusion_state`, 'REQUESTED');
                });
            }
        </script>
    </head>
    <body>
        <h1>Camera Dashboard</h1>

        <button class="all-btn" onclick='captureAll({{ camera_ids | tojson }})'>CAPTURE ALL CAMERAS</button>

        <div class="camera-grid">
            {% for cid in camera_ids %}
            <div class="camera-card">
                <h2>Camera {{ cid }}</h2>
                <div class="controls">
                    <button onclick="setApi('camera/{{ cid }}/fusion_state', 'REQUESTED')">Capture Fusion</button>
                    <a href="/api/get_image?id={{ cid }}&im=fused_result" target="_blank">
                        <button style="background:#6c757d;">Get Fused Result</button>
                    </a>
                </div>
                <img src="/api/get_image?id={{ cid }}&im=latest_frame" alt="Preview">
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, camera_ids=camera_ids)


def _parse_value(v: str) -> Any:
    """Helper to guess type from string value"""
    if v.lower() == 'true': return True
    if v.lower() == 'false': return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


@app.route("/api/set", methods=["GET"])
@app.route("/api/set_data", methods=["GET"])
def set_data():
    """
    Usage: /api/set?k=camera/0/fusion_state&v=REQUESTED
    """
    shared_state = app.config.get("shared_state")
    if shared_state is None:
        abort(500, description="Shared state not initialized")

    k = request.args.get("k")
    v = request.args.get("v")
    sep = request.args.get("sep", "/")

    if not k or v is None:
        abort(400, description="Missing 'k' (path) or 'v' (value)")

    val = _parse_value(v)
    parts = [p for p in k.split(sep) if p]
    if not parts:
        abort(400, description="Path is empty")

    target_key = parts[-1]
    parent_path = sep.join(parts[:-1])

    try:
        parent = shared_state
        if parent_path:
            parent = resolve_path(shared_state, parent_path, sep)
        if isinstance(parent, dict):
            parent[target_key] = val
        elif isinstance(parent, list):
            idx = int(target_key)
            parent[idx] = val
        else:
            abort(400, description="Target parent is not a dict or list.")

        return jsonify({"success": True, "k": k, "v": val})

    except Exception as e:
        abort(400, description=str(e))


@app.route("/api/get", methods=["GET"])
@app.route("/api/get_data", methods=["GET"])
def get_data():
    shared_state = app.config.get("shared_state")
    if shared_state is None:
        abort(500)

    v = request.args.get("v")
    if v is None:
        abort(400)

    sep = request.args.get("sep", "/")
    try:
        value = resolve_path(shared_state, v, sep=sep)
        sanitized = sanitize_value(value)
        return jsonify(sanitized)
    except Exception as e:
        abort(404, description=str(e))


def _get_camera_node(shared_state: dict, cam_id: str) -> dict:
    cameras = shared_state.get("camera") or {}
    cam = cameras.get(str(cam_id))
    if cam is None:
        abort(404)
    return cam


def _extract_image_from_state(cam: dict, im_key: str) -> Any:
    if im_key == "fused_result":
        return cam.get("fused_result")
    if im_key == "latest_frame":
        latest = cam.get("latest_frame_data")
        if not latest or len(latest) < 2:
            return None
        return latest[1]
    abort(400)


def _encode_image_to_response(img: Any, quality: int = 100) -> Response:
    if isinstance(img, (bytes, bytearray, memoryview)):
        return Response(bytes(img), mimetype="image/jpeg")

    if isinstance(img, np.ndarray):
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok: abort(500)
        return Response(buf.tobytes(), mimetype="image/jpeg")

    abort(500, description="Unknown image type")


@app.route("/api/get_image", methods=["GET"])
def get_image():
    shared_state = app.config.get("shared_state")
    if shared_state is None: abort(500)

    cam_id = request.args.get("id", "0")
    im_key = request.args.get("im")
    if not im_key: abort(400)

    cam = _get_camera_node(shared_state, cam_id)
    img = _extract_image_from_state(cam, im_key)
    if img is None: abort(404)

    return _encode_image_to_response(img)


def run_server(shared_state: dict) -> None:
    app.config["shared_state"] = shared_state
    host = shared_state.get('ipv4', '0.0.0.0')
    port = shared_state.get('port', 5000)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)