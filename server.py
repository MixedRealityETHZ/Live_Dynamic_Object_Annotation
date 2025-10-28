# sam2_ws_server_mask_only.py
import asyncio, json
import cv2, numpy as np, websockets, torch
from collections import deque
from dataclasses import dataclass, field
from sam2.build_sam import build_sam2_camera_predictor

HOST, PORT = "0.0.0.0", 8765


try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    sdpa_kernel(SDPBackend.MATH).__enter__()
except Exception:
    pass

def decode_jpeg(buf: bytes):
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def png_from_mask(mask_u8: np.ndarray) -> bytes:
    """
    Encode a HxW uint8 mask (0 or 255) as a 1-channel PNG.
    """
    ok, png = cv2.imencode(".png", mask_u8)
    return png.tobytes() if ok else b""

@dataclass
class Session:
    predictor: any
    q: deque = field(default_factory=lambda: deque(maxlen=1))
    inited: bool = False
    have_prompt: bool = False
    frame_wh: tuple | None = None
    pending_point: tuple[int,int] | None = None
    obj_id: tuple = (1,)
    auto_seed_done: bool = False           # <--- NEW

async def process_loop(ws, st: Session):
    while True:
        # Wait for a frame
        while not st.q:
            await asyncio.sleep(0.001)

        jpg = st.q.pop()  # newest only
        frame = decode_jpeg(jpg)
        if frame is None:
            # keep client alive with an empty mask
            h, w = 360, 640
            empty = np.zeros((h, w), np.uint8)
            await ws.send(png_from_mask(empty))
            continue

        h, w = frame.shape[:2]
        wh = (w, h)
        #print(f"[SERVER] frame received: {wh}")

        # Initialize once to current stream size (if not initialized yet or size changed)
        if (not st.inited) or (st.frame_wh != wh):
            print(f"[SERVER] load_first_frame @ {wh}")
            st.predictor.load_first_frame(frame)
            st.inited = True
            st.have_prompt = False
            st.frame_wh = wh

            # Auto-select a single test point once (center of the frame)
            if not st.auto_seed_done:
                cx, cy = w // 2, h // 2
                st.pending_point = (cx, cy)
                st.auto_seed_done = True
                print(f"[SERVER] auto-seed test point at ({cx},{cy})")

        # If a click arrived, re-seed on this frame and send the initial logits mask immediately
        if st.pending_point is not None:
            x, y = st.pending_point
            x = int(np.clip(x, 0, w-1))
            y = int(np.clip(y, 0, h-1))
            print(f"[SERVER] CLICK -> reset, seed on current frame at ({x},{y})")

            # hard reset to avoid anchoring to an older first frame
            st.predictor.reset_state()
            st.predictor.load_first_frame(frame)

            labels = np.array([1], dtype=np.int32)
            points = np.array([[x, y]], dtype=np.float32)
            with torch.no_grad():
                cur_out, ids, logits = st.predictor.add_new_prompt(
                    frame_idx=0, obj_id=st.obj_id, points=points, labels=labels
                )

            # Convert initial logits to a binary mask (uint8 {0,255}) and send NOW
            try:
                mask = (logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
                mask = mask[..., 0]  # HxW
            except Exception as e:
                print(f"[SERVER] initial mask conversion error: {e}")
                mask = np.zeros((h, w), np.uint8)

            await ws.send(png_from_mask(mask))
            st.have_prompt = True
            st.pending_point = None

            # Skip tracking this frame (we already sent the seeded result)
            st.q.clear()
            continue

        # No prompt yet? Send empty mask to keep preview responsive
        if not st.have_prompt:
            empty = np.zeros((h, w), np.uint8)
            await ws.send(png_from_mask(empty))
            continue

        # Normal tracking on subsequent frames
        with torch.no_grad():
            out_ids, out_logits = st.predictor.track(frame)

        try:
            mask = (out_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
            mask = mask[..., 0]
        except Exception as e:
            print(f"[SERVER] tracking mask conversion error: {e}")
            mask = np.zeros((h, w), np.uint8)

        await ws.send(png_from_mask(mask))

async def handler(ws):
    print("Client connected")

    model_version = "sam2"  # or "sam2.1"
    ckpt = f"./checkpoints/{model_version}/{model_version}_hiera_small.pt"
    cfg  = f"{model_version}/{model_version}_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(cfg, ckpt)

    st = Session(predictor=predictor)
    task = asyncio.create_task(process_loop(ws, st))

    try: # maybe send jpeg and string to select point of that specific frame
        async for msg in ws:
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    print("[SERVER] bad JSON")
                    continue
                cmd = data.get("cmd", "")
                if cmd in ("init", "clear"):
                    print(f"[SERVER] control: {cmd} (will reinit on next frame)")
                    st.inited = False
                    st.have_prompt = False
                    st.frame_wh = None
                    st.pending_point = None
                elif cmd == "prompt":
                    x, y = int(data["x"]), int(data["y"])
                    print(f"[SERVER] recv prompt ({x},{y})")
                    st.pending_point = (x, y)
                # (no color handling, server sends mask only)
            else:
                st.q.append(msg)  # JPEG frame
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        task.cancel()
        try: await task
        except: pass

async def main():
    print(f"SAM2 WebSocket server: ws://{HOST}:{PORT}")
    torch.set_grad_enabled(False)
    async with websockets.serve(handler, HOST, PORT, max_size=8*1024*1024):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
