# sam2_ws_server_mask_only.py
import asyncio, json, time, traceback
import cv2, numpy as np, websockets, torch
from collections import deque
from dataclasses import dataclass, field
from sam2.build_sam import build_sam2_camera_predictor
import os

HOST, PORT = "0.0.0.0", 8765
DEBOUNCE_SECONDS = 0.35  # idle time after last click before auto-start tracking

# Optional: prefer math attention (broad compatibility)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    sdpa_kernel(SDPBackend.MATH).__enter__()
except Exception:
    pass


def decode_jpeg(buf: bytes):
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def png_from_mask(mask_u8: np.ndarray) -> bytes:
    ok, png = cv2.imencode(".png", mask_u8)
    return png.tobytes() if ok else b""


def tensor_union_logits_to_u8_mask(t, h, w) -> np.ndarray:
    """Accept [B,1,H,W] or [B,H,W] or [1,H,W] or [H,W] -> HxW uint8 union (0/255)."""
    if t is None:
        return np.zeros((h, w), np.uint8)
    if t.ndim == 2:
        t = t.unsqueeze(0)               # [1,H,W]
    if t.ndim == 4 and t.shape[1] == 1:
        t = t.squeeze(1)                 # [B,H,W]
    union = (t > 0.0).any(dim=0)         # [H,W]
    return union.detach().cpu().numpy().astype(np.uint8) * 255


def squeeze_video_masks(video_res_masks: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] -> [B,H,W] if needed."""
    if video_res_masks.ndim == 4 and video_res_masks.shape[1] == 1:
        return video_res_masks.squeeze(1)
    return video_res_masks


@dataclass
class Session:
    predictor: any
    q: deque = field(default_factory=lambda: deque(maxlen=1))

    # stream/predictor state
    inited: bool = False
    frame_wh: tuple | None = None

    # prompt staging (before tracking)
    pending_clicks: deque = field(default_factory=lambda: deque(maxlen=128))  # (x,y)
    last_click_time: float | None = None
    last_seed_union: np.ndarray | None = None  # latest seeded union preview (u8 HxW)
    next_obj_id: int = 1

    # tracking state
    tracking_started: bool = False
    last_obj_ids: list[int] | None = None
    last_video_masks: torch.Tensor | None = None  # [B,1,H,W] or [B,H,W]

    # re-anchor
    reanchor_click: tuple[int, int] | None = None  # set when user clicks during tracking


async def process_loop(ws, st: Session):
    while True:
        try:
            # Wait for a frame
            while not st.q:
                await asyncio.sleep(0.001)

            jpg = st.q.pop()
            frame = decode_jpeg(jpg)
            if frame is None:
                await ws.send(png_from_mask(np.zeros((360, 640), np.uint8)))
                continue

            h, w = frame.shape[:2]
            wh = (w, h)

            # Initialize predictor if first frame or resolution changed
            if (not st.inited) or (st.frame_wh != wh):
                print(f"[SERVER] init @ {wh}")
                st.predictor.load_first_frame(frame)  # sets internal state
                st.inited = True
                st.frame_wh = wh
                st.tracking_started = False
                st.pending_clicks.clear()
                st.last_click_time = None
                st.last_seed_union = None
                st.next_obj_id = 1
                st.last_obj_ids = None
                st.last_video_masks = None
                st.reanchor_click = None

            # ---------- synchronous RE-ANCHOR if requested ----------
            if st.reanchor_click is not None:
                rx, ry = st.reanchor_click
                st.reanchor_click = None  # consume the request

                # 1) Re-init with *current* frame as conditioning frame 0
                st.predictor.load_first_frame(frame)
                st.inited = True
                st.frame_wh = wh
                st.tracking_started = False
                st.last_seed_union = None

                # 2) Re-create existing objects via add_new_mask from last logits
                if st.last_obj_ids is not None and st.last_video_masks is not None:
                    try:
                        masks_logits = squeeze_video_masks(st.last_video_masks)  # [B,H,W]
                        for idx, oid in enumerate(st.last_obj_ids):
                            if idx >= masks_logits.shape[0]:
                                break
                            m = (masks_logits[idx] > 0.0).detach().cpu().numpy().astype(np.bool_)
                            with torch.no_grad():
                                st.predictor.add_new_mask(
                                    frame_idx=0,
                                    obj_id=int(oid),
                                    mask=m,
                                )
                        try:
                            max_old = max(int(oid) for oid in st.last_obj_ids)
                            st.next_obj_id = max(st.next_obj_id, max_old + 1)
                        except Exception:
                            pass
                    except Exception:
                        print("[SERVER] re-anchor: re-add old masks failed:")
                        traceback.print_exc()

                # 3) Add the NEW object as a click
                px = int(np.clip(rx, 0, w - 1)); py = int(np.clip(ry, 0, h - 1))
                new_id = st.next_obj_id
                st.next_obj_id += 1
                try:
                    with torch.no_grad():
                        _, obj_ids, video_res_masks = st.predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=int(new_id),
                            points=np.array([[px, py]], dtype=np.float32),
                            labels=np.array([1], dtype=np.int32),
                            bbox=None,
                            clear_old_points=True,
                            normalize_coords=True,
                        )
                    st.last_seed_union = tensor_union_logits_to_u8_mask(
                        squeeze_video_masks(video_res_masks), h, w
                    )
                    st.last_click_time = time.monotonic()
                    print(f"[SERVER] re-anchored; added new object {new_id} at ({px},{py}); total objs={len(obj_ids)}")
                except Exception:
                    print("[SERVER] re-anchor: add_new_prompt failed:")
                    traceback.print_exc()

                # 4) Immediately show the seeded union, then resume tracking next frames
                await ws.send(png_from_mask(st.last_seed_union if st.last_seed_union is not None
                                            else np.zeros((h, w), np.uint8)))
                st.tracking_started = True
                # Skip track() for this frame; continue loop to next frame
                st.q.clear()
                continue

            # ---------- multi-object seeding BEFORE tracking ----------
            while st.pending_clicks and not st.tracking_started:
                x, y = st.pending_clicks.popleft()
                x = int(np.clip(x, 0, w - 1))
                y = int(np.clip(y, 0, h - 1))
                obj_id = st.next_obj_id
                st.next_obj_id += 1
                try:
                    with torch.no_grad():
                        seed_frame_idx, obj_ids, video_res_masks = st.predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=int(obj_id),  # int ID per class API
                            points=np.array([[x, y]], dtype=np.float32),
                            labels=np.array([1], dtype=np.int32),
                            bbox=None,
                            clear_old_points=True,
                            normalize_coords=True,
                        )
                    st.last_seed_union = tensor_union_logits_to_u8_mask(
                        squeeze_video_masks(video_res_masks), h, w
                    )
                    st.last_click_time = time.monotonic()
                    print(f"[SERVER] added object {obj_id} at ({x},{y}); total objs={len(obj_ids)}")
                except Exception:
                    print("[SERVER] add_new_prompt (staging) failed:")
                    traceback.print_exc()

            # Auto-start tracking after short idle
            if not st.tracking_started:
                can_start = st.last_click_time is not None and \
                            (time.monotonic() - st.last_click_time) >= DEBOUNCE_SECONDS
                if can_start:
                    st.tracking_started = True
                    print("[SERVER] starting tracking")

            # While still staging (no tracking yet): show latest seeded preview
            if not st.tracking_started:
                if st.last_seed_union is not None:
                    await ws.send(png_from_mask(st.last_seed_union))
                else:
                    await ws.send(png_from_mask(np.zeros((h, w), np.uint8)))
                continue

            # ---------- Tracking mode ----------
            try:
                with torch.no_grad():
                    obj_ids, video_res_masks = st.predictor.track(frame)
                st.last_obj_ids = list(obj_ids) if isinstance(obj_ids, list) else obj_ids
                st.last_video_masks = video_res_masks  # keep logits (not binarized)
                mask = tensor_union_logits_to_u8_mask(
                    squeeze_video_masks(video_res_masks), h, w
                )
            except Exception as e:
                print(f"[SERVER] track() error: {e}")
                traceback.print_exc()
                mask = np.zeros((h, w), np.uint8)

            await ws.send(png_from_mask(mask))

        except Exception as e:
            print(f"[SERVER] process loop error: {e}")
            traceback.print_exc()
            # keep client alive
            try:
                await ws.send(png_from_mask(np.zeros((frame.shape[0], frame.shape[1]), np.uint8)))
            except:
                pass


async def handler(ws):
    print("Client connected")

    model_version = "sam2"  # or "sam2.1"
    ckpt = f"./checkpoints/{model_version}/{model_version}_hiera_tiny.pt"
    cfg  = f"{model_version}/{model_version}_hiera_t.yaml"
    predictor = build_sam2_camera_predictor(cfg, ckpt)

    st = Session(predictor=predictor)
    task = asyncio.create_task(process_loop(ws, st))

    #### camera capture ####
    frame_counter = 0

    video_base_folder = "videos"
    video_name = str(time.time())
    video_folder = os.path.join(video_base_folder, video_name)

    ########################

    try:
        async for msg in ws:
            if isinstance(msg, str):
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    print("[SERVER] bad JSON")
                    continue

                cmd = data.get("cmd", "")
                if cmd in ("init", "clear"):
                    print(f"[SERVER] control: {cmd} -> reinit on next frame")
                    st.inited = False
                    st.frame_wh = None
                    st.tracking_started = False
                    st.pending_clicks.clear()
                    st.last_click_time = None
                    st.last_seed_union = None
                    st.last_obj_ids = None
                    st.last_video_masks = None
                    st.next_obj_id = 1
                    st.reanchor_click = None

                elif cmd == "prompt":
                    x, y = int(data["x"]), int(data["y"])
                    print(f"[SERVER] recv prompt ({x},{y})")
                    if not st.tracking_started:
                        st.pending_clicks.append((x, y))
                    else:
                        # Set re-anchor request; handled synchronously on next frame
                        st.reanchor_click = (x, y)

                elif cmd == "start":
                    if not st.tracking_started:
                        st.tracking_started = True
                        print("[SERVER] manual start -> tracking")

                elif cmd == "intrinsics":
                    print(data)
            else:
                # JPEG frame
                st.q.append(msg)

                #### camera capture ####
                #save msg converted to png to file_path


                file_name = str(frame_counter) + ".png"
                file_path = os.path.join(video_folder, file_name)

                image = decode_jpeg(msg)
                cv2.imwrite(file_path, image)

                frame_counter += 1
                ########################



    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

        # -----------------------------------------------
        # 🔑 GPU Memory Cleanup Section - ADDED
        # -----------------------------------------------
        print("[SERVER] Starting GPU memory cleanup...")
        try:
            # 1. Clear internal state and tracking results
            st.predictor.reset_state()
            # 2. Explicitly delete references
            del st
            del predictor

            # 3. Force PyTorch to clear its cache and Python GC to run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("[SERVER] GPU memory successfully freed.")
        except Exception as e:
            print(f"[SERVER] Error during cleanup: {e}")
        # -----------------------------------------------

async def main():
    print(f"SAM2 WebSocket server: ws://{HOST}:{PORT}")
    torch.set_grad_enabled(False)
    async with websockets.serve(handler, HOST, PORT, max_size=8 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
