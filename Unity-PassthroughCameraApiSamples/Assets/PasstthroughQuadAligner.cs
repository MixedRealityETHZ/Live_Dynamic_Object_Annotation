/*using System;
using System.Collections;
using System.IO;
using System.Net.WebSockets;
using System.Text; // for UTF8 JSON
using System.Threading;
using System.Threading.Tasks;
using Meta.XR.Samples;
using PassthroughCameraSamples;
using UnityEngine;

public class PassthroughQuadAlignerWithSegmentation : MonoBehaviour
{
    [Header("Passthrough Setup")]
    [SerializeField] private WebCamTextureManager webCamTextureManager;
    [SerializeField] private GameObject cameraQuad;
    [SerializeField] private float quadDistance = 1.0f;

    [Header("WebSocket Streaming")]
    public string serverUrl = "ws://127.0.0.1:8765";
    public int targetFps = 30;
    [Range(1, 100)] public int jpgQuality = 75;

    // Hand / pinch config
    [Header("Pinch Clicks")]
    [SerializeField] private OVRHand leftHand;       // assign in Inspector
    [SerializeField] private OVRHand rightHand;      // assign in Inspector
    [SerializeField] private LayerMask raycastMask = ~0;  // default: Everything
    [SerializeField] private float raycastMaxDistance = 5f;

    // === Pointer visualization ===
    [Header("Pointer Viz")]
    [SerializeField] private bool showPointer = true;
    [SerializeField] private float rayWidth = 0.003f;
    [SerializeField] private float rayMaxDistance = 5f;   // can reuse raycastMaxDistance if you want
    [SerializeField] private Color rayColorIdle = new Color(0f, 1f, 1f, 0.9f);   // cyan
    [SerializeField] private Color rayColorHit = new Color(0f, 1f, 0f, 0.9f);   // green on hit
    [SerializeField] private float reticleSize = 0.01f;   // ~1 cm

    private LineRenderer lrLeft, lrRight;
    private GameObject reticleLeft, reticleRight;
    private Material rayMat, reticleMat;

    private ClientWebSocket ws;
    private CancellationTokenSource cts;
    private WaitForSeconds wait;

    private Texture2D encodeTex;
    private Color32[] pixelBuffer;
    private PassthroughCameraEye Eye => webCamTextureManager.Eye;

    private byte[] latestJpgFromServer;
    private readonly object jpgLock = new object();

    // send serialization and state
    private readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);
    private int frameId = 0;
    private bool wasPinchingL, wasPinchingR;

    private Renderer quadRenderer;
    private Material mat;

    

    // JSON message shape
    [Serializable]
    private class ClickMsg
    {
        public string cmd = "prompt";
        public int x;
        public int y;
*//*        public int frame;
        public float u;     // 0..1 (x)
        public float v;     // 0..1 (y; top-left origin)
        public int texW;
        public int texH;
        public float worldX, worldY, worldZ;
        public string hand; // "left" / "right"
        public string eye;  // passthrough eye
        public long t;      // unix ms*//*
    }

    private void Start()
    {
        if (webCamTextureManager == null || cameraQuad == null)
        {
            Debug.LogError("Missing WebCamTextureManager or cameraQuad reference.");
            enabled = false;
            return;
        }

        // Start camera feed
        var tex = webCamTextureManager.WebCamTexture;
        if (tex != null && !tex.isPlaying) tex.Play();

        // Ensure material is transparent (URP/Built-in safe fallback)
        quadRenderer = cameraQuad.GetComponent<Renderer>();
        var unlitTransparent = Shader.Find("Unlit/Transparent");
        if (unlitTransparent == null) unlitTransparent = Shader.Find("Sprites/Default"); // fallback
        mat = quadRenderer.material; // instance; we’ll clean it up on destroy
        mat.shader = unlitTransparent;
        mat.mainTexture = tex;

        // MeshCollider needed for textureCoord UVs
        if (cameraQuad.GetComponent<MeshCollider>() == null)
            Debug.LogWarning("cameraQuad needs a MeshCollider to get hit.textureCoord UVs.");

        ScaleQuadToCameraFOV();

        wait = new WaitForSeconds(1f / Mathf.Max(1, targetFps));
        encodeTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);

        // Build pointer visuals once
        if (showPointer)
        {
            // Materials that work in URP or Built-in
            var lineShader = Shader.Find("Sprites/Default"); // widely available & colorable
            if (lineShader == null) lineShader = Shader.Find("Unlit/Color");
            rayMat = new Material(lineShader);

            var retShader = Shader.Find("Sprites/Default");
            if (retShader == null) retShader = Shader.Find("Unlit/Color");
            reticleMat = new Material(retShader);
            reticleMat.color = new Color(1f, 1f, 1f, 0.85f);

            lrLeft = CreateRayGO("LeftPointerRay", rayColorIdle);
            lrRight = CreateRayGO("RightPointerRay", rayColorIdle);

            reticleLeft = CreateReticleGO("LeftReticle");
            reticleRight = CreateReticleGO("RightReticle");
        }

        _ = ConnectAndStartAsync();
        StartCoroutine(CaptureAndSendLoop());
    }

    private void LateUpdate()
    {
        var cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(Eye);
        cameraQuad.transform.SetPositionAndRotation(
            cameraPose.position + cameraPose.rotation * Vector3.forward * quadDistance,
            cameraPose.rotation
        );

        // Apply alpha-blended segmentation if we received a frame (NOTE: this allocates each frame; consider shader later)
        byte[] incoming = null;
        lock (jpgLock)
        {
            if (latestJpgFromServer != null)
            {
                incoming = latestJpgFromServer;
                latestJpgFromServer = null;
            }
        }

        if (incoming != null)
        {
            Texture2D maskTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            maskTex.LoadImage(incoming);

            Color[] pixels = maskTex.GetPixels();
            for (int i = 0; i < pixels.Length; i++)
            {
                float maskValue = pixels[i].r; // threshold or segmentation mask
                if (maskValue > 0.5f)
                    pixels[i] = new Color(0f, 0f, 1f, 0.7f); // semi-transparent blue
                else
                    pixels[i] = new Color(0f, 0f, 0f, 0f); // transparent
            }
            maskTex.SetPixels(pixels);
            maskTex.Apply();

            //cameraQuad.GetComponent<Renderer>().material.mainTexture = maskTex;
            mat.mainTexture = maskTex;
        }

        // Pinch edge detection
        bool pinchL = IsIndexPinching(leftHand);
        bool pinchR = IsIndexPinching(rightHand);

        if (pinchL && !wasPinchingL) HandlePinch(leftHand, "left");
        if (pinchR && !wasPinchingR) HandlePinch(rightHand, "right");

        wasPinchingL = pinchL;
        wasPinchingR = pinchR;

        // Update pointer rays & reticles
        if (showPointer)
        {
            UpdatePointerViz(leftHand, lrLeft, reticleLeft);
            UpdatePointerViz(rightHand, lrRight, reticleRight);
        }
    }

    private void ScaleQuadToCameraFOV()
    {
        var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(Eye);
        var resolution = intrinsics.Resolution;

        var leftRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(0, resolution.y / 2));
        var rightRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(resolution.x, resolution.y / 2));
        float horizontalFovDeg = Vector3.Angle(leftRay.direction, rightRay.direction);
        float horizontalFovRad = horizontalFovDeg * Mathf.Deg2Rad;

        float quadWidth = 2f * quadDistance * Mathf.Tan(horizontalFovRad / 2f);
        float aspect = (float)resolution.y / resolution.x;
        float quadHeight = quadWidth * aspect;

        cameraQuad.transform.localScale = new Vector3(quadWidth, quadHeight, 1f);
    }

    // --- WebSocket Logic ---
    private async Task ConnectAndStartAsync()
    {
        ws = new ClientWebSocket();
        cts = new CancellationTokenSource();
        try
        {
            await ws.ConnectAsync(new Uri(serverUrl), cts.Token);
            Debug.Log("WebSocket connected.");
            _ = ReceiveLoopAsync();
        }
        catch (Exception ex)
        {
            Debug.LogWarning("WS connect error: " + ex.Message);
        }
    }

    private IEnumerator CaptureAndSendLoop()
    {
        WebCamTexture cam = null;
        while (true)
        {
            cam = webCamTextureManager?.WebCamTexture;
            if (cam != null && cam.width > 16 && cam.height > 16) break;
            yield return null;
        }

        pixelBuffer = new Color32[cam.width * cam.height];
        encodeTex.Reinitialize(cam.width, cam.height, TextureFormat.RGBA32, false);

        while (true)
        {
            yield return wait;
            if (ws == null || ws.State != WebSocketState.Open) continue;
            if (cam == null || !cam.didUpdateThisFrame) continue;

            cam.GetPixels32(pixelBuffer);
            encodeTex.SetPixels32(pixelBuffer);
            encodeTex.Apply(false, false);

            // bump frameId to correlate with click JSON
            frameId++;

            var jpg = encodeTex.EncodeToJPG(jpgQuality);

            // serialize all sends (no concurrent SendAsync on ClientWebSocket)
            Task sendTask = SendBytesAsyncSafe(jpg, WebSocketMessageType.Binary);
            while (!sendTask.IsCompleted) yield return null;

            if (sendTask.IsFaulted)
                Debug.LogWarning("WS send error: " + sendTask.Exception?.GetBaseException().Message);
        }
    }

    private async Task ReceiveLoopAsync()
    {
        var buffer = new byte[2 * 1024 * 1024];
        var ms = new MemoryStream(2 * 1024 * 1024);
        try
        {
            while (ws != null && ws.State == WebSocketState.Open)
            {
                ms.SetLength(0);
                WebSocketReceiveResult result;
                do
                {
                    result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", cts.Token);
                        Debug.Log("WS closed by server.");
                        return;
                    }
                    ms.Write(buffer, 0, result.Count);
                }
                while (!result.EndOfMessage);

                lock (jpgLock) latestJpgFromServer = ms.ToArray();
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("WS receive error: " + ex.Message);
        }
    }

    private async void OnDestroy()
    {
        try
        {
            cts?.Cancel();
            if (ws != null)
            {
                if (ws.State == WebSocketState.Open)
                    await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Bye", CancellationToken.None);
                ws.Dispose();
            }
        }
        catch { }

        // Clean up pointer viz
        if (lrLeft != null) Destroy(lrLeft.gameObject);
        if (lrRight != null) Destroy(lrRight.gameObject);
        if (reticleLeft != null) Destroy(reticleLeft);
        if (reticleRight != null) Destroy(reticleRight);
        if (rayMat != null) Destroy(rayMat);
        if (reticleMat != null) Destroy(reticleMat);
    }

    // ======================
    // Pinch + Raycast
    // ======================

    private static bool IsIndexPinching(OVRHand h)
    {
        if (h == null) return false;
        return h.GetFingerIsPinching(OVRHand.HandFinger.Index) &&
               h.GetFingerConfidence(OVRHand.HandFinger.Index) == OVRHand.TrackingConfidence.High;
    }

    private void HandlePinch(OVRHand hand, string whichHand)
    {
        if (hand == null) return;

        var pose = hand.PointerPose; // a Transform that points where the user aims
        if (pose == null) return;

        Ray ray = new Ray(pose.position, pose.forward);
        if (Physics.Raycast(ray, out RaycastHit hit, raycastMaxDistance, raycastMask))
        {
            if (hit.collider != null && hit.collider.gameObject == cameraQuad)
            {
                // Requires MeshCollider on cameraQuad to populate textureCoord
                Vector2 uv = hit.textureCoord;

                var cam = webCamTextureManager?.WebCamTexture;
                int w = cam != null ? cam.width : 0;
                int h = cam != null ? cam.height : 0;

                SendClickJson(uv, hit.point, whichHand, w, h);
            }
        }
    }

    private async void SendClickJson(Vector2 uv, Vector3 world, string whichHand, int texW, int texH)
    {
        if (ws == null || ws.State != WebSocketState.Open) return;

        float u = uv.x;
        float v = 1f - uv.y;

        int px = (int)Math.Round(u * texW);
        int py = (int)Math.Round(v * texH);

        string type = "";
        if(whichHand == "left"){
            type = "clear";
        }else{
            type = "prompt";
        }
        var msg = new ClickMsg
        {
            cmd = type,
            x = px,
            y = py
            *//*            frame = frameId,
                        u = uv.x,
                        v = 1f - uv.y, // flip to top-left origin
                        texW = texW,
                        texH = texH,
                        worldX = world.x,
                        worldY = world.y,
                        worldZ = world.z,
                        hand = whichHand,
                        eye = Eye.ToString(),
                        t = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()*//*
        };
        
       
        string json = JsonUtility.ToJson(msg);
        byte[] bytes = Encoding.UTF8.GetBytes(json);

        try
        {
            await SendBytesAsyncSafe(bytes, WebSocketMessageType.Text);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("WS click send error: " + ex.Message);
        }
    }

    // serialize sends for JPEG and JSON
    private async Task SendBytesAsyncSafe(byte[] bytes, WebSocketMessageType type)
    {
        if (ws == null || ws.State != WebSocketState.Open) return;
        await sendLock.WaitAsync(cts.Token);
        try
        {
            await ws.SendAsync(new ArraySegment<byte>(bytes), type, true, cts.Token);
        }
        finally
        {
            sendLock.Release();
        }
    }

    // ======================
    // Pointer Viz helpers
    // ======================

    private LineRenderer CreateRayGO(string name, Color col)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        var lr = go.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.useWorldSpace = true;
        lr.material = rayMat; // safe material
        lr.startWidth = lr.endWidth = rayWidth;
        lr.numCapVertices = 4;
        lr.numCornerVertices = 4;
        lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lr.receiveShadows = false;
        lr.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
        lr.textureMode = LineTextureMode.Stretch;
        lr.enabled = true;
        lr.startColor = lr.endColor = col;
        return lr;
    }

    private GameObject CreateReticleGO(string name)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name;
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col); // no collider
        go.layer = LayerMask.NameToLayer("Ignore Raycast"); // avoid self-hit
        var mr = go.GetComponent<MeshRenderer>();
        mr.material = reticleMat; // URP-safe material
        go.transform.localScale = Vector3.one * reticleSize;
        return go;
    }

    private void UpdatePointerViz(OVRHand hand, LineRenderer lr, GameObject reticle)
    {
        if (!showPointer || hand == null || lr == null || reticle == null)
            return;

        var pose = hand.PointerPose;
        if (pose == null)
        {
            lr.enabled = false;
            reticle.SetActive(false);
            return;
        }

        var origin = pose.position;
        var dir = pose.forward;

        bool hitSomething = Physics.Raycast(
            new Ray(origin, dir),
            out RaycastHit hit,
            rayMaxDistance,
            raycastMask
        );

        Vector3 end = hitSomething ? hit.point : origin + dir * rayMaxDistance;

        // Set line
        lr.enabled = true;
        lr.SetPosition(0, origin);
        lr.SetPosition(1, end);
        var targetCol = (hitSomething && hit.collider != null && hit.collider.gameObject == cameraQuad) ? rayColorHit : rayColorIdle;
        lr.startColor = lr.endColor = targetCol;

        // Place reticle
        reticle.SetActive(true);
        reticle.transform.position = end;

        // Face the main camera so it stays visible
        var cam = Camera.main;
        if (cam != null)
            reticle.transform.rotation = Quaternion.LookRotation(reticle.transform.position - cam.transform.position);

        // Optional: tint reticle on true cameraQuad hit
        var mr = reticle.GetComponent<MeshRenderer>();
        if (mr != null) mr.material.color = (targetCol == rayColorHit) ? new Color(0f, 1f, 0f, 0.9f) : new Color(1f, 1f, 1f, 0.6f);
    }
}*/

using System;
using System.Collections;
using System.IO;
using System.Net.WebSockets;
using System.Text; // for UTF8 JSON
using System.Threading;
using System.Threading.Tasks;
using Meta.XR.Samples;
using PassthroughCameraSamples;
using UnityEngine;

public class PassthroughQuadAlignerWithSegmentation : MonoBehaviour
{
    [Header("Passthrough Setup")]
    [SerializeField] private WebCamTextureManager webCamTextureManager;
    [SerializeField] private GameObject cameraQuad;
    [SerializeField] private float quadDistance = 1.0f;

    [Header("WebSocket Streaming")]
    public string serverUrl = "ws://127.0.0.1:8765";
    public int targetFps = 30;
    [Range(1, 100)] public int jpgQuality = 75;

    // Hand / pinch config
    [Header("Pinch Clicks")]
    [SerializeField] private OVRHand leftHand;       // assign in Inspector
    [SerializeField] private OVRHand rightHand;      // assign in Inspector
    [SerializeField] private LayerMask raycastMask = ~0;  // default: Everything
    [SerializeField] private float raycastMaxDistance = 5f;

    // === Pointer visualization ===
    [Header("Pointer Viz")]
    [SerializeField] private bool showPointer = true;
    [SerializeField] private float rayWidth = 0.003f;
    [SerializeField] private float rayMaxDistance = 5f;   // can reuse raycastMaxDistance if you want
    [SerializeField] private Color rayColorIdle = new Color(0f, 1f, 1f, 0.9f);   // cyan
    [SerializeField] private Color rayColorHit = new Color(0f, 1f, 0f, 0.9f);   // green on hit
    [SerializeField] private float reticleSize = 0.01f;   // ~1 cm

    private LineRenderer lrLeft, lrRight;
    private GameObject reticleLeft, reticleRight;
    private Material rayMat, reticleMat;

    private ClientWebSocket ws;
    private CancellationTokenSource cts;
    private WaitForSeconds wait;

    private Texture2D encodeTex;
    private Color32[] pixelBuffer;
    private PassthroughCameraEye Eye => webCamTextureManager.Eye;

    private byte[] latestJpgFromServer;
    private readonly object jpgLock = new object();

    // send serialization and state
    private readonly SemaphoreSlim sendLock = new SemaphoreSlim(1, 1);
    private int frameId = 0;
    private bool wasPinchingL, wasPinchingR;

    private Renderer quadRenderer;
    private Material mat;

    private Texture2D maskTex;
    private Color[] maskBuffer;

    //private int counterFrame = 0;

    // JSON message shape
    [Serializable]

    
    private class ClickMsg
    {
        public string cmd = "prompt";
        public int x;
        public int y;
        public float focalx;
        public float focaly;
        public float principalx;
        public float principaly;
        /*public int frame;
             /*   public float u;     // 0..1 (x)
                public float v;     // 0..1 (y; top-left origin)
                public int texW;
                public int texH;
                public float worldX, worldY, worldZ;
                public string hand; // "left" / "right"
                public string eye;  // passthrough eye
                public long t;      // unix ms*/
    }

    private void Start()
    {
        if (webCamTextureManager == null || cameraQuad == null)
        {
            Debug.LogError("Missing WebCamTextureManager or cameraQuad reference.");
            enabled = false;
            return;
        }

        // Start camera feed
        var tex = webCamTextureManager.WebCamTexture;
        if (tex != null && !tex.isPlaying) tex.Play();

        // Ensure material is transparent (URP/Built-in safe fallback)
        quadRenderer = cameraQuad.GetComponent<Renderer>();
        var unlitTransparent = Shader.Find("Unlit/Transparent");
        if (unlitTransparent == null) unlitTransparent = Shader.Find("Sprites/Default"); // fallback
        mat = quadRenderer.material; // instance; we’ll clean it up on destroy
        mat.shader = unlitTransparent;
        mat.mainTexture = tex;

        // MeshCollider needed for textureCoord UVs
        if (cameraQuad.GetComponent<MeshCollider>() == null)
            Debug.LogWarning("cameraQuad needs a MeshCollider to get hit.textureCoord UVs.");

        ScaleQuadToCameraFOV();

        wait = new WaitForSeconds(1f / Mathf.Max(1, targetFps));
        encodeTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);

        // Build pointer visuals once
        if (showPointer)
        {
            // Materials that work in URP or Built-in
            var lineShader = Shader.Find("Sprites/Default"); // widely available & colorable
            if (lineShader == null) lineShader = Shader.Find("Unlit/Color");
            rayMat = new Material(lineShader);

            var retShader = Shader.Find("Sprites/Default");
            if (retShader == null) retShader = Shader.Find("Unlit/Color");
            reticleMat = new Material(retShader);
            reticleMat.color = new Color(1f, 1f, 1f, 0.85f);

            lrLeft = CreateRayGO("LeftPointerRay", rayColorIdle);
            lrRight = CreateRayGO("RightPointerRay", rayColorIdle);

            reticleLeft = CreateReticleGO("LeftReticle");
            reticleRight = CreateReticleGO("RightReticle");
        }


        _ = ConnectAndStartAsync();
        StartCoroutine(CaptureAndSendLoop());
    }

    /*private void LateUpdate()
    {
        var cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(Eye);
        cameraQuad.transform.SetPositionAndRotation(
            cameraPose.position + cameraPose.rotation * Vector3.forward * quadDistance,
            cameraPose.rotation
        );

        // Apply alpha-blended segmentation if we received a frame (NOTE: this allocates each frame; consider shader later)
        byte[] incoming = null;
        lock (jpgLock)
        {
            if (latestJpgFromServer != null)
            {
                incoming = latestJpgFromServer;
                latestJpgFromServer = null;
            }
        }

        if (incoming != null)
        {
            Texture2D maskTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            maskTex.LoadImage(incoming);

            Color[] pixels = maskTex.GetPixels();
            for (int i = 0; i < pixels.Length; i++)
            {
                float maskValue = pixels[i].r; // threshold or segmentation mask
                if (maskValue > 0.5f)
                    pixels[i] = new Color(0f, 0f, 1f, 0.7f); // semi-transparent blue
                else
                    pixels[i] = new Color(0f, 0f, 0f, 0f); // transparent
            }
            maskTex.SetPixels(pixels);
            maskTex.Apply();

            //cameraQuad.GetComponent<Renderer>().material.mainTexture = maskTex;
            mat.mainTexture = maskTex;
        }

        // Pinch edge detection
        bool pinchL = IsIndexPinching(leftHand);
        bool pinchR = IsIndexPinching(rightHand);

        if (pinchL && !wasPinchingL) HandlePinch(leftHand, "left");
        if (pinchR && !wasPinchingR) HandlePinch(rightHand, "right");

        wasPinchingL = pinchL;
        wasPinchingR = pinchR;

        // Update pointer rays & reticles
        if (showPointer)
        {
            UpdatePointerViz(leftHand, lrLeft, reticleLeft);
            UpdatePointerViz(rightHand, lrRight, reticleRight);
        }
    }*/

    private void LateUpdate()
    {
        // Position quad in front of passthrough camera
        var cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(Eye);
        cameraQuad.transform.SetPositionAndRotation(
            cameraPose.position + cameraPose.rotation * Vector3.forward * quadDistance,
            cameraPose.rotation
        );

        // Apply alpha-blended segmentation if we received a frame
        byte[] incoming = null;
        lock (jpgLock)
        {
            if (latestJpgFromServer != null)
            {
                incoming = latestJpgFromServer;
                latestJpgFromServer = null;
            }
        }

        if (incoming != null)
        {
            // Create the mask texture once and reuse it
            if (maskTex == null)
            {
                maskTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            }

            if (!maskTex.LoadImage(incoming))
            {
                Debug.LogWarning("Failed to load mask image from incoming bytes.");
            }
            else
            {
                // Work with Color[] like before
                Color[] pixels = maskTex.GetPixels();
                for (int i = 0; i < pixels.Length; i++)
                {
                    float maskValue = pixels[i].r; // red channel as mask 0..1
                    if (maskValue > 0.5f)
                        pixels[i] = new Color(0f, 0f, 1f, 0.7f); // semi-transparent blue
                    else
                        pixels[i] = new Color(0f, 0f, 0f, 0f);    // transparent
                }

                maskTex.SetPixels(pixels);
                maskTex.Apply(false, false);

                if (mat != null)
                    mat.mainTexture = maskTex;
            }
        }

        // Pinch edge detection
        bool pinchL = IsIndexPinching(leftHand);
        bool pinchR = IsIndexPinching(rightHand);

        if (pinchL && !wasPinchingL) HandlePinch(leftHand, "left");
        if (pinchR && !wasPinchingR) HandlePinch(rightHand, "right");

        wasPinchingL = pinchL;
        wasPinchingR = pinchR;

        // Update pointer rays & reticles
        if (showPointer)
        {
            UpdatePointerViz(leftHand, lrLeft, reticleLeft);
            UpdatePointerViz(rightHand, lrRight, reticleRight);
        }
    }

    private void ScaleQuadToCameraFOV()
    {
        var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(Eye);
        var resolution = intrinsics.Resolution;

        var leftRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(0, resolution.y / 2));
        var rightRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(resolution.x, resolution.y / 2));
        float horizontalFovDeg = Vector3.Angle(leftRay.direction, rightRay.direction);
        float horizontalFovRad = horizontalFovDeg * Mathf.Deg2Rad;

        float quadWidth = 2f * quadDistance * Mathf.Tan(horizontalFovRad / 2f);
        float aspect = (float)resolution.y / resolution.x;
        float quadHeight = quadWidth * aspect;

        cameraQuad.transform.localScale = new Vector3(quadWidth, quadHeight, 1f);
    }

    // --- WebSocket Logic ---
    private async Task ConnectAndStartAsync()
    {
        ws = new ClientWebSocket();
        cts = new CancellationTokenSource();
        try
        {
            await ws.ConnectAsync(new Uri(serverUrl), cts.Token);
            Debug.Log("WebSocket connected.");
            _ = ReceiveLoopAsync();
        }
        catch (Exception ex)
        {
            Debug.LogWarning("WS connect error: " + ex.Message);
        }
    }

    private IEnumerator CaptureAndSendLoop()
    {
        WebCamTexture cam = null;
        while (true)
        {
            cam = webCamTextureManager?.WebCamTexture;
            if (cam != null && cam.width > 16 && cam.height > 16) break;
            yield return null;
        }

        pixelBuffer = new Color32[cam.width * cam.height];
        encodeTex.Reinitialize(cam.width, cam.height, TextureFormat.RGBA32, false);

        while (true)
        {
            yield return wait;
            if (ws == null || ws.State != WebSocketState.Open) continue;
            if (cam == null || !cam.didUpdateThisFrame) continue;

            cam.GetPixels32(pixelBuffer);
            encodeTex.SetPixels32(pixelBuffer);
            encodeTex.Apply(false, false);

            // bump frameId to correlate with click JSON
            frameId++;

            var jpg = encodeTex.EncodeToJPG(jpgQuality);

            // serialize all sends (no concurrent SendAsync on ClientWebSocket)
            Task sendTask = SendBytesAsyncSafe(jpg, WebSocketMessageType.Binary);
            while (!sendTask.IsCompleted) yield return null;

            if (sendTask.IsFaulted)
                Debug.LogWarning("WS send error: " + sendTask.Exception?.GetBaseException().Message);
        }
    }

    private async Task ReceiveLoopAsync()
    {
        var buffer = new byte[2 * 1024 * 1024];
        var ms = new MemoryStream(2 * 1024 * 1024);
        try
        {
            while (ws != null && ws.State == WebSocketState.Open)
            {
                ms.SetLength(0);
                WebSocketReceiveResult result;
                do
                {
                    result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", cts.Token);
                        Debug.Log("WS closed by server.");
                        return;
                    }
                    ms.Write(buffer, 0, result.Count);
                }
                while (!result.EndOfMessage);

                lock (jpgLock) latestJpgFromServer = ms.ToArray();
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("WS receive error: " + ex.Message);
        }
    }

    private async void OnDestroy()
    {
        try
        {
            cts?.Cancel();
            if (ws != null)
            {
                if (ws.State == WebSocketState.Open)
                    await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Bye", CancellationToken.None);
                ws.Dispose();
            }
        }
        catch { }

        if (webCamTextureManager != null && webCamTextureManager.WebCamTexture != null)
        {
            var camTex = webCamTextureManager.WebCamTexture;
            if (camTex.isPlaying) camTex.Stop();
        }

        if (encodeTex != null) Destroy(encodeTex);
        if (maskTex != null) Destroy(maskTex);
        if (mat != null) Destroy(mat);

        // Clean up pointer viz
        if (lrLeft != null) Destroy(lrLeft.gameObject);
        if (lrRight != null) Destroy(lrRight.gameObject);
        if (reticleLeft != null) Destroy(reticleLeft);
        if (reticleRight != null) Destroy(reticleRight);
        if (rayMat != null) Destroy(rayMat);
        if (reticleMat != null) Destroy(reticleMat);
    }

    // ======================
    // Pinch + Raycast
    // ======================

    private static bool IsIndexPinching(OVRHand h)
    {
        if (h == null) return false;
        return h.GetFingerIsPinching(OVRHand.HandFinger.Index) &&
               h.GetFingerConfidence(OVRHand.HandFinger.Index) == OVRHand.TrackingConfidence.High;
    }

    private void HandlePinch(OVRHand hand, string whichHand)
    {
        if (hand == null) return;

        var pose = hand.PointerPose; // a Transform that points where the user aims
        if (pose == null) return;

        Ray ray = new Ray(pose.position, pose.forward);
        if (Physics.Raycast(ray, out RaycastHit hit, raycastMaxDistance, raycastMask))
        {
            if (hit.collider != null && hit.collider.gameObject == cameraQuad)
            {
                // Requires MeshCollider on cameraQuad to populate textureCoord
                Vector2 uv = hit.textureCoord;

                var cam = webCamTextureManager?.WebCamTexture;
                int w = cam != null ? cam.width : 0;
                int h = cam != null ? cam.height : 0;

                SendClickJson(uv, hit.point, whichHand, w, h);
            }
        }
    }

    private async void SendClickJson(Vector2 uv, Vector3 world, string whichHand, int texW, int texH)
    {
        if (ws == null || ws.State != WebSocketState.Open) return;

        float u = uv.x;
        float v = 1f - uv.y;

        int px = (int)Math.Round(u * texW);
        int py = (int)Math.Round(v * texH);

        string type = "";
        if (whichHand == "left")
        {
            type = "clear";
        }
        else
        {
            type = "prompt";
        }

        //send intrinsics
        var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(Eye);
        Vector2 focalLength = intrinsics.FocalLength;
        Vector2 principalPoint = intrinsics.PrincipalPoint;

        var msg = new ClickMsg
        {
            cmd = type,
            x = px,
            y = py
            /*cmd = "intrinsics",
            focalx = focalLength.x,
            focaly = focalLength.y,
            principalx = principalPoint.x,
            principaly = principalPoint.y*/
            /*            frame = frameId,
                        u = uv.x,
                        v = 1f - uv.y, // flip to top-left origin
                        texW = texW,
                        texH = texH,
                        worldX = world.x,
                        worldY = world.y,
                        worldZ = world.z,
                        hand = whichHand,
                        eye = Eye.ToString(),
                        t = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()*/
        };


        string json = JsonUtility.ToJson(msg);
        byte[] bytes = Encoding.UTF8.GetBytes(json);

        try
        {
            await SendBytesAsyncSafe(bytes, WebSocketMessageType.Text);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("WS click send error: " + ex.Message);
        }
    }

    // serialize sends for JPEG and JSON
    private async Task SendBytesAsyncSafe(byte[] bytes, WebSocketMessageType type)
    {
        if (ws == null || ws.State != WebSocketState.Open) return;
        await sendLock.WaitAsync(cts.Token);
        try
        {
            await ws.SendAsync(new ArraySegment<byte>(bytes), type, true, cts.Token);
        }
        finally
        {
            sendLock.Release();
        }
    }

    // ======================
    // Pointer Viz helpers
    // ======================

    private LineRenderer CreateRayGO(string name, Color col)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        var lr = go.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.useWorldSpace = true;
        lr.material = rayMat; // safe material
        lr.startWidth = lr.endWidth = rayWidth;
        lr.numCapVertices = 4;
        lr.numCornerVertices = 4;
        lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lr.receiveShadows = false;
        lr.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
        lr.textureMode = LineTextureMode.Stretch;
        lr.enabled = true;
        lr.startColor = lr.endColor = col;
        return lr;
    }

    private GameObject CreateReticleGO(string name)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name;
        var col = go.GetComponent<Collider>();
        if (col != null) Destroy(col); // no collider
        go.layer = LayerMask.NameToLayer("Ignore Raycast"); // avoid self-hit
        var mr = go.GetComponent<MeshRenderer>();
        mr.material = reticleMat; // URP-safe material
        go.transform.localScale = Vector3.one * reticleSize;
        return go;
    }

    private void UpdatePointerViz(OVRHand hand, LineRenderer lr, GameObject reticle)
    {
        if (!showPointer || hand == null || lr == null || reticle == null)
            return;

        var pose = hand.PointerPose;
        if (pose == null)
        {
            lr.enabled = false;
            reticle.SetActive(false);
            return;
        }

        var origin = pose.position;
        var dir = pose.forward;

        bool hitSomething = Physics.Raycast(
            new Ray(origin, dir),
            out RaycastHit hit,
            rayMaxDistance,
            raycastMask
        );

        Vector3 end = hitSomething ? hit.point : origin + dir * rayMaxDistance;

        // Set line
        lr.enabled = true;
        lr.SetPosition(0, origin);
        lr.SetPosition(1, end);
        var targetCol = (hitSomething && hit.collider != null && hit.collider.gameObject == cameraQuad) ? rayColorHit : rayColorIdle;
        lr.startColor = lr.endColor = targetCol;

        // Place reticle
        reticle.SetActive(true);
        reticle.transform.position = end;

        // Face the main camera so it stays visible
        var cam = Camera.main;
        if (cam != null)
            reticle.transform.rotation = Quaternion.LookRotation(reticle.transform.position - cam.transform.position);

        // Optional: tint reticle on true cameraQuad hit
        var mr = reticle.GetComponent<MeshRenderer>();
        if (mr != null) mr.material.color = (targetCol == rayColorHit) ? new Color(0f, 1f, 0f, 0.9f) : new Color(1f, 1f, 1f, 0.6f);
    }
}