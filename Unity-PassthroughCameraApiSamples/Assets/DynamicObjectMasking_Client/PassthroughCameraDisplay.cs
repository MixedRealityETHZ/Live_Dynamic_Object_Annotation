/*using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.WebSockets;
using PassthroughCameraSamples;
using UnityEngine;

public class PassthroughCameraDisplay : MonoBehaviour
{
    public WebCamTextureManager webcamManager;
    public Renderer quadRenderer;

    // Start is called before the first frame update
    void Start()
    {
        if (webcamManager != null && quadRenderer != null)
        {
            // Make sure the quad uses a shader compatible with the camera texture
            //quadRenderer.material.shader = Shader.Find("Unlit/Texture");

            // Assign the passthrough camera feed
            //quadRenderer.material.mainTexture = webcamManager.WebCamTexture;
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (webcamManager.WebCamTexture != null)
        {
            quadRenderer.material.mainTexture = webcamManager.WebCamTexture;
        }

    }
}*/


// PassthroughWebSocketStreamer.cs
// Streams JPEG frames from WebCamTextureManager.WebCamTexture to your Python server
// Uses System.Net.WebSockets (not supported on WebGL).
/*using UnityEngine;
using System;
using System.IO;
using System.Net.WebSockets;
using System.Collections;
using System.Threading;
using System.Threading.Tasks;
using PassthroughCameraSamples; // <- as shown in your snippet

public class PassthroughWebSocketStreamer : MonoBehaviour
{
    [Header("Server")]
    public string serverUrl = "ws://127.0.0.1:8765";

    [Header("Passthrough source")]
    public WebCamTextureManager webcamManager;   // assign your prefab instance in Inspector

    [Header("Capture")]
    public int targetFps = 30;                   // throttle sending to this FPS
    [Range(1, 100)] public int jpgQuality = 75;

    [Header("Preview (optional)")]
    public Renderer previewRenderer;             // show frames echoed back by the server

    // internals
    ClientWebSocket ws;
    CancellationTokenSource cts;

    Texture2D encodeTex;     // CPU-side encoder texture
    Texture2D previewTex;    // shows server-returned JPEG
    Color32[] pixelBuffer;
    WaitForSeconds wait;

    volatile byte[] latestJpgFromServer;
    readonly object jpgLock = new object();

    void Start()
    {

        //PlaceQuad();

        if (webcamManager == null)
        {
            Debug.LogError("PassthroughWebSocketStreamer: webcamManager is not assigned.");
            enabled = false;
            return;
        }

        // Passthrough prefab should be starting/owning its WebCamTexture internally.
        // We'll wait until it reports a valid size.
        wait = new WaitForSeconds(1f / Mathf.Max(1, targetFps));
        encodeTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
        previewTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);

        _ = ConnectAndStartAsync();
        StartCoroutine(CaptureAndSendLoop());
    }

    async Task ConnectAndStartAsync()
    {
        ws = new ClientWebSocket();
        cts = new CancellationTokenSource();
        try
        {
            await ws.ConnectAsync(new Uri(serverUrl), cts.Token);
            Debug.Log("WS connected.");
            _ = ReceiveLoopAsync();
        }
        catch (Exception ex)
        {
            Debug.LogWarning("WS connect error: " + ex.Message);
        }
    }

    IEnumerator CaptureAndSendLoop()
    {
        // Wait until the prefab exposes a valid webcam texture with dimensions
        WebCamTexture cam = null;
        while (true)
        {
            cam = webcamManager != null ? webcamManager.WebCamTexture : null;
            if (cam != null && cam.width > 16 && cam.height > 16) break;
            yield return null;
        }

        // Allocate once based on source dimensions
        pixelBuffer = new Color32[cam.width * cam.height];
        encodeTex.Reinitialize(cam.width, cam.height, TextureFormat.RGBA32, false);

        while (true)
        {
            yield return wait;

            if (ws == null || ws.State != WebSocketState.Open) continue;
            if (cam == null) continue;
            if (!cam.didUpdateThisFrame) continue;

            // Copy pixels from passthrough frame
            cam.GetPixels32(pixelBuffer);
            encodeTex.SetPixels32(pixelBuffer);
            encodeTex.Apply(false, false);

            // JPEG encode & send
            var jpg = encodeTex.EncodeToJPG(jpgQuality);
            var seg = new ArraySegment<byte>(jpg);

            Task sendTask = null;
            try
            {
                sendTask = ws.SendAsync(seg, WebSocketMessageType.Binary, true, cts.Token);
            }
            catch (Exception ex)
            {
                Debug.LogWarning("WS send start error: " + ex.Message);
                continue;
            }
            // don’t block the main thread; just spin until the task completes
            while (!sendTask.IsCompleted) yield return null;

            if (sendTask.IsFaulted)
                Debug.LogWarning("WS send error: " + sendTask.Exception?.GetBaseException().Message);
        }
    }

    async Task ReceiveLoopAsync()
    {
        var buffer = new byte[2 * 1024 * 1024]; // chunk size (server allows up to 8MB per message)
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
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Debug.LogWarning("WS receive error: " + ex.Message);
        }
    }

    void Update()
    {
        // Update preview with the most recent returned frame (optional)
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
            //previewTex.LoadImage(incoming, false);

            int width = previewTex.width;
            int height = previewTex.height;

            //// Create a new texture with RGBA32 format
            //Texture2D maskTex = new Texture2D(width, height, TextureFormat.RGBA32, false);



            //// Example: create a simple bitmask for testing (all 1s in top half, all 0s in bottom half)
            //for (int y = 0; y < height; y++)
            //{
            //    for (int x = 0; x < width; x++)
            //    {
            //        // For demo: top half = 1, bottom half = 0
            //        float maskValue = (y > height / 2) ? 1f : 0f;

            //        Color pixelColor;
            //        if (maskValue > 0.5f)
            //            pixelColor = new Color(1f, 0f, 0f, 0.5f); // semi-transparent red
            //        else
            //            pixelColor = new Color(0f, 0f, 0f, 0f);   // fully transparent

            //        maskTex.SetPixel(x, y, pixelColor);
            //    }
            //}

            //// Apply changes to the texture
            //maskTex.Apply();

            Texture2D maskTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            maskTex.LoadImage(incoming); // automatically resizes to image dimension

            Color[] pixels = maskTex.GetPixels(); // returns Color[] with r=g=b in 0..1
            for (int i = 0; i < pixels.Length; i++)
            {
                float maskValue = pixels[i].r; // r = 0..1
                if (maskValue > 0.5f) // threshold, adjust if needed
                    pixels[i] = new Color(1f, 0f, 0f, 0.5f); // semi-transparent red
                else
                    pixels[i] = new Color(0f, 1f, 0f, 0.5f);   // fully transparent
            }

            maskTex.SetPixels(pixels);
            maskTex.Apply();


            if (previewRenderer != null)
                //previewRenderer.material.mainTexture = previewTex;
                previewRenderer.material.mainTexture = maskTex;
        }
    }

    async void OnDestroy()
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
    }

    public void PlaceQuad()
    {
        Transform quadTransform = previewRenderer.transform;

        Pose cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(PassthroughCameraEye.Left);

        Vector2Int resolution = PassthroughCameraUtils
            .GetCameraIntrinsics(PassthroughCameraEye.Left)
            .Resolution;

        int quadDistance = 1;

        quadTransform.position = cameraPose.position + cameraPose.forward * quadDistance;
        quadTransform.rotation = cameraPose.rotation;

        Ray leftSide = PassthroughCameraUtils.ScreenPointToRayInCamera(
            PassthroughCameraEye.Left,
            new Vector2Int(0, resolution.y / 2)
        );
        Ray rightSide = PassthroughCameraUtils.ScreenPointToRayInCamera(
            PassthroughCameraEye.Left,
            new Vector2Int(resolution.x, resolution.y / 2)
        );

        float horizontalFov = Vector3.Angle(leftSide.direction, rightSide.direction);

        float quadScale = 2 * quadDistance * Mathf.Tan((horizontalFov * Mathf.Deg2Rad) / 2);

        float ratio = (float)1280 / (float)960;

        quadTransform.localScale = new Vector3(quadScale, quadScale * ratio, 1);
    }
}

*/