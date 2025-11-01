/*

using System.Collections;
using Meta.XR.Samples;
using PassthroughCameraSamples;
using UnityEngine;

public class PassthroughQuadAligner : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager webCamTextureManager;
    [SerializeField] private GameObject cameraQuad;        // The Quad that displays the camera feed
    [SerializeField] private float quadDistance = 1.0f;    // Distance in meters in front of the passthrough camera

    private PassthroughCameraEye Eye => webCamTextureManager != null ? webCamTextureManager.Eye : default;

    private IEnumerator Start()
    {
        // Defensive checks
        if (webCamTextureManager == null)
        {
            Debug.LogError("PassthroughQuadAligner: WebCamTextureManager is not assigned.");
            enabled = false;
            yield break;
        }

        if (cameraQuad == null)
        {
            Debug.LogError("PassthroughQuadAligner: cameraQuad is not assigned.");
            enabled = false;
            yield break;
        }

        // Wait until camera permission is explicitly granted (HasCameraPermission is nullable bool)
        while (PassthroughCameraPermissions.HasCameraPermission != true)
            yield return null;

        // Ensure the WebCamTextureManager is enabled (it may be disabled by default until permission is granted)
        if (!webCamTextureManager.enabled)
            webCamTextureManager.enabled = true;

        var renderer = cameraQuad.GetComponent<Renderer>();
        if (renderer == null)
        {
            Debug.LogError("PassthroughQuadAligner: cameraQuad has no Renderer component.");
            enabled = false;
            yield break;
        }

        // Ensure we have a proper unlit material
        if (renderer.material == null)
            renderer.material = new Material(Shader.Find("Unlit/Texture"));
        else if (renderer.material.shader == null)
            renderer.material.shader = Shader.Find("Unlit/Texture");

        var tex = webCamTextureManager.WebCamTexture;
        if (tex != null && !tex.isPlaying)
            tex.Play();

        renderer.material.mainTexture = tex;

        // Scale the quad based on camera FOV
        ScaleQuadToCameraFOV();
    }

    private void LateUpdate()
    {
        // If manager or quad missing, skip
        if (webCamTextureManager == null || cameraQuad == null)
            return;

        // Use the calibrated passthrough camera pose
        var cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(Eye);

        // Position the quad in front of the passthrough camera (respecting calibrated pose)
        cameraQuad.transform.position = cameraPose.position + cameraPose.rotation * Vector3.forward * quadDistance;
        cameraQuad.transform.rotation = cameraPose.rotation;

        // Keep the texture assigned in case it was recreated or restarted elsewhere
        var renderer = cameraQuad.GetComponent<Renderer>();
        var tex = webCamTextureManager.WebCamTexture;
        if (renderer != null && tex != null)
            renderer.material.mainTexture = tex;
    }

    private void ScaleQuadToCameraFOV()
    {
        if (webCamTextureManager == null || cameraQuad == null)
            return;

        var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(Eye);
        var resolution = intrinsics.Resolution;

        // Compute horizontal FOV from the camera intrinsics
        var leftRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(0, resolution.y / 2));
        var rightRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(resolution.x, resolution.y / 2));
        float horizontalFovDeg = Vector3.Angle(leftRay.direction, rightRay.direction);
        float horizontalFovRad = horizontalFovDeg * Mathf.Deg2Rad;

        // Compute width and height based on FOV and resolution
        float quadWidth = 2f * quadDistance * Mathf.Tan(horizontalFovRad / 2f);
        float aspect = (float)resolution.x / resolution.y; // Correct aspect ratio
        float quadHeight = quadWidth / aspect;

        cameraQuad.transform.localScale = new Vector3(quadWidth, quadHeight, 1f);
    }
}
*/

using System;
using System.Collections;
using System.IO;
using System.Net.WebSockets;
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

    private ClientWebSocket ws;
    private CancellationTokenSource cts;
    private WaitForSeconds wait;

    private Texture2D encodeTex;
    private Color32[] pixelBuffer;
    private PassthroughCameraEye Eye => webCamTextureManager.Eye;

    private byte[] latestJpgFromServer;
    private readonly object jpgLock = new object();

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

        // Ensure material is transparent  
        var mat = cameraQuad.GetComponent<Renderer>().material;
        mat.shader = Shader.Find("Unlit/Transparent");
        mat.mainTexture = tex;

        ScaleQuadToCameraFOV();

        wait = new WaitForSeconds(1f / Mathf.Max(1, targetFps));
        encodeTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);

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
            Texture2D maskTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            maskTex.LoadImage(incoming);

            Color[] pixels = maskTex.GetPixels();
            for (int i = 0; i < pixels.Length; i++)
            {
                float maskValue = pixels[i].r; // threshold or segmentation mask  
                if (maskValue > 0.5f)
                    pixels[i] = new Color(0f, 0f, 1f, 0.7f); // semi-transparent red  
                else
                    pixels[i] = new Color(0f, 0f, 0f, 0f); // transparent
            }
            maskTex.SetPixels(pixels);
            maskTex.Apply();

            cameraQuad.GetComponent<Renderer>().material.mainTexture = maskTex;
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
    }  


}
