/*using System;
using System.Collections;
using System.Collections.Generic;
using Meta.XR.Samples;
using PassthroughCameraSamples;
using UnityEngine;

public class PassthroughQuadAligner : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager webCamTextureManager;
    [SerializeField] private GameObject centerEyeAnchor;   // Usually OVRCameraRig's CenterEyeAnchor
    [SerializeField] private GameObject cameraQuad;        // The Quad that displays the camera feed
    [SerializeField] private float quadDistance = 1.0f;    // Distance in meters in front of the headset

    private PassthroughCameraEye Eye => webCamTextureManager.Eye;

    *//*private void Start()
    {
        // Assign camera texture to the quad’s material
        var renderer = cameraQuad.GetComponent<Renderer>();
        renderer.material.mainTexture = webCamTextureManager.WebCamTexture;
        renderer.material.shader = Shader.Find("Unlit/Texture");

        // Match the camera's FOV to the quad size
        ScaleQuadToCameraFOV();
    }

    private void Update()
    {
        // Position and orient the quad in front of the headset
        var headPose = centerEyeAnchor.transform;
        cameraQuad.transform.position = headPose.position + headPose.forward * quadDistance;
        cameraQuad.transform.rotation = headPose.rotation;
    }*//*

    private void Start()
    {
        var renderer = cameraQuad.GetComponent<Renderer>();

        // Make sure shader exists
        if (renderer.material.shader == null)
            renderer.material.shader = Shader.Find("Unlit/Texture");

        var tex = webCamTextureManager.WebCamTexture;
        if (tex != null && !tex.isPlaying)
            tex.Play();

        renderer.material.mainTexture = tex;

        ScaleQuadToCameraFOV();
    }

    private void Update()
    {
        var tex = webCamTextureManager.WebCamTexture;
        if (tex != null && tex.isPlaying)
        {
            cameraQuad.GetComponent<Renderer>().material.mainTexture = tex;

            var headPose = centerEyeAnchor.transform;
            cameraQuad.transform.position = headPose.position + headPose.forward * quadDistance;
            cameraQuad.transform.rotation = headPose.rotation;
        }
    }


    private void ScaleQuadToCameraFOV()
    {
        var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(Eye);
        var resolution = intrinsics.Resolution;

        // Compute horizontal FOV from the camera intrinsics
        var leftRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(0, resolution.y / 2));
        var rightRay = PassthroughCameraUtils.ScreenPointToRayInCamera(Eye, new Vector2Int(resolution.x, resolution.y / 2));
        float horizontalFovDeg = Vector3.Angle(leftRay.direction, rightRay.direction);
        float horizontalFovRad = horizontalFovDeg * Mathf.Deg2Rad;

        // Use FOV to determine quad width/height
        float quadWidth = 2f * quadDistance * Mathf.Tan(horizontalFovRad / 2f);
        float aspect = (float)resolution.y / resolution.x;
        float quadHeight = quadWidth * aspect;

        cameraQuad.transform.localScale = new Vector3(quadWidth, quadHeight, 1f);
    }
}



*/

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


