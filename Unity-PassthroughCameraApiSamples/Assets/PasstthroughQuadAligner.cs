using System;
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

    /*private void Start()
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
    }*/

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
