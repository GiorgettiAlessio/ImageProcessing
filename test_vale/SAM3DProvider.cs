using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class SAM3DProvider : MonoBehaviour, IPoseProvider
{
    [Range(1f, 60f)] public float smoothing = 12f;
    [Range(1f, 100f)] public float fastSmoothing = 25f;
    [Range(0f, 5f)] public float rotationDeadZone = 0.5f;
    [Range(0f, 3f)] public float rootPositionScale = 1.0f;
    [Range(0.1f, 10f)] public float maxRootDistance = 2.0f;
    [Range(1f, 30f)] public float positionSmoothing = 8f;
    public float calibrationDelay = 5f;

    private Animator animator;
    private Dictionary<string, Transform> boneMap = new Dictionary<string, Transform>();
    private Dictionary<string, Quaternion> bindRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> calibrationRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> lastRawRotation = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> targetRotations = new Dictionary<string, Quaternion>();

    private Vector3 lastRootPosition;
    private Vector3 calibrationRootPosition;
    private Vector3 avatarInitialPosition;
    private bool isCalibrated = false;
    private float calibrationTimer = 0f;

    [System.Serializable]
    public class JointData { public float x, y, z; }

    [System.Serializable]
    public class MotionPayload {
        public int person_id;
        public double timestamp;
        public Dictionary<string, JointData> unity_rotations_deg;
        public JointData root_position;
    }

    void Awake()
    {
        enabled = false; // Disattivato di default
    }

    private string ConvertSamBoneName(string samName)
    {
        switch (samName)
        {
            case "hip": return "hip";
            case "abdomen": return "abdomen";
            case "chest": return "chest";
            case "neck": return "neck";
            case "head": return "head";
            case "lShldr": return "lShldr";
            case "lForeArm": return "lForeArm";
            case "lHand": return "lHand";
            case "rShldr": return "rShldr";
            case "rForeArm": return "rForeArm";
            case "rHand": return "rHand";
            case "lThigh": return "lThigh";
            case "lShin": return "lShin";
            case "lFoot": return "lFoot";
            case "rThigh": return "rThigh";
            case "rShin": return "rShin";
            case "rFoot": return "rFoot";
            default: return null;
        }
    }

    public void ApplyPose(string jsonPayload, Animator targetAnimator)
    {
        if (targetAnimator == null) return;
        if (animator == null)
        {
            animator = targetAnimator;
            InitializeBoneMapping();
            foreach (var kvp in boneMap) bindRotations[kvp.Key] = kvp.Value.localRotation;
            avatarInitialPosition = transform.position;
        }

        if (!isCalibrated)
        {
            calibrationTimer += Time.deltaTime;
            if (calibrationTimer >= calibrationDelay && lastRawRotation.Count > 0)
            {
                calibrationRotations = new Dictionary<string, Quaternion>(lastRawRotation);
                calibrationRootPosition = lastRootPosition;
                isCalibrated = true;
                Debug.Log("✅ SAM3D Calibrazione completata!");
            }
        }

        MotionPayload motionData;
        try { motionData = JsonConvert.DeserializeObject<MotionPayload>(jsonPayload); }
        catch { return; }

        if (motionData == null) return;

        if (motionData.root_position != null)
        {
            lastRootPosition = new Vector3(motionData.root_position.x, motionData.root_position.y, motionData.root_position.z);
        }

        if (isCalibrated && motionData.root_position != null)
        {
            Vector3 delta = (lastRootPosition - calibrationRootPosition) * rootPositionScale;
            if (delta.magnitude > maxRootDistance) delta = delta.normalized * maxRootDistance;
            transform.position = Vector3.Lerp(transform.position, avatarInitialPosition + delta, 1f - Mathf.Exp(-positionSmoothing * Time.deltaTime));
        }

        if (motionData.unity_rotations_deg == null) return;

        foreach (var item in motionData.unity_rotations_deg)
        {
            string samName = item.Key;
            if (item.Value == null) continue;

            Quaternion raw = Quaternion.Euler(item.Value.x, item.Value.y, item.Value.z);
            lastRawRotation[samName] = raw;

            if (!isCalibrated) continue;

            string unityBoneName = ConvertSamBoneName(samName);
            if (unityBoneName == null || !boneMap.TryGetValue(unityBoneName, out Transform boneTransform)) continue;
            if (!calibrationRotations.TryGetValue(samName, out Quaternion calibration)) continue;
            if (!bindRotations.TryGetValue(unityBoneName, out Quaternion bindRotation)) continue;

            Quaternion delta = Quaternion.Inverse(calibration) * raw;
            targetRotations[unityBoneName] = bindRotation * delta;
        }

        if (!isCalibrated) return;

        foreach (var kvp in targetRotations)
        {
            if (boneMap.TryGetValue(kvp.Key, out Transform boneTransform))
            {
                float angle = Quaternion.Angle(boneTransform.localRotation, kvp.Value);
                if (angle < rotationDeadZone) continue;
                float adaptiveSmoothing = Mathf.Lerp(smoothing, fastSmoothing, Mathf.Clamp01(angle / 45f));
                float t = 1f - Mathf.Exp(-adaptiveSmoothing * Time.deltaTime);
                boneTransform.localRotation = Quaternion.Slerp(boneTransform.localRotation, kvp.Value, t);
            }
        }
    }

    private void InitializeBoneMapping()
    {
        MapBone("hip", HumanBodyBones.Hips);
        MapBone("abdomen", HumanBodyBones.Spine);
        MapBone("chest", HumanBodyBones.Chest);
        MapBone("neck", HumanBodyBones.Neck);
        MapBone("head", HumanBodyBones.Head);
        MapBone("rShldr", HumanBodyBones.RightUpperArm);
        MapBone("rForeArm", HumanBodyBones.RightLowerArm);
        MapBone("rHand", HumanBodyBones.RightHand);
        MapBone("lShldr", HumanBodyBones.LeftUpperArm);
        MapBone("lForeArm", HumanBodyBones.LeftLowerArm);
        MapBone("lHand", HumanBodyBones.LeftHand);
        MapBone("rThigh", HumanBodyBones.RightUpperLeg);
        MapBone("rShin", HumanBodyBones.RightLowerLeg);
        MapBone("rFoot", HumanBodyBones.RightFoot);
        MapBone("lThigh", HumanBodyBones.LeftUpperLeg);
        MapBone("lShin", HumanBodyBones.LeftLowerLeg);
        MapBone("lFoot", HumanBodyBones.LeftFoot);
    }

    private void MapBone(string samName, HumanBodyBones humanoidBone)
    {
        Transform boneTransform = animator.GetBoneTransform(humanoidBone);
        if (boneTransform != null) boneMap[samName] = boneTransform;
    }
}