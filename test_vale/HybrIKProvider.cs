using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class HybrIKProvider : MonoBehaviour, IPoseProvider
{
    [Header("Rotazioni")]
    [Range(1f, 60f)] public float smoothing = 20f;
    [Range(1f, 100f)] public float fastSmoothing = 45f;
    [Range(0f, 5f)] public float rotationDeadZone = 0.15f;

    [Header("Movimento Avatar")]
    [Range(0f, 3f)] public float rootPositionScale = 1.0f;
    [Range(0.1f, 10f)] public float maxRootDistance = 2.0f;
    [Range(1f, 30f)] public float positionSmoothing = 8f;
    public float calibrationDelay = 5f;

    private Animator animator;
    private Dictionary<string, Transform> boneMap = new Dictionary<string, Transform>();
    private Dictionary<string, Quaternion> bindRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> calibrationRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> lastRawRotation = new Dictionary<string, Quaternion>();
    
    private Vector3 lastRootPosition;
    private Vector3 calibrationRootPosition;
    private Vector3 avatarInitialPosition;
    private bool isCalibrated = false;
    private float calibrationTimer = 0f;

    [System.Serializable]
    public class JointData { public float x, y, z, w; }

    [System.Serializable]
    public class MotionPayload {
        public int person_id;
        public double timestamp;
        public Dictionary<string, JointData> unity_rotations_deg;
        public List<float> joint_xyz_3d;
        public JointData root_position;
    }

    void Awake()
    {
        enabled = false; // Disattivato finché il ConnectionManager non lo avvia
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
                Debug.Log("✅ HybrIK Calibrazione completata!");
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

        // Root Movement
        if (isCalibrated && motionData.root_position != null)
        {
            Vector3 delta = (lastRootPosition - calibrationRootPosition) * rootPositionScale;
            if (delta.magnitude > maxRootDistance) delta = delta.normalized * maxRootDistance;
            transform.position = Vector3.Lerp(transform.position, avatarInitialPosition + delta, Time.deltaTime * positionSmoothing);
        }

        if (motionData.unity_rotations_deg == null) return;

        foreach (var item in motionData.unity_rotations_deg)
        {
            Quaternion raw = new Quaternion(item.Value.x, item.Value.y, item.Value.z, item.Value.w);
            lastRawRotation[item.Key] = raw;

            if (!isCalibrated) continue;
            if (!boneMap.TryGetValue(item.Key, out Transform boneTransform)) continue;
            if (!calibrationRotations.TryGetValue(item.Key, out Quaternion calibration)) continue;

            Quaternion deltaRot = Quaternion.Inverse(calibration) * raw;
            Quaternion targetRotation = bindRotations[item.Key] * deltaRot;

            float angle = Quaternion.Angle(boneTransform.localRotation, targetRotation);
            if (angle < rotationDeadZone) continue;

            float adaptiveSmoothing = Mathf.Lerp(smoothing, fastSmoothing, Mathf.Clamp01(angle / 45f));
            float t = 1f - Mathf.Exp(-adaptiveSmoothing * Time.deltaTime);

            boneTransform.localRotation = Quaternion.Slerp(boneTransform.localRotation, targetRotation, t);
        }
    }

    private void InitializeBoneMapping()
    {
        MapBone("Pelvis", HumanBodyBones.Hips);
        MapBone("Spine1", HumanBodyBones.Spine);
        MapBone("Spine2", HumanBodyBones.Chest);
        MapBone("Spine3", HumanBodyBones.UpperChest);
        MapBone("Neck", HumanBodyBones.Neck);
        MapBone("Head", HumanBodyBones.Head);
        MapBone("L_Shoulder", HumanBodyBones.LeftUpperArm);
        MapBone("L_Elbow", HumanBodyBones.LeftLowerArm);
        MapBone("L_Wrist", HumanBodyBones.LeftHand);
        MapBone("R_Shoulder", HumanBodyBones.RightUpperArm);
        MapBone("R_Elbow", HumanBodyBones.RightLowerArm);
        MapBone("R_Wrist", HumanBodyBones.RightHand);
        MapBone("L_Hip", HumanBodyBones.LeftUpperLeg);
        MapBone("L_Knee", HumanBodyBones.LeftLowerLeg);
        MapBone("L_Ankle", HumanBodyBones.LeftFoot);
        MapBone("R_Hip", HumanBodyBones.RightUpperLeg);
        MapBone("R_Knee", HumanBodyBones.RightLowerLeg);
        MapBone("R_Ankle", HumanBodyBones.RightFoot);
    }

    private void MapBone(string samName, HumanBodyBones humanoidBone)
    {
        Transform boneTransform = animator.GetBoneTransform(humanoidBone);
        if (boneTransform != null) boneMap[samName] = boneTransform;
    }
}
