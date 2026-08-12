using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class AvatarController : MonoBehaviour
{
    [Header("Riferimento di Rete")]
    public UDPReceiver udpReceiver;

    [Header("Impostazioni di Fluidità")]
    [Range(1f, 30f)]
    public float smoothing = 15f;

    private Animator animator;

    private Dictionary<string, Transform> boneMap =
        new Dictionary<string, Transform>();

    // Rotazione di riposo (bind pose) di ogni osso, salvata all'avvio
    private Dictionary<string, Quaternion> bindRotations =
        new Dictionary<string, Quaternion>();

    // Offset di calibrazione T-pose, calcolato quando premi "C"
    private Dictionary<string, Quaternion> calibrationOffset =
        new Dictionary<string, Quaternion>();

    // Ultima rotazione grezza ricevuta da Python per ogni osso
    private Dictionary<string, Quaternion> lastRawRotation =
        new Dictionary<string, Quaternion>();

    private bool isCalibrated = false;


    // ============================================================
    // DATI JSON
    // ============================================================

    [System.Serializable]
    public class JointData
    {
        public float x;
        public float y;
        public float z;
    }

    [System.Serializable]
    public class MotionPayload
    {
        public int person_id;
        public double timestamp;
        public Dictionary<string, JointData> unity_rotations_deg;
        public List<float> joint_xyz_3d;
        public JointData root_position;
    }


    // ============================================================
    // START
    // ============================================================

    void Start()
    {
        animator = GetComponent<Animator>();

        if (animator == null)
            animator = GetComponentInChildren<Animator>();

        if (animator == null)
        {
            Debug.LogError("❌ AvatarController: Animator NON trovato!");
            return;
        }

        Debug.Log("✅ Animator trovato: " + animator.gameObject.name);

        if (udpReceiver == null)
            udpReceiver = FindObjectOfType<UDPReceiver>();

        if (udpReceiver == null)
        {
            Debug.LogError("❌ AvatarController: UDPReceiver NON trovato nella scena!");
        }
        else
        {
            Debug.Log("✅ UDPReceiver trovato: " + udpReceiver.gameObject.name);
        }

        InitializeBoneMapping();

        // Salva la rotazione di bind (riposo) di ogni osso mappato
        foreach (var kvp in boneMap)
        {
            bindRotations[kvp.Key] = kvp.Value.localRotation;
        }

        Debug.Log("✅ Bone mapping completato. Ossa trovate: " + boneMap.Count);
        Debug.Log("👉 Mettiti in T-pose davanti alla webcam e premi 'C' per calibrare.");
    }


    // ============================================================
    // MAPPATURA OSSA
    // ============================================================

    private void InitializeBoneMapping()
    {
        // TRONCO
        MapBone("Pelvis", HumanBodyBones.Hips);
        MapBone("Spine1", HumanBodyBones.Spine);
        MapBone("Spine2", HumanBodyBones.Chest);
        MapBone("Spine3", HumanBodyBones.UpperChest);
        MapBone("Neck", HumanBodyBones.Neck);
        MapBone("Head", HumanBodyBones.Head);

        // BRACCIO SINISTRO
        MapBone("L_Collar", HumanBodyBones.LeftShoulder);
        MapBone("L_Shoulder", HumanBodyBones.LeftUpperArm);
        MapBone("L_Elbow", HumanBodyBones.LeftLowerArm);
        MapBone("L_Wrist", HumanBodyBones.LeftHand);

        // BRACCIO DESTRO
        MapBone("R_Collar", HumanBodyBones.RightShoulder);
        MapBone("R_Shoulder", HumanBodyBones.RightUpperArm);
        MapBone("R_Elbow", HumanBodyBones.RightLowerArm);
        MapBone("R_Wrist", HumanBodyBones.RightHand);

        // GAMBA SINISTRA
        MapBone("L_Hip", HumanBodyBones.LeftUpperLeg);
        MapBone("L_Knee", HumanBodyBones.LeftLowerLeg);
        MapBone("L_Ankle", HumanBodyBones.LeftFoot);
        MapBone("L_Foot", HumanBodyBones.LeftToes);

        // GAMBA DESTRA
        MapBone("R_Hip", HumanBodyBones.RightUpperLeg);
        MapBone("R_Knee", HumanBodyBones.RightLowerLeg);
        MapBone("R_Ankle", HumanBodyBones.RightFoot);
        MapBone("R_Foot", HumanBodyBones.RightToes);

        // MANO SINISTRA
        MapBone("L_Thumb1", HumanBodyBones.LeftThumbProximal);
        MapBone("L_Thumb2", HumanBodyBones.LeftThumbIntermediate);
        MapBone("L_Thumb3", HumanBodyBones.LeftThumbDistal);
        MapBone("L_Index1", HumanBodyBones.LeftIndexProximal);
        MapBone("L_Index2", HumanBodyBones.LeftIndexIntermediate);
        MapBone("L_Index3", HumanBodyBones.LeftIndexDistal);
        MapBone("L_Middle1", HumanBodyBones.LeftMiddleProximal);
        MapBone("L_Middle2", HumanBodyBones.LeftMiddleIntermediate);
        MapBone("L_Middle3", HumanBodyBones.LeftMiddleDistal);
        MapBone("L_Ring1", HumanBodyBones.LeftRingProximal);
        MapBone("L_Ring2", HumanBodyBones.LeftRingIntermediate);
        MapBone("L_Ring3", HumanBodyBones.LeftRingDistal);
        MapBone("L_Pinky1", HumanBodyBones.LeftLittleProximal);
        MapBone("L_Pinky2", HumanBodyBones.LeftLittleIntermediate);
        MapBone("L_Pinky3", HumanBodyBones.LeftLittleDistal);

        // MANO DESTRA
        MapBone("R_Thumb1", HumanBodyBones.RightThumbProximal);
        MapBone("R_Thumb2", HumanBodyBones.RightThumbIntermediate);
        MapBone("R_Thumb3", HumanBodyBones.RightThumbDistal);
        MapBone("R_Index1", HumanBodyBones.RightIndexProximal);
        MapBone("R_Index2", HumanBodyBones.RightIndexIntermediate);
        MapBone("R_Index3", HumanBodyBones.RightIndexDistal);
        MapBone("R_Middle1", HumanBodyBones.RightMiddleProximal);
        MapBone("R_Middle2", HumanBodyBones.RightMiddleIntermediate);
        MapBone("R_Middle3", HumanBodyBones.RightMiddleDistal);
        MapBone("R_Ring1", HumanBodyBones.RightRingProximal);
        MapBone("R_Ring2", HumanBodyBones.RightRingIntermediate);
        MapBone("R_Ring3", HumanBodyBones.RightRingDistal);
        MapBone("R_Pinky1", HumanBodyBones.RightLittleProximal);
        MapBone("R_Pinky2", HumanBodyBones.RightLittleIntermediate);
        MapBone("R_Pinky3", HumanBodyBones.RightLittleDistal);
    }

    private void MapBone(string samName, HumanBodyBones humanoidBone)
    {
        if (animator == null) return;

        Transform boneTransform = animator.GetBoneTransform(humanoidBone);

        if (boneTransform != null)
        {
            boneMap[samName] = boneTransform;
        }
        else
        {
            Debug.LogWarning("⚠️ Bone NON trovata: " + samName);
        }
    }


    // ============================================================
    // UPDATE — gestisce il tasto di calibrazione
    // ============================================================

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
        {
            CalibrateNow();
        }
    }

    private void CalibrateNow()
    {
        calibrationOffset.Clear();

        foreach (var kvp in lastRawRotation)
        {
            calibrationOffset[kvp.Key] = Quaternion.Inverse(kvp.Value);
        }

        isCalibrated = true;

        Debug.Log("✅ Calibrazione T-pose eseguita su " + calibrationOffset.Count + " giunti");
    }


    // ============================================================
    // LATE UPDATE
    // ============================================================

    void LateUpdate()
    {
        if (udpReceiver == null || string.IsNullOrEmpty(udpReceiver.latestJSON))
            return;

        MotionPayload motionData = null;

        try
        {
            motionData = JsonConvert.DeserializeObject<MotionPayload>(udpReceiver.latestJSON);
        }
        catch (System.Exception e)
        {
            Debug.LogError("❌ ERRORE DESERIALIZZAZIONE JSON: " + e.Message);
            return;
        }

        if (motionData?.unity_rotations_deg == null) return;

        foreach (var item in motionData.unity_rotations_deg)
        {
            Quaternion raw = Quaternion.Euler(item.Value.x, item.Value.y, item.Value.z);
            lastRawRotation[item.Key] = raw;

            if (!isCalibrated) continue;

            if (boneMap.TryGetValue(item.Key, out Transform boneTransform) &&
                bindRotations.TryGetValue(item.Key, out Quaternion bindRot) &&
                calibrationOffset.TryGetValue(item.Key, out Quaternion calibRaw))
            {
                // Nuovo ordine: delta = raw relativo alla calibrazione, letto "dall'esterno"
                Quaternion delta = Quaternion.Inverse(calibRaw) * raw;
                Quaternion targetRotation = bindRot * delta;

                boneTransform.localRotation = Quaternion.Slerp(
                    boneTransform.localRotation,
                    targetRotation,
                    Time.deltaTime * smoothing
                );
            }
        }
    }
}