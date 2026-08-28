using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class AvatarController : MonoBehaviour
{
    [Header("Riferimento di Rete")]
    public UDPReceiver udpReceiver;

    [Header("Rotazioni")]
    [Range(1f, 60f)]
    public float smoothing = 20f;

    [Range(1f, 100f)]
    public float fastSmoothing = 45f;

    [Range(0f, 5f)]
    public float rotationDeadZone = 0.15f;

    [Header("Gambe e Bacino - Naturalezza")]
    [Tooltip("HybrIK da webcam singola sottostima la rotazione delle gambe: alza se restano rigide, abbassa se tremano.")]
    [Range(1f, 3f)]
    public float legSwingBoost = 1.6f;

    // Più bassa di rotationDeadZone: senza, le micro-rotazioni delle gambe venivano scartate e restavano "congelate".
    [Range(0f, 5f)]
    public float legRotationDeadZone = 0.05f;

    [Range(1f, 60f)]
    public float legSmoothing = 14f;

    [Range(1f, 100f)]
    public float legFastSmoothing = 35f;

    [Header("Modalità Specchio (gambe)")]
    [Tooltip("Attivo: la gamba/anca sinistra tracciata pilota quella destra dell'avatar e viceversa.")]
    public bool mirrorLegs = false;

    [Tooltip("Da usare solo se, con lo specchio attivo, la gamba giusta si muove ma piega nel verso sbagliato.")]
    public bool invertLegRotationX = false;
    public bool invertLegRotationY = false;
    public bool invertLegRotationZ = false;

    private static readonly Dictionary<string, string> legMirrorMap = new Dictionary<string, string>
    {
        { "L_Hip", "R_Hip" }, { "R_Hip", "L_Hip" },
        { "L_Knee", "R_Knee" }, { "R_Knee", "L_Knee" },
        { "L_Ankle", "R_Ankle" }, { "R_Ankle", "L_Ankle" },
        { "L_Foot", "R_Foot" }, { "R_Foot", "L_Foot" }
    };

    [Header("Movimento naturale del bacino (idle sway)")]
    [Tooltip("Oscillazione continua del bacino per evitare l'effetto manichino fermo quando il tracking è statico.")]
    public bool enableHipSway = true;

    [Range(0f, 3f)]
    public float hipSwayAmount = 1.2f;

    [Range(0.05f, 1f)]
    public float hipSwaySpeed = 0.25f;

    [Header("Movimento Avatar")]
    [Range(0f, 3f)]
    public float rootPositionScale = 1.0f;

    [Range(0.1f, 10f)]
    public float maxRootDistance = 2.0f;

    [Range(1f, 30f)]
    public float positionSmoothing = 8f;

    [Header("Calibrazione")]
    public float calibrationDelay = 5f;

    [Header("Debug")]
    public bool debugSingleBone = false;
    public string debugBoneName = "R_Shoulder";

    private Vector3 lastRootPosition;
    private Animator animator;

    private Dictionary<string, Transform> boneMap = new Dictionary<string, Transform>();
    private Dictionary<string, Quaternion> bindRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> calibrationRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> lastRawRotation = new Dictionary<string, Quaternion>();
    private Vector3[] lastJoints3D = new Vector3[24];
    private bool isCalibrated = false;
    private Dictionary<string, Vector3> bindBoneDirections = new Dictionary<string, Vector3>();

    private static readonly HashSet<string> legBoneNames = new HashSet<string>
    {
        "L_Hip", "R_Hip",
        "L_Knee", "R_Knee",
        "L_Ankle", "R_Ankle",
        "L_Foot", "R_Foot"
    };

    private Vector3 calibrationRootPosition;
    private Vector3 avatarInitialPosition;
    private float calibrationTimer = 0f;

    private void InitializeBindBoneDirections()
    {
        SaveBindDirection("R_Shoulder", HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm);
        SaveBindDirection("L_Shoulder", HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm);
        SaveBindDirection("R_Elbow", HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand);
        SaveBindDirection("L_Elbow", HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand);
    }

    private void SaveBindDirection(string boneName, HumanBodyBones boneType, HumanBodyBones childType)
    {
        Transform bone = animator.GetBoneTransform(boneType);
        Transform child = animator.GetBoneTransform(childType);

        if (bone == null || child == null)
            return;

        Vector3 direction = child.position - bone.position;

        if (direction.sqrMagnitude < 0.000001f)
            return;

        bindBoneDirections[boneName] = direction.normalized;
    }

    private Quaternion SmoothRotation(
        Quaternion current,
        Quaternion target,
        float deltaTime,
        float deadZone,
        float baseSmoothing,
        float fastSmoothingValue)
    {
        float angle = Quaternion.Angle(current, target);

        if (angle < deadZone)
            return current;

        float normalizedAngle = Mathf.Clamp01(angle / 45f);
        float adaptiveSmoothing = Mathf.Lerp(baseSmoothing, fastSmoothingValue, normalizedAngle);
        float t = 1f - Mathf.Exp(-adaptiveSmoothing * deltaTime);

        return Quaternion.Slerp(current, target, t);
    }

    // Amplifica l'ampiezza della rotazione delta mantenendo lo stesso asse, per compensare la sottostima
    // di HybrIK sulle gambe. Normalizza il segno del quaternione (w >= 0) prima di estrarre angolo/asse:
    // altrimenti anca sinistra e destra, essendo specchiate, potevano finire su emisfere diverse e
    // l'amplificazione usciva sbagliata (gamba "storta a X" anche in T-pose).
    private Quaternion AmplifyRotation(Quaternion delta, float factor)
    {
        if (Mathf.Approximately(factor, 1f))
            return delta;

        if (delta.w < 0f)
        {
            delta = new Quaternion(-delta.x, -delta.y, -delta.z, -delta.w);
        }

        delta.ToAngleAxis(out float angle, out Vector3 axis);
        angle *= factor;

        return Quaternion.AngleAxis(angle, axis);
    }

    // Oscillazione procedurale del bacino, indipendente dai dati della webcam: evita l'effetto
    // "manichino fermo" quando il tracking resta piatto per qualche secondo. Il bob usa il doppio
    // della frequenza dello sway laterale così i due movimenti non risultano sincronizzati/meccanici.
    private Quaternion GetHipSwayOffset()
    {
        if (!enableHipSway)
            return Quaternion.identity;

        float t = Time.time * hipSwaySpeed * Mathf.PI * 2f;
        float sway = Mathf.Sin(t) * hipSwayAmount;
        float bob = Mathf.Sin(t * 2f) * (hipSwayAmount * 0.3f);

        return Quaternion.Euler(bob, 0f, sway);
    }

    private Vector3 GetJointDirection(int parentIndex, int childIndex)
    {
        Vector3 parent = lastJoints3D[parentIndex];
        Vector3 child = lastJoints3D[childIndex];
        Vector3 direction = child - parent;

        if (direction.sqrMagnitude < 0.000001f)
            return Vector3.zero;

        return direction.normalized;
    }

    [System.Serializable]
    public class JointData
    {
        public float x;
        public float y;
        public float z;
        public float w;
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

    void Start()
    {
        animator = GetComponent<Animator>();

        if (animator == null)
            animator = GetComponentInChildren<Animator>();

        if (animator == null)
        {
            Debug.LogError("AvatarController: Animator non trovato.");
            return;
        }

        if (udpReceiver == null)
            udpReceiver = FindObjectOfType<UDPReceiver>();

        if (udpReceiver == null)
        {
            Debug.LogError("AvatarController: UDPReceiver non trovato.");
        }

        InitializeBoneMapping();
        InitializeBindBoneDirections();

        foreach (var kvp in boneMap)
        {
            bindRotations[kvp.Key] = kvp.Value.localRotation;
        }

        avatarInitialPosition = transform.position;

        Debug.Log("Bone mapping completato. Ossa trovate: " + boneMap.Count);
        Debug.Log("Calibrazione automatica tra " + calibrationDelay + " secondi...");
    }

    private void InitializeBoneMapping()
    {
        MapBone("Pelvis", HumanBodyBones.Hips);
        MapBone("Spine1", HumanBodyBones.Spine);
        MapBone("Spine2", HumanBodyBones.Chest);
        MapBone("Spine3", HumanBodyBones.UpperChest);
        MapBone("Neck", HumanBodyBones.Neck);
        MapBone("Head", HumanBodyBones.Head);

        MapBone("L_Collar", HumanBodyBones.LeftShoulder);
        MapBone("L_Shoulder", HumanBodyBones.LeftUpperArm);
        MapBone("L_Elbow", HumanBodyBones.LeftLowerArm);
        MapBone("L_Wrist", HumanBodyBones.LeftHand);

        MapBone("R_Collar", HumanBodyBones.RightShoulder);
        MapBone("R_Shoulder", HumanBodyBones.RightUpperArm);
        MapBone("R_Elbow", HumanBodyBones.RightLowerArm);
        MapBone("R_Wrist", HumanBodyBones.RightHand);

        MapBone("L_Hip", HumanBodyBones.LeftUpperLeg);
        MapBone("L_Knee", HumanBodyBones.LeftLowerLeg);
        MapBone("L_Ankle", HumanBodyBones.LeftFoot);
        MapBone("L_Foot", HumanBodyBones.LeftToes);

        MapBone("R_Hip", HumanBodyBones.RightUpperLeg);
        MapBone("R_Knee", HumanBodyBones.RightLowerLeg);
        MapBone("R_Ankle", HumanBodyBones.RightFoot);
        MapBone("R_Foot", HumanBodyBones.RightToes);

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
        if (animator == null)
            return;

        Transform boneTransform = animator.GetBoneTransform(humanoidBone);

        if (boneTransform != null)
        {
            boneMap[samName] = boneTransform;
        }
        else
        {
            Debug.LogWarning("Bone non trovata: " + samName);
        }
    }

    void Update()
    {
        if (isCalibrated)
            return;

        calibrationTimer += Time.deltaTime;

        if (calibrationTimer >= calibrationDelay)
        {
            CalibrateNow();
        }
    }

    private void CalibrateNow()
    {
        if (lastRawRotation.Count == 0)
        {
            Debug.LogWarning("Nessun dato ricevuto da Python. Riprovo tra 1 secondo...");
            calibrationTimer = calibrationDelay - 1f;
            return;
        }

        calibrationRotations.Clear();

        foreach (var kvp in lastRawRotation)
        {
            calibrationRotations[kvp.Key] = kvp.Value;
        }

        calibrationRootPosition = lastRootPosition;
        isCalibrated = true;

        Debug.Log("Root position calibrata. HybrIK root = " + calibrationRootPosition);
        Debug.Log("Avatar initial position = " + avatarInitialPosition);
        Debug.Log("Calibrazione completata: " + calibrationRotations.Count + " giunti.");
    }

    void LateUpdate()
    {
        if (udpReceiver == null)
            return;

        if (string.IsNullOrEmpty(udpReceiver.latestJSON))
            return;

        MotionPayload motionData = null;

        try
        {
            motionData = JsonConvert.DeserializeObject<MotionPayload>(udpReceiver.latestJSON);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Errore deserializzazione JSON: " + e.Message);
            return;
        }

        if (motionData == null)
            return;

        if (motionData.root_position != null)
        {
            lastRootPosition = new Vector3(
                motionData.root_position.x,
                motionData.root_position.y,
                motionData.root_position.z
            );
        }

        if (motionData.joint_xyz_3d != null && motionData.joint_xyz_3d.Count >= 72)
        {
            for (int i = 0; i < 24; i++)
            {
                int index = i * 3;

                lastJoints3D[i] = new Vector3(
                    motionData.joint_xyz_3d[index],
                    motionData.joint_xyz_3d[index + 1],
                    -motionData.joint_xyz_3d[index + 2]
                );
            }
        }

        if (isCalibrated && motionData.root_position != null)
        {
            Vector3 currentRoot = new Vector3(
                motionData.root_position.x,
                motionData.root_position.y,
                motionData.root_position.z
            );

            Vector3 delta = currentRoot - calibrationRootPosition;
            delta *= rootPositionScale;

            if (delta.magnitude > maxRootDistance)
            {
                delta = delta.normalized * maxRootDistance;
            }

            Vector3 targetPosition = avatarInitialPosition + delta;

            transform.position = Vector3.Lerp(
                transform.position,
                targetPosition,
                Time.deltaTime * positionSmoothing
            );
        }

        if (motionData.unity_rotations_deg == null)
            return;

        foreach (var item in motionData.unity_rotations_deg)
        {
            if (debugSingleBone && item.Key != debugBoneName)
            {
                continue;
            }

            Quaternion raw = new Quaternion(
                item.Value.x,
                item.Value.y,
                item.Value.z,
                item.Value.w
            );

            // Inversione per-asse solo sulle gambe: serve a correggere il verso di piegamento
            // quando mirrorLegs è attivo e il ginocchio piega dal lato sbagliato.
            if (legBoneNames.Contains(item.Key) &&
                (invertLegRotationX || invertLegRotationY || invertLegRotationZ))
            {
                raw = new Quaternion(
                    invertLegRotationX ? -raw.x : raw.x,
                    invertLegRotationY ? -raw.y : raw.y,
                    invertLegRotationZ ? -raw.z : raw.z,
                    raw.w
                );
            }

            lastRawRotation[item.Key] = raw;

            if (!isCalibrated)
                continue;

            string targetBoneName = item.Key;

            if (mirrorLegs && legMirrorMap.TryGetValue(item.Key, out string mirroredName))
            {
                targetBoneName = mirroredName;
            }

            if (!boneMap.TryGetValue(targetBoneName, out Transform boneTransform))
            {
                continue;
            }

            if (!calibrationRotations.TryGetValue(item.Key, out Quaternion calibration))
            {
                continue;
            }

            Quaternion delta = Quaternion.Inverse(calibration) * raw;

            bool isLegBone = legBoneNames.Contains(item.Key);

            if (isLegBone)
            {
                delta = AmplifyRotation(delta, legSwingBoost);
            }

            Quaternion bindRotation = bindRotations[targetBoneName];
            Quaternion targetRotation = bindRotation * delta;

            if (targetBoneName == "Pelvis")
            {
                targetRotation = targetRotation * GetHipSwayOffset();
            }

            float deadZone = isLegBone ? legRotationDeadZone : rotationDeadZone;
            float baseSmoothing = isLegBone ? legSmoothing : smoothing;
            float fastSmoothingValue = isLegBone ? legFastSmoothing : fastSmoothing;

            boneTransform.localRotation = SmoothRotation(
                boneTransform.localRotation,
                targetRotation,
                Time.deltaTime,
                deadZone,
                baseSmoothing,
                fastSmoothingValue
            );
        }
    }
}