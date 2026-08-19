using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class AvatarController : MonoBehaviour
{
    [Header("Riferimento di Rete")]
    public UDPReceiver udpReceiver;

    [Header("Rotazioni")]
    [Tooltip("Smoothing generale delle ossa.")]
    [Range(1f, 60f)]
    public float smoothing = 20f;

    [Tooltip("Smoothing durante movimenti veloci.")]
    [Range(1f, 100f)]
    public float fastSmoothing = 45f;

    [Tooltip("Piccole variazioni sotto questa soglia vengono ignorate.")]
    [Range(0f, 5f)]
    public float rotationDeadZone = 0.15f;

    [Header("Movimento Avatar")]
    [Tooltip("Quanto il movimento reale viene amplificato nell'avatar.")]
    [Range(0f, 3f)]
    public float rootPositionScale = 1.0f;

    [Tooltip("Massimo spostamento dell'avatar dal punto iniziale.")]
    [Range(0.1f, 10f)]
    public float maxRootDistance = 2.0f;

    [Tooltip("Velocità con cui l'avatar segue la posizione.")]
    [Range(1f, 30f)]
    public float positionSmoothing = 8f;

    [Header("Calibrazione")]
    [Tooltip("Secondi di attesa prima della calibrazione automatica.")]
    public float calibrationDelay = 5f;

    [Header("Debug")]
    public bool debugSingleBone = false;
    public string debugBoneName = "R_Shoulder";

    private Vector3 lastRootPosition;

    private Animator animator;

    private Dictionary<string, Transform> boneMap =
        new Dictionary<string, Transform>();

    private Dictionary<string, Quaternion> bindRotations =
        new Dictionary<string, Quaternion>();

    private Dictionary<string, Quaternion> calibrationRotations =
        new Dictionary<string, Quaternion>();

    private Dictionary<string, Quaternion> lastRawRotation =
        new Dictionary<string, Quaternion>();

    private Vector3[] lastJoints3D = new Vector3[24];

    private bool isCalibrated = false;

    private Dictionary<string, Vector3> bindBoneDirections =
    new Dictionary<string, Vector3>();



    private void InitializeBindBoneDirections()
    {
        SaveBindDirection(
            "R_Shoulder",
            HumanBodyBones.RightUpperArm,
            HumanBodyBones.RightLowerArm
        );

        SaveBindDirection(
            "L_Shoulder",
            HumanBodyBones.LeftUpperArm,
            HumanBodyBones.LeftLowerArm
        );

        SaveBindDirection(
            "R_Elbow",
            HumanBodyBones.RightLowerArm,
            HumanBodyBones.RightHand
        );

        SaveBindDirection(
            "L_Elbow",
            HumanBodyBones.LeftLowerArm,
            HumanBodyBones.LeftHand
        );
    }

    private void SaveBindDirection(
        string boneName,
        HumanBodyBones boneType,
        HumanBodyBones childType)
    {
        Transform bone =
            animator.GetBoneTransform(boneType);

        Transform child =
            animator.GetBoneTransform(childType);

        if (bone == null || child == null)
            return;

        Vector3 direction =
            child.position - bone.position;

        if (direction.sqrMagnitude < 0.000001f)
            return;

        bindBoneDirections[boneName] =
            direction.normalized;
    }

    private Quaternion SmoothRotation(
        Quaternion current,
        Quaternion target,
        float deltaTime)
    {
        float angle = Quaternion.Angle(current, target);

        // Se la differenza è minuscola, non muovere l'osso.
        // Questo elimina parte del jitter.
        if (angle < rotationDeadZone)
            return current;

        // Più grande è il movimento,
        // più velocemente inseguiamo il target.
        float normalizedAngle =
            Mathf.Clamp01(angle / 45f);

        float adaptiveSmoothing =
            Mathf.Lerp(
                smoothing,
                fastSmoothing,
                normalizedAngle
            );

        float t =
            1f - Mathf.Exp(
                -adaptiveSmoothing * deltaTime
            );

        return Quaternion.Slerp(
            current,
            target,
            t
        );
    }

    private Vector3 GetJointDirection(
        int parentIndex,
        int childIndex)
    {
        Vector3 parent =
            lastJoints3D[parentIndex];

        Vector3 child =
            lastJoints3D[childIndex];

        Vector3 direction =
            child - parent;

        if (direction.sqrMagnitude < 0.000001f)
            return Vector3.zero;

        return direction.normalized;
    }

    // ============================================================
    // ROOT POSITION
    // ============================================================

    // Posizione HybrIK al momento della calibrazione
    private Vector3 calibrationRootPosition;

    // Posizione originale dell'avatar in Unity
    private Vector3 avatarInitialPosition;

    private float calibrationTimer = 0f;


    // ============================================================
    // JSON
    // ============================================================

    [System.Serializable]
    public class JointData
    {
        // Python manda QUATERNION x,y,z,w
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
            Debug.LogError(
                "❌ AvatarController: Animator NON trovato!"
            );
            return;
        }

        Debug.Log(
            "✅ Animator trovato: "
            + animator.gameObject.name
        );

        Debug.Log(
            "Animator isHuman = "
            + animator.isHuman
        );


        if (udpReceiver == null)
            udpReceiver = FindObjectOfType<UDPReceiver>();

        if (udpReceiver == null)
        {
            Debug.LogError(
                "❌ AvatarController: UDPReceiver NON trovato!"
            );
        }
        else
        {
            Debug.Log(
                "✅ UDPReceiver trovato: "
                + udpReceiver.gameObject.name
            );
        }


        InitializeBoneMapping();
        InitializeBindBoneDirections();

        // Salva bind pose
        foreach (var kvp in boneMap)
        {
            bindRotations[kvp.Key] =
                kvp.Value.localRotation;
        }


        // Salva posizione iniziale dell'avatar
        avatarInitialPosition = transform.position;


        Debug.Log(
            "✅ Bone mapping completato. "
            + "Ossa trovate: "
            + boneMap.Count
        );

        Debug.Log(
            "⏱️ Calibrazione automatica tra "
            + calibrationDelay
            + " secondi..."
        );
    }


    // ============================================================
    // MAPPING OSSA
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


    private void MapBone(
        string samName,
        HumanBodyBones humanoidBone)
    {
        if (animator == null)
            return;

        Transform boneTransform =
            animator.GetBoneTransform(humanoidBone);

        if (boneTransform != null)
        {
            boneMap[samName] = boneTransform;
        }
        else
        {
            Debug.LogWarning(
                "⚠️ Bone NON trovata: "
                + samName
            );
        }
    }


    // ============================================================
    // UPDATE
    // ============================================================

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


    // ============================================================
    // CALIBRAZIONE
    // ============================================================

    private void CalibrateNow()
    {
        if (lastRawRotation.Count == 0)
        {
            Debug.LogWarning(
                "⚠️ Nessun dato ricevuto da Python. "
                + "Riprovo tra 1 secondo..."
            );

            calibrationTimer =
                calibrationDelay - 1f;

            return;
        }


        // --------------------------------------------------------
        // CALIBRAZIONE ROTAZIONI
        // --------------------------------------------------------

        calibrationRotations.Clear();

        foreach (var kvp in lastRawRotation)
        {
            calibrationRotations[kvp.Key] =
                kvp.Value;
        }


        // --------------------------------------------------------
        // CALIBRAZIONE POSIZIONE
        // --------------------------------------------------------

        // La posizione HybrIK in questo momento
        // diventa il nostro ZERO.
        //
        // La posizione dell'avatar in Unity rimane invece
        // quella impostata nella scena.
        calibrationRootPosition = lastRootPosition;
        isCalibrated = true;

        Debug.Log(
            "📍 Root position calibrata."
        );

        Debug.Log(
            "📍 HybrIK root = "
            + calibrationRootPosition
        );

        Debug.Log(
            "📍 Avatar initial position = "
            + avatarInitialPosition
        );

        Debug.Log(
            "✅ CALIBRAZIONE COMPLETATA: "
            + calibrationRotations.Count
            + " giunti."
        );
    }


    // ============================================================
    // LATE UPDATE
    // ============================================================

    void LateUpdate()
    {
        if (udpReceiver == null)
            return;

        if (string.IsNullOrEmpty(
            udpReceiver.latestJSON))
            return;


        MotionPayload motionData = null;

        try
        {
            motionData =
                JsonConvert.DeserializeObject<MotionPayload>(
                    udpReceiver.latestJSON
                );
        }
        catch (System.Exception e)
        {
            Debug.LogError(
                "❌ ERRORE DESERIALIZZAZIONE JSON: "
                + e.Message
            );

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

        if (motionData.joint_xyz_3d != null &&
            motionData.joint_xyz_3d.Count >= 72)
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


        


        // ========================================================
        // ROOT POSITION
        // ========================================================

        if (
            isCalibrated &&
            motionData.root_position != null
        )
        {
            Vector3 currentRoot =
                new Vector3(
                    motionData.root_position.x,
                    motionData.root_position.y,
                    motionData.root_position.z
                );


            // Differenza rispetto alla posizione
            // al momento della calibrazione.
            Vector3 delta =
                currentRoot -
                calibrationRootPosition;


            // Applica scala
            delta *= rootPositionScale;


            // Limita lo spostamento massimo
            if (delta.magnitude > maxRootDistance)
            {
                delta =
                    delta.normalized *
                    maxRootDistance;
            }


            // Posizione target dell'avatar
            Vector3 targetPosition =
                avatarInitialPosition +
                delta;


            // Movimento fluido
            transform.position =
                Vector3.Lerp(
                    transform.position,
                    targetPosition,
                    Time.deltaTime *
                    positionSmoothing
                );
        }


        // ========================================================
        // ROTAZIONI
        // ========================================================

        if (motionData.unity_rotations_deg == null)
            return;


        foreach (
            var item
            in motionData.unity_rotations_deg
        )
        {
            // Debug singolo bone
            if (
                debugSingleBone &&
                item.Key != debugBoneName
            )
            {
                continue;
            }


            // ----------------------------------------------------
            // Quaternion Python → Unity
            // ----------------------------------------------------

            Quaternion raw =
                new Quaternion(
                    item.Value.x,
                    item.Value.y,
                    item.Value.z,
                    item.Value.w
                );


            lastRawRotation[item.Key] =
                raw;


            if (!isCalibrated)
                continue;


            // ----------------------------------------------------
            // Bone
            // ----------------------------------------------------

            if (
                !boneMap.TryGetValue(
                    item.Key,
                    out Transform boneTransform
                )
            )
            {
                continue;
            }


            // ----------------------------------------------------
            // Calibrazione
            // ----------------------------------------------------

            if (
                !calibrationRotations.TryGetValue(
                    item.Key,
                    out Quaternion calibration
                )
            )
            {
                continue;
            }


            // ----------------------------------------------------
            // Delta rispetto alla T-pose
            // ----------------------------------------------------

            Quaternion delta =
                Quaternion.Inverse(
                    calibration
                ) * raw;


            // ----------------------------------------------------
            // Bind pose avatar
            // ----------------------------------------------------

            Quaternion bindRotation =
                bindRotations[item.Key];


            Quaternion targetRotation =
                bindRotation * delta;


            // ----------------------------------------------------
            // Smoothing
            // ----------------------------------------------------

            boneTransform.localRotation =
                SmoothRotation(
                    boneTransform.localRotation,
                    targetRotation,
                    Time.deltaTime
                );
        }
    }
}
