using UnityEngine;
using System.Collections.Generic;
using Newtonsoft.Json;

public class AvatarController : MonoBehaviour
{
    // ============================================================
    // RIFERIMENTO DI RETE
    // ============================================================

    [Header("Riferimento di Rete")]
    public UDPReceiver udpReceiver;


    // ============================================================
    // ROTAZIONI
    // ============================================================

    [Header("Rotazioni")]

    [Tooltip("Smoothing generale delle ossa.")]
    [Range(1f, 60f)]
    public float smoothing = 12f;

    [Tooltip("Smoothing durante movimenti veloci.")]
    [Range(1f, 100f)]
    public float fastSmoothing = 25f;

    [Tooltip("Piccole variazioni sotto questa soglia vengono ignorate.")]
    [Range(0f, 5f)]
    public float rotationDeadZone = 0.5f;


    // ============================================================
    // MOVIMENTO AVATAR
    // ============================================================

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


    // ============================================================
    // CALIBRAZIONE
    // ============================================================

    [Header("Calibrazione")]

    [Tooltip("Secondi di attesa prima della calibrazione automatica.")]
    public float calibrationDelay = 5f;


    // ============================================================
    // DEBUG
    // ============================================================

    [Header("Debug")]

    public bool debugSingleBone = false;

    public string debugBoneName = "R_Shoulder";

    [Tooltip("Stampa i messaggi di mapping mancanti.")]
    public bool debugMapping = false;


    // ============================================================
    // DATI INTERNI
    // ============================================================

    private HashSet<string> receivedBoneNames =
        new HashSet<string>();

    private HashSet<string> warnedMissingBones =
        new HashSet<string>();

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

    // Target corrente ricevuto da SAM.
    private Dictionary<string, Quaternion> targetRotations =
        new Dictionary<string, Quaternion>();

    // Ultima rotazione filtrata.
    private Dictionary<string, Quaternion> filteredRotations =
        new Dictionary<string, Quaternion>();

    private Vector3[] lastJoints3D =
        new Vector3[24];

    private bool isCalibrated = false;

    private float calibrationTimer = 0f;


    // ============================================================
    // ROOT POSITION
    // ============================================================

    private Vector3 calibrationRootPosition;

    private Vector3 avatarInitialPosition;


    // ============================================================
    // BIND DIRECTIONS
    // ============================================================

    private Dictionary<string, Vector3> bindBoneDirections =
        new Dictionary<string, Vector3>();


    // ============================================================
    // CONVERSIONE NOMI SAM → UNITY
    // ============================================================

    private string ConvertSamBoneName(string samName)
    {
        switch (samName)
        {
            // ----------------------------------------------------
            // Tronco
            // ----------------------------------------------------

            case "hip":
                return "hip";

            case "abdomen":
                return "abdomen";

            case "chest":
                return "chest";

            case "neck":
                return "neck";

            case "head":
                return "head";


            // ----------------------------------------------------
            // Braccio sinistro
            // ----------------------------------------------------

            case "lCollar":
                return "lCollar";

            case "lShldr":
                return "lShldr";

            case "lForeArm":
                return "lForeArm";

            case "lHand":
                return "lHand";


            // ----------------------------------------------------
            // Braccio destro
            // ----------------------------------------------------

            case "rCollar":
                return "rCollar";

            case "rShldr":
                return "rShldr";

            case "rForeArm":
                return "rForeArm";

            case "rHand":
                return "rHand";


            // ----------------------------------------------------
            // Gamba sinistra
            // ----------------------------------------------------

            case "lThigh":
                return "lThigh";

            case "lShin":
                return "lShin";

            case "lFoot":
                return "lFoot";


            // ----------------------------------------------------
            // Gamba destra
            // ----------------------------------------------------

            case "rThigh":
                return "rThigh";

            case "rShin":
                return "rShin";

            case "rFoot":
                return "rFoot";


            default:
                return null;
        }
    }


    // ============================================================
    // INITIALIZE BIND DIRECTIONS
    // ============================================================

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


    // ============================================================
    // SMOOTHING ROTAZIONE
    // ============================================================

    private Quaternion SmoothRotation(
        Quaternion current,
        Quaternion target,
        float deltaTime)
    {
        float angle =
            Quaternion.Angle(
                current,
                target
            );


        // --------------------------------------------------------
        // Dead zone anti-jitter
        // --------------------------------------------------------

        if (angle < rotationDeadZone)
            return current;


        // --------------------------------------------------------
        // Smoothing adattivo
        // --------------------------------------------------------

        float normalizedAngle =
            Mathf.Clamp01(angle / 45f);


        float adaptiveSmoothing =
            Mathf.Lerp(
                smoothing,
                fastSmoothing,
                normalizedAngle
            );


        // --------------------------------------------------------
        // Smoothing indipendente dal frame rate
        // --------------------------------------------------------

        float t =
            1f -
            Mathf.Exp(
                -adaptiveSmoothing *
                deltaTime
            );


        return Quaternion.Slerp(
            current,
            target,
            t
        );
    }


    // ============================================================
    // JOINT DIRECTION
    // ============================================================

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
    // JSON
    // ============================================================

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

        public Dictionary<string, JointData>
            unity_rotations_deg;

        public List<float>
            joint_xyz_3d;

        public JointData
            root_position;
    }


    // ============================================================
    // START
    // ============================================================

    void Start()
    {
        animator =
            GetComponent<Animator>();


        if (animator == null)
        {
            animator =
                GetComponentInChildren<Animator>();
        }


        if (animator == null)
        {
            Debug.LogError(
                "❌ AvatarController: Animator NON trovato!"
            );

            return;
        }


        Debug.Log(
            "✅ Animator trovato: " +
            animator.gameObject.name
        );


        Debug.Log(
            "Animator isHuman = " +
            animator.isHuman
        );


        // --------------------------------------------------------
        // UDP
        // --------------------------------------------------------

        if (udpReceiver == null)
        {
            udpReceiver =
                FindObjectOfType<UDPReceiver>();
        }


        if (udpReceiver == null)
        {
            Debug.LogError(
                "❌ AvatarController: UDPReceiver NON trovato!"
            );
        }
        else
        {
            Debug.Log(
                "✅ UDPReceiver trovato: " +
                udpReceiver.gameObject.name
            );
        }


        // --------------------------------------------------------
        // Mapping
        // --------------------------------------------------------

        InitializeBoneMapping();

        InitializeBindBoneDirections();


        // --------------------------------------------------------
        // Bind pose
        // --------------------------------------------------------

        foreach (var kvp in boneMap)
        {
            bindRotations[kvp.Key] =
                kvp.Value.localRotation;
        }


        // --------------------------------------------------------
        // Posizione iniziale avatar
        // --------------------------------------------------------

        avatarInitialPosition =
            transform.position;


        Debug.Log(
            "✅ Bone mapping completato. " +
            "Ossa trovate: " +
            boneMap.Count
        );


        Debug.Log(
            "⏱️ Calibrazione automatica tra " +
            calibrationDelay +
            " secondi..."
        );
    }


    // ============================================================
    // MAPPING OSSA
    // ============================================================

    private void InitializeBoneMapping()
    {
        boneMap.Clear();


        // ========================================================
        // CORPO
        // ========================================================

        MapBone(
            "hip",
            HumanBodyBones.Hips
        );

        MapBone(
            "abdomen",
            HumanBodyBones.Spine
        );

        MapBone(
            "chest",
            HumanBodyBones.Chest
        );

        MapBone(
            "neck",
            HumanBodyBones.Neck
        );

        MapBone(
            "head",
            HumanBodyBones.Head
        );


        // ========================================================
        // BRACCIO DESTRO
        // ========================================================

        MapBone(
            "rCollar",
            HumanBodyBones.RightShoulder
        );

        MapBone(
            "rShldr",
            HumanBodyBones.RightUpperArm
        );

        MapBone(
            "rForeArm",
            HumanBodyBones.RightLowerArm
        );

        MapBone(
            "rHand",
            HumanBodyBones.RightHand
        );


        // ========================================================
        // BRACCIO SINISTRO
        // ========================================================

        MapBone(
            "lCollar",
            HumanBodyBones.LeftShoulder
        );

        MapBone(
            "lShldr",
            HumanBodyBones.LeftUpperArm
        );

        MapBone(
            "lForeArm",
            HumanBodyBones.LeftLowerArm
        );

        MapBone(
            "lHand",
            HumanBodyBones.LeftHand
        );


        // ========================================================
        // GAMBA DESTRA
        // ========================================================

        MapBone(
            "rThigh",
            HumanBodyBones.RightUpperLeg
        );

        MapBone(
            "rShin",
            HumanBodyBones.RightLowerLeg
        );

        MapBone(
            "rFoot",
            HumanBodyBones.RightFoot
        );


        // ========================================================
        // GAMBA SINISTRA
        // ========================================================

        MapBone(
            "lThigh",
            HumanBodyBones.LeftUpperLeg
        );

        MapBone(
            "lShin",
            HumanBodyBones.LeftLowerLeg
        );

        MapBone(
            "lFoot",
            HumanBodyBones.LeftFoot
        );


        // ========================================================
        // MANO DESTRA
        // ========================================================

        MapBone(
            "rthumb",
            HumanBodyBones.RightThumbProximal
        );

        MapBone(
            "finger1-2.r",
            HumanBodyBones.RightThumbIntermediate
        );

        MapBone(
            "finger1-3.r",
            HumanBodyBones.RightThumbDistal
        );


        MapBone(
            "finger2-1.r",
            HumanBodyBones.RightIndexProximal
        );

        MapBone(
            "finger2-2.r",
            HumanBodyBones.RightIndexIntermediate
        );

        MapBone(
            "finger2-3.r",
            HumanBodyBones.RightIndexDistal
        );


        MapBone(
            "finger3-1.r",
            HumanBodyBones.RightMiddleProximal
        );

        MapBone(
            "finger3-2.r",
            HumanBodyBones.RightMiddleIntermediate
        );

        MapBone(
            "finger3-3.r",
            HumanBodyBones.RightMiddleDistal
        );


        MapBone(
            "finger4-1.r",
            HumanBodyBones.RightRingProximal
        );

        MapBone(
            "finger4-2.r",
            HumanBodyBones.RightRingIntermediate
        );

        MapBone(
            "finger4-3.r",
            HumanBodyBones.RightRingDistal
        );


        MapBone(
            "finger5-1.r",
            HumanBodyBones.RightLittleProximal
        );

        MapBone(
            "finger5-2.r",
            HumanBodyBones.RightLittleIntermediate
        );

        MapBone(
            "finger5-3.r",
            HumanBodyBones.RightLittleDistal
        );


        // ========================================================
        // MANO SINISTRA
        // ========================================================

        MapBone(
            "lthumb",
            HumanBodyBones.LeftThumbProximal
        );

        MapBone(
            "finger1-2.l",
            HumanBodyBones.LeftThumbIntermediate
        );

        MapBone(
            "finger1-3.l",
            HumanBodyBones.LeftThumbDistal
        );


        MapBone(
            "finger2-1.l",
            HumanBodyBones.LeftIndexProximal
        );

        MapBone(
            "finger2-2.l",
            HumanBodyBones.LeftIndexIntermediate
        );

        MapBone(
            "finger2-3.l",
            HumanBodyBones.LeftIndexDistal
        );


        MapBone(
            "finger3-1.l",
            HumanBodyBones.LeftMiddleProximal
        );

        MapBone(
            "finger3-2.l",
            HumanBodyBones.LeftMiddleIntermediate
        );

        MapBone(
            "finger3-3.l",
            HumanBodyBones.LeftMiddleDistal
        );


        MapBone(
            "finger4-1.l",
            HumanBodyBones.LeftRingProximal
        );

        MapBone(
            "finger4-2.l",
            HumanBodyBones.LeftRingIntermediate
        );

        MapBone(
            "finger4-3.l",
            HumanBodyBones.LeftRingDistal
        );


        MapBone(
            "finger5-1.l",
            HumanBodyBones.LeftLittleProximal
        );

        MapBone(
            "finger5-2.l",
            HumanBodyBones.LeftLittleIntermediate
        );

        MapBone(
            "finger5-3.l",
            HumanBodyBones.LeftLittleDistal
        );


        Debug.Log(
            "========== AVATAR BONE MAP =========="
        );


        foreach (var kvp in boneMap)
        {
            Debug.Log(
                "[MAP OK] " +
                kvp.Key +
                " -> " +
                kvp.Value.name
            );
        }


        Debug.Log(
            "Totale ossa mappate: " +
            boneMap.Count
        );


        Debug.Log(
            "====================================="
        );
    }


    // ============================================================
    // MAP SINGLE BONE
    // ============================================================

    private void MapBone(
        string samName,
        HumanBodyBones humanoidBone)
    {
        if (animator == null)
            return;


        Transform boneTransform =
            animator.GetBoneTransform(
                humanoidBone
            );


        if (boneTransform != null)
        {
            boneMap[samName] =
                boneTransform;


            if (debugMapping)
            {
                Debug.Log(
                    "[MAP OK] " +
                    samName +
                    " -> " +
                    boneTransform.name
                );
            }
        }
        else
        {
            if (debugMapping)
            {
                /*
                Debug.LogWarning(
                    "[MAP FAIL] " +
                    samName +
                    " -> " +
                    humanoidBone +
                    " NON assegnato."
                );
                */
            }
        }
    }


    // ============================================================
    // UPDATE
    // ============================================================

    void Update()
    {
        if (isCalibrated)
            return;


        calibrationTimer +=
            Time.deltaTime;


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
                "⚠️ Nessun dato ricevuto da Python. " +
                "Riprovo tra 1 secondo..."
            );


            calibrationTimer =
                calibrationDelay - 1f;


            return;
        }


        // --------------------------------------------------------
        // Calibrazione rotazioni
        // --------------------------------------------------------

        calibrationRotations.Clear();


        foreach (var kvp in lastRawRotation)
        {
            calibrationRotations[kvp.Key] =
                kvp.Value;
        }


        // --------------------------------------------------------
        // Calibrazione posizione
        // --------------------------------------------------------

        calibrationRootPosition =
            lastRootPosition;


        isCalibrated = true;


        // --------------------------------------------------------
        // Inizializza target
        // --------------------------------------------------------

        targetRotations.Clear();

        filteredRotations.Clear();


        foreach (var kvp in boneMap)
        {
            string samName =
                kvp.Key;


            if (!calibrationRotations.TryGetValue(
                    samName,
                    out Quaternion calibration))
            {
                continue;
            }


            if (!bindRotations.TryGetValue(
                    samName,
                    out Quaternion bind))
            {
                continue;
            }


            Quaternion target =
                bind *
                (
                    Quaternion.Inverse(
                        calibration
                    ) *
                    calibration
                );


            targetRotations[samName] =
                target;


            filteredRotations[samName] =
                target;
        }


        Debug.Log(
            "📍 Root position calibrata: " +
            calibrationRootPosition
        );


        Debug.Log(
            "📍 Avatar initial position: " +
            avatarInitialPosition
        );


        Debug.Log(
            "✅ CALIBRAZIONE COMPLETATA: " +
            calibrationRotations.Count +
            " giunti."
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


        MotionPayload motionData =
            null;


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
                "❌ ERRORE DESERIALIZZAZIONE JSON: " +
                e.Message
            );

            return;
        }


        if (motionData == null)
            return;


        // ========================================================
        // ROOT POSITION
        // ========================================================

        if (motionData.root_position != null)
        {
            lastRootPosition =
                new Vector3(
                    motionData.root_position.x,
                    motionData.root_position.y,
                    motionData.root_position.z
                );
        }


        // ========================================================
        // JOINT 3D
        // ========================================================

        if (
            motionData.joint_xyz_3d != null &&
            motionData.joint_xyz_3d.Count >= 72
        )
        {
            for (int i = 0; i < 24; i++)
            {
                int index =
                    i * 3;


                lastJoints3D[i] =
                    new Vector3(
                        motionData.joint_xyz_3d[index],
                        motionData.joint_xyz_3d[index + 1],
                        -motionData.joint_xyz_3d[index + 2]
                    );
            }
        }


        // ========================================================
        // ROOT MOVEMENT
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


            Vector3 delta =
                currentRoot -
                calibrationRootPosition;


            delta *=
                rootPositionScale;


            if (delta.magnitude >
                maxRootDistance)
            {
                delta =
                    delta.normalized *
                    maxRootDistance;
            }


            Vector3 targetPosition =
                avatarInitialPosition +
                delta;


            float positionT =
                1f -
                Mathf.Exp(
                    -positionSmoothing *
                    Time.deltaTime
                );


            transform.position =
                Vector3.Lerp(
                    transform.position,
                    targetPosition,
                    positionT
                );
        }


        // ========================================================
        // ROTAZIONI
        // ========================================================

        if (
            motionData.unity_rotations_deg ==
            null
        )
        {
            return;
        }


        foreach (
            var item
            in motionData.unity_rotations_deg
        )
        {
            string samName =
                item.Key;


            // ----------------------------------------------------
            // Debug single bone
            // ----------------------------------------------------

            if (
                debugSingleBone &&
                samName != debugBoneName
            )
            {
                continue;
            }


            // ----------------------------------------------------
            // Dati validi?
            // ----------------------------------------------------

            if (item.Value == null)
                continue;


            // ----------------------------------------------------
            // Python/BVH → Unity
            //
            // Il C++ attualmente invia Euler XYZ.
            // ----------------------------------------------------

            Quaternion raw =
                Quaternion.Euler(
                    item.Value.x,
                    item.Value.y,
                    item.Value.z
                );


            lastRawRotation[samName] =
                raw;


            if (receivedBoneNames.Add(
                samName))
            {
                if (debugMapping)
                {
                    /*
                    Debug.Log(
                        "[POSE BONE] " +
                        samName
                    );
                    */
                }
            }


            // ----------------------------------------------------
            // Prima della calibrazione
            // ----------------------------------------------------

            if (!isCalibrated)
                continue;


            // ----------------------------------------------------
            // Conversione nome
            // ----------------------------------------------------

            string unityBoneName =
                ConvertSamBoneName(
                    samName
                );


            if (unityBoneName == null)
            {
                if (
                    debugMapping &&
                    warnedMissingBones.Add(
                        "CONVERT_" + samName
                    )
                )
                {
                    /*
                    Debug.LogWarning(
                        "[MAP FAIL] Ricevuto '" +
                        samName +
                        "' senza conversione."
                    );
                    */
                }

                continue;
            }


            // ----------------------------------------------------
            // Trova bone
            // ----------------------------------------------------

            if (
                !boneMap.TryGetValue(
                    unityBoneName,
                    out Transform boneTransform
                )
            )
            {
                if (
                    debugMapping &&
                    warnedMissingBones.Add(
                        "BONE_" + unityBoneName
                    )
                )
                {
                    /*
                    Debug.LogWarning(
                        "[MAP FAIL] Conversione '" +
                        samName +
                        "' -> '" +
                        unityBoneName +
                        "' ma bone Unity non trovata."
                    );
                    */
                }

                continue;
            }


            // ----------------------------------------------------
            // Calibrazione
            // ----------------------------------------------------

            if (
                !calibrationRotations.TryGetValue(
                    samName,
                    out Quaternion calibration
                )
            )
            {
                if (
                    debugMapping &&
                    warnedMissingBones.Add(
                        "CAL_" + samName
                    )
                )
                {
                    Debug.LogWarning(
                        "[CALIBRATION FAIL] " +
                        "Nessuna calibrazione per: " +
                        samName
                    );
                }

                continue;
            }


            // ----------------------------------------------------
            // Bind rotation
            // ----------------------------------------------------

            if (
                !bindRotations.TryGetValue(
                    unityBoneName,
                    out Quaternion bindRotation
                )
            )
            {
                if (
                    debugMapping &&
                    warnedMissingBones.Add(
                        "BIND_" + unityBoneName
                    )
                )
                {
                    Debug.LogWarning(
                        "[BIND FAIL] Nessuna bind rotation per: " +
                        unityBoneName
                    );
                }

                continue;
            }


            // ----------------------------------------------------
            // Delta dalla calibrazione
            // ----------------------------------------------------

            Quaternion delta =
                Quaternion.Inverse(
                    calibration
                ) *
                raw;


            // ----------------------------------------------------
            // Target
            // ----------------------------------------------------

            Quaternion targetRotation =
                bindRotation *
                delta;


            targetRotations[
                unityBoneName
            ] =
                targetRotation;
        }


        // ========================================================
        // INTERPOLAZIONE CONTINUA
        //
        // Questa parte viene eseguita ad ogni frame Unity,
        // anche quando SAM non ha ancora prodotto una nuova posa.
        // ========================================================

        if (!isCalibrated)
            return;


        foreach (
            var kvp
            in targetRotations
        )
        {
            string boneName =
                kvp.Key;


            if (
                !boneMap.TryGetValue(
                    boneName,
                    out Transform boneTransform
                )
            )
            {
                continue;
            }


            Quaternion current =
                boneTransform.localRotation;


            Quaternion target =
                kvp.Value;


            Quaternion smoothed =
                SmoothRotation(
                    current,
                    target,
                    Time.deltaTime
                );


            boneTransform.localRotation =
                smoothed;


            filteredRotations[boneName] =
                smoothed;
        }
    }
}
