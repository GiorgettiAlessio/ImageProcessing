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

    [Header("Correzione Root (Pelvis)")]
    [Tooltip(
        "Le rotazioni delle ossa figlie sono relative al Pelvis: se il " +
        "Pelvis ha un offset di convenzione (SMPL/HybrIK vs Unity) sbagliato, " +
        "trascina con sé l'intero corpo pur restando internamente coerente " +
        "(braccia/gambe nella posa giusta ma corpo capovolto/ruotato). " +
        "Default 180° su X: se dopo il fix l'avatar risulta ancora ruotato " +
        "(es. capovolto lateralmente invece che testa in giù), prova " +
        "(0,180,0) o (0,0,180) qui in Play Mode finché non torna dritto."
    )]
    public Vector3 rootRotationOffsetEuler = new Vector3(180f, 0f, 0f);

    [Header("Debug")]
    public bool debugSingleBone = false;
    public string debugBoneName = "R_Shoulder";

    private Vector3 lastRootPosition;
    private Animator animator;

    private Dictionary<string, Transform> boneMap = new Dictionary<string, Transform>();
    private Dictionary<string, Quaternion> bindRotations = new Dictionary<string, Quaternion>();
    private Dictionary<string, Quaternion> calibrationRotations = new Dictionary<string, Quaternion>();
    private Vector3[] lastJoints3D = new Vector3[24];
    private bool isCalibrated = false;

    // Posizione HybrIK al momento della calibrazione
    private Vector3 calibrationRootPosition;
    // Posizione originale dell'avatar in Unity
    private Vector3 avatarInitialPosition;

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
            Debug.LogError("❌ AvatarController: Animator NON trovato!");
            return;
        }

        if (udpReceiver == null)
            udpReceiver = FindObjectOfType<UDPReceiver>();

        if (udpReceiver == null)
        {
            Debug.LogError("❌ AvatarController: UDPReceiver NON trovato!");
        }

        InitializeBoneMapping();

        foreach (var kvp in boneMap)
        {
            bindRotations[kvp.Key] = kvp.Value.localRotation;
        }

        avatarInitialPosition = transform.position;

        Debug.Log("✅ AvatarController avviato. In attesa del primo pacchetto UDP per la calibrazione automatica...");
    }

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
    }

    // Ossa per cui la Z locale del rig porta il vero movimento "hinge"
    // (alza/piega) invece della X: per queste la conversione destrorso ->
    // sinistrorso va fatta negando X (mantenendo Z) invece del solito
    // negando Z. Individuato osservando in Play Mode che ruotando il
    // braccio la Rotation.X di RightShoulder restava quasi ferma (101°-106°)
    // mentre Z variava parecchio (-101° a -62°) — vedi conversazione.
    private static readonly HashSet<string> AxisSwapBones = new HashSet<string>
    {
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow"
    };

    private Quaternion ConvertToUnityQuaternion(string boneName, JointData v)
    {
        if (AxisSwapBones.Contains(boneName))
        {
            return new Quaternion(-v.x, v.y, v.z, v.w);
        }

        return new Quaternion(v.x, v.y, -v.z, v.w);
    }

    private void MapBone(string samName, HumanBodyBones humanoidBone)
    {
        if (animator == null) return;
        Transform boneTransform = animator.GetBoneTransform(humanoidBone);
        if (boneTransform != null)
        {
            boneMap[samName] = boneTransform;
        }
    }

    private Quaternion SmoothRotation(Quaternion current, Quaternion target, float deltaTime)
    {
        float angle = Quaternion.Angle(current, target);

        if (angle < rotationDeadZone)
            return current;

        float normalizedAngle = Mathf.Clamp01(angle / 45f);
        float adaptiveSmoothing = Mathf.Lerp(smoothing, fastSmoothing, normalizedAngle);
        float t = 1f - Mathf.Exp(-adaptiveSmoothing * deltaTime);

        return Quaternion.Slerp(current, target, t);
    }

    void Update()
    {
        // Nessun input da tastiera necessario: la calibrazione avviene in automatico al primo pacchetto UDP.
    }

    void LateUpdate()
    {
        if (udpReceiver == null) return;
        if (string.IsNullOrEmpty(udpReceiver.latestJSON)) return;

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

        if (motionData == null) return;

        if (motionData.root_position != null)
        {
            lastRootPosition = new Vector3(
                motionData.root_position.x,
                motionData.root_position.y,
                motionData.root_position.z
            );
        }

        if (motionData.unity_rotations_deg == null)
            return;

        // CALIBRAZIONE AUTOMATICA AL PRIMO FRAME VALIDO
        if (!isCalibrated)
        {
            calibrationRotations.Clear();
            foreach (var item in motionData.unity_rotations_deg)
            {
                Quaternion rawInit = ConvertToUnityQuaternion(item.Key, item.Value);
                calibrationRotations[item.Key] = rawInit;
            }

            if (motionData.root_position != null)
            {
                calibrationRootPosition = new Vector3(
                    motionData.root_position.x,
                    motionData.root_position.y,
                    motionData.root_position.z
                );
            }

            isCalibrated = true;
            Debug.Log("✅ Calibrazione automatica eseguita sul primo frame UDP!");
        }

        // ROOT POSITION
        if (motionData.root_position != null)
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

        // ROTAZIONI
        foreach (var item in motionData.unity_rotations_deg)
        {
            if (debugSingleBone && item.Key != debugBoneName)
            {
                continue;
            }

            Quaternion raw = ConvertToUnityQuaternion(item.Key, item.Value);

            if (!boneMap.TryGetValue(item.Key, out Transform boneTransform))
            {
                continue;
            }

            Quaternion delta = Quaternion.identity;

            if (calibrationRotations.TryGetValue(item.Key, out Quaternion calibration))
            {
                delta = Quaternion.Inverse(calibration) * raw;
            }
            else
            {
                delta = raw;
            }

            // IMPORTANTE: il delta calibrato va applicato SOPRA la rotazione
            // di bind (T-pose) dell'osso, non al suo posto. bindRotations
            // viene popolato in Start() ma prima non veniva mai riletto qui:
            // il risultato era che ogni osso finiva vicino a
            // Quaternion.identity (perché al calibration frame delta ≈
            // identità), e su un rig Mixamo/X-Bot con assi locali non
            // allineati "identità" produce la posa accartocciata, non la
            // T-pose.
            Quaternion targetLocalRotation = delta;
            if (bindRotations.TryGetValue(item.Key, out Quaternion bindRotation))
            {
                if (item.Key == "Pelvis")
                {
                    // Il Pelvis è la radice: tutte le ossa figlie sono
                    // relative a lui, quindi se SOLO il suo orientamento
                    // assoluto ha un offset di convenzione (es. asse Y
                    // camera-space vs Y-up Unity) l'intero corpo appare
                    // capovolto pur restando internamente coerente — il
                    // sintomo esatto osservato (arti nella posa giusta,
                    // corpo intero ruotato di 180°). Vedi tooltip su
                    // rootRotationOffsetEuler per come tararlo se 180° su X
                    // non fosse l'asse giusto per il tuo setup.
                    Quaternion rootOffset = Quaternion.Euler(rootRotationOffsetEuler);
                    targetLocalRotation = bindRotation * (rootOffset * delta);
                }
                else
                {
                    targetLocalRotation = bindRotation * delta;
                }
            }

            boneTransform.localRotation = SmoothRotation(
                boneTransform.localRotation,
                targetLocalRotation,
                Time.deltaTime
            );
        }
    }
}