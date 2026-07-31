# Guida e Relazione Tecnica: Setup Avatar 3D e Sistema di Ricezione UDP in Unity (Modello SMPL a 24 Giunti)

Questo documento sintetizza l'intero lavoro svolto per la **Fase 1** (importazione e configurazione dell'avatar 3D in Unity) e la **Fase 2** (implementazione dell'infrastruttura di rete UDP, risoluzione dei problemi di binding locale e animazione procedurale tramite retargeting scheletrico). 

L'obiettivo dell'architettura è ricevere in tempo reale i dati di stima di posa (provenienti da framework basati su SMPL come *AlphaPose* e *MMDetection* integrati con *HybrIK*) e animare in modo fluido un modello 3D all'interno del motore Unity.

---

## 1. Configurazione del Modello 3D in Unity (Mixamo X-Bot)

Per far sì che un modello esterno risponda correttamente alle istruzioni matematiche di rotazione delle ossa, è stato configurato lo scheletro nello standard **Humanoid** di Unity.

### 1.1 Download da Mixamo
* **Fonte:** [Mixamo.com](https://www.mixamo.com)
* **Modello scelto:** `X-Bot` (modello neutro ideale per il retargeting e l'analisi biomeccanica).
* **Impostazioni di Download:**
  * **Formato:** FBX Binary (`.fbx`)
  * **Posa:** T-Pose (posizione di riferimento standard con braccia distese e gambe unite, fondamentale per calcolare correttamente le rotazioni locali zero).

### 1.2 Importazione nel Progetto e Setup Scheletrico
1. **Importazione nei file di progetto (`Assets`):** Il file `.fbx` è stato inserito nella finestra **Project** di Unity.
2. **Configurazione "Humanoid" nell'Inspector:**
   * Selezionato il file del modello nella finestra *Project*.
   * Nella finestra **Inspector**, selezionata la scheda **Rig**.
   * Modificato l'**Animation Type** da `Generic` a **`Humanoid`**.
   * Premuto su **Apply**: Unity ha analizzato automaticamente la gerarchia delle ossa mappandole sullo standard interno *HumanBodyBones*.

### 1.3 Posizionamento nella Scena (`Hierarchy`)
* Il modello configurato è stato trascinato nella finestra **Hierarchy**, rendendolo attivo nello spazio 3D della scena.

---

## 2. Architettura del Software e Librerie

Per gestire la decodifica di JSON complessi (dizionari di chiavi testuali con coordinate di rotazione), è stata integrata la libreria **Newtonsoft Json** (Json.NET) tramite il Package Manager di Unity (`com.unity.nuget.newtonsoft-json`).

### Diagramma del Flusso dei Dati
```
+--------------------------+       JSON (UDP : 5065)       +------------------------+
|   Sorgente Dati (AI)     | ----------------------------> |   UDPReceiver.cs       |
|  (Python / C++ / Test)   |                               |  (Thread Background)   |
+--------------------------+                               +------------------------+
                                                                       |
                                                                       v  string latestJSON
                                                           +------------------------+
                                                           |  AvatarController.cs   |
                                                           |  (LateUpdate / Slerp)  |
                                                           +------------------------+
                                                                       |
                                                                       v
                                                           +------------------------+
                                                           |   X-Bot (Avatar 3D)    |
                                                           +------------------------+
```

---

## 3. Implementazione Dettagliata degli Script C# (Con Correzioni)

### 3.1 Script 1: `UDPReceiver.cs` (Gestore di Rete e Correzioni di Binding)

Questo componente ascolta la porta di rete locale (`5065`), riceve i pacchetti UDP in arrivo e memorizza l'ultimo messaggio JSON in modo sicuro.

#### Note di Risoluzione Problemi (Troubleshooting)
1. **Errore di compilazione iniziale (`CS0117`):** La proprietà `.Any` non appartiene a `IPEndPoint`, bensì a `IPAddress`. Sostituito `IPEndPoint.Any` con `IPAddress.Any`.
2. **Ottimizzazione per il traffico locale (Loopback):** Per evitare che il sistema operativo blocchi il traffico UDP in locale a causa di restrizioni di rete o firewall, il socket è stato configurato per associarsi esplicitamente all'indirizzo di loopback `127.0.0.1`.

#### Codice Sorgente Definitivo e Funzionante
```csharp
using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class UDPReceiver : MonoBehaviour
{
    private Thread receiveThread;
    private UdpClient client;
    public int port = 5065; 
    
    [HideInInspector] public string latestJSON = "";
    private readonly object lockObject = new object();
    private bool running = true;

    void Start()
    {
        InitializeServer();
    }

    private void InitializeServer()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log("Thread UDP avviato in ascolto su 127.0.0.1:" + port);
    }

    private void ReceiveData()
    {
        try
        {
            // Binding esplicito sull'indirizzo di loopback locale per garantire la ricezione dei pacchetti di test
            IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);
            client = new UdpClient(localEndPoint);
            
            IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

            while (running)
            {
                byte[] data = client.Receive(ref remoteEndPoint);
                string text = Encoding.UTF8.GetString(data);
                
                lock (lockObject)
                {
                    latestJSON = text;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Errore critico nel thread UDP: " + e.Message);
        }
    }

    void OnDisable()
    {
        running = false;
        if (client != null) client.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
    }
}
```

#### Concetti Architetturali Chiave
* **Multithreading:** L'ascolto UDP in blocco viene eseguito su un thread separato in background per preservare la fluidità di rendering di Unity (60+ FPS).
* **Thread-Safety:** L'utilizzo del blocco `lock (lockObject)` previene race conditions tra il thread di rete e il thread principale di Unity.

---

### 3.2 Script 2: `AvatarController.cs` (Retargeting e Animazione)

Mappa i giunti del formato SMPL (24 giunti) alle ossa Humanoid di Unity e applica le rotazioni con interpolazione sferica.

#### Codice Sorgente Definitivo
```csharp
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
    private Dictionary<string, Transform> boneMap = new Dictionary<string, Transform>();

    public class JointData
    {
        public float x { get; set; }
        public float y { get; set; }
        public float z { get; set; }
    }

    public class MotionPayload
    {
        public Dictionary<string, JointData> unity_rotations_deg { get; set; }
        public JointData root_position { get; set; }
    }

    void Start()
    {
        animator = GetComponent<Animator>();
        InitializeBoneMapping();
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

        MapBone("R_Hip", HumanBodyBones.RightUpperLeg);
        MapBone("R_Knee", HumanBodyBones.RightLowerLeg);
        MapBone("R_Ankle", HumanBodyBones.RightFoot);
    }

    private void MapBone(string smplName, HumanBodyBones humanoidBone)
    {
        Transform boneTransform = animator.GetBoneTransform(humanoidBone);
        if (boneTransform != null)
        {
            boneMap[smplName] = boneTransform;
        }
    }

    void LateUpdate()
    {
        if (udpReceiver == null || string.IsNullOrEmpty(udpReceiver.latestJSON))
            return;

        MotionPayload motionData = null;

        try
        {
            motionData = JsonConvert.DeserializeObject<MotionPayload>(udpReceiver.latestJSON);
        }
        catch
        {
            return;
        }

        if (motionData == null) return;

        if (motionData.root_position != null && boneMap.ContainsKey("Pelvis"))
        {
            Vector3 targetPos = new Vector3(
                motionData.root_position.x,
                motionData.root_position.y,
                motionData.root_position.z
            );
            boneMap["Pelvis"].localPosition = Vector3.Lerp(
                boneMap["Pelvis"].localPosition,
                targetPos,
                Time.deltaTime * smoothing
            );
        }

        if (motionData.unity_rotations_deg != null)
        {
            foreach (var item in motionData.unity_rotations_deg)
            {
                if (boneMap.TryGetValue(item.Key, out Transform boneTransform))
                {
                    Quaternion targetRotation = Quaternion.Euler(item.Value.x, item.Value.y, item.Value.z);
                    boneTransform.localRotation = Quaternion.Slerp(
                        boneTransform.localRotation,
                        targetRotation,
                        Time.deltaTime * smoothing
                    );
                }
            }
        }
    }
}
```

---

## 4. Tabella Mappatura Biomeccanica SMPL -> Unity

| Nome Giunto SMPL | Osso Humanoid Unity (`HumanBodyBones`) | Descrizione Biomeccanica |
| :--- | :--- | :--- |
| `Pelvis` | `Hips` | Bacino / Centro di massa |
| `Spine1` / `Spine2` / `Spine3` | `Spine` / `Chest` / `UpperChest` | Colonna vertebrale e Torace |
| `Neck` / `Head` | `Neck` / `Head` | Collo e Testa |
| `L_Shoulder` / `R_Shoulder` | `LeftUpperArm` / `RightUpperArm` | Braccio superiore |
| `L_Elbow` / `R_Elbow` | `LeftLowerArm` / `RightLowerArm` | Avambraccio |
| `L_Wrist` / `R_Wrist` | `LeftHand` / `RightHand` | Polso / Mano |
| `L_Hip` / `R_Hip` | `LeftUpperLeg` / `RightUpperLeg` | Coscia |
| `L_Knee` / `R_Knee` | `LeftLowerLeg` / `RightLowerLeg` | Gamba |
| `L_Ankle` / `R_Ankle` | `LeftFoot` / `RightFoot` | Piede |

---

## 5. Esito del Test di Simulazione Interna (Confermato)

1. **Test Eseguito:** Tramite l'aggiunta di un componente `UDPSimulator` temporaneo, sono stati inviati pacchetti JSON di prova in locale sulla porta `5065`.
2. **Risultato:** L'avatar `X-Bot` ha risposto correttamente alla ricezione dei pacchetti, muovendo la testa e le braccia in tempo reale.
3. **Stato Attuale:** La pipeline di ricezione UDP, decodifica JSON e retargeting scheletrico Humanoid è **completamente funzionante e validata**.
