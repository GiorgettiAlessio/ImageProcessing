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
---
# Guida e Relazione Tecnica: Setup Avatar 3D e Sistema di Ricezione UDP in Unity

Questo documento costituisce la relazione tecnica completa del progetto, divisa cronologicamente: parte dalla configurazione e dalla versione iniziale della pipeline (modelli limitati a 24 giunti) e aggiunge **in coda** la nuova evoluzione e l'analisi di retrocompatibilità per il supporto full-body con **Fast-SAM-3D-Body**.

---

## PARTE 1: Configurazione Iniziale e Versione Base (Modelli SMPL / 24 Giunti)

### 1. Configurazione del Modello 3D in Unity (Mixamo X-Bot)
Per consentire a un modello 3D di rispondere correttamente alle rotazioni calcolate dagli algoritmi di intelligenza artificiale, è stato configurato lo scheletro nello standard **Humanoid** di Unity.
* **Modello:** `X-Bot` (Mixamo) in T-Pose.
* **Rig:** Tipo di animazione impostato su **Humanoid** nella scheda *Rig* dell'Inspector, consentendo a Unity di mappare automaticamente la gerarchia anatomica interna (`HumanBodyBones`).

### 2. Architettura di Rete e Ricezione UDP Base
Il sistema si basa su una comunicazione **UDP asincrona in tempo reale**:
* **Thread Separato (`UDPReceiver.cs`):** Mette in ascolto un socket sulla porta `5065` (con binding esplicito su `127.0.0.1` per garantire il corretto loopback locale) evitando blocchi sul thread di rendering principale di Unity.
* **Thread-Safety:** L'accesso alla stringa JSON condivisa è protetto da un costrutto `lock` per evitare *race conditions*.

#### Script Base 1: `UDPReceiver.cs`
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

---

## PARTE 2: Evoluzione e Integrazione (Fast-SAM-3D-Body Full-Body e Retrocompatibilità)

*(Aggiunta successiva per il supporto avanzato di mani, dita e piedi)*

### 1. Confronto tra la Versione Precedente e Fast-SAM-3D-Body
* **Versione Precedente (AlphaPose / MMDetection + HybrIK):**
  * *Copertura:* Limitata principalmente a tronco, testa e ai giunti principali di arti superiori e inferiori (circa 24 giunti).
  * *Limitazioni:* Incapacità di tracciare le articolazioni fini di mani, dita e piedi, rendendo la pipeline inadatta a compiti di teleoperazione avanzata di precisione.
* **Nuova Versione (Fast-SAM-3D-Body / Modello MHR):**
  * *Copertura:* **Full-body ad altissima precisione**, che estende la mappatura includendo in modo dettagliato le falangi di tutte le dita di entrambe le mani e la struttura dei piedi.

### 2. Gestione della Retrocompatibilità nel Codice
Il controller aggiornato non richiede di riscrivere l'architettura di rete, ma introduce un'iterazione dinamica sulle chiavi presenti nel pacchetto JSON ricevuto:
1. **Scansione selettiva:** Il ciclo elabora unicamente le chiavi trovate all'interno del dizionario `unity_rotations_deg` del payload.
2. **Compatibilità con i pacchetti vecchi:** Se il server invia un pacchetto legacy con soli 24 giunti, il sistema aggiorna solo quelli, lasciando inalterate le dita e i piedi senza generare eccezioni.
3. **Robustezza:** Grazie ai controlli condizionali sull'esistenza dei trasform (`if (boneTransform != null)`), eventuali modelli privi di dita funzionano comunque senza causare crash.

### 3. Script Aggiornato: `AvatarController.cs` (Full-Body & Retrocompatibile)
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
        // === 1. Tronco e Testa ===
        MapBone("Pelvis", HumanBodyBones.Hips);
        MapBone("Spine1", HumanBodyBones.Spine);
        MapBone("Spine2", HumanBodyBones.Chest);
        MapBone("Spine3", HumanBodyBones.UpperChest);
        MapBone("Neck", HumanBodyBones.Neck);
        MapBone("Head", HumanBodyBones.Head);

        // === 2. Arti Superiori ===
        MapBone("L_Collar", HumanBodyBones.LeftShoulder);
        MapBone("L_Shoulder", HumanBodyBones.LeftUpperArm);
        MapBone("L_Elbow", HumanBodyBones.LeftLowerArm);
        MapBone("L_Wrist", HumanBodyBones.LeftHand);

        MapBone("R_Collar", HumanBodyBones.RightShoulder);
        MapBone("R_Shoulder", HumanBodyBones.RightUpperArm);
        MapBone("R_Elbow", HumanBodyBones.RightLowerArm);
        MapBone("R_Wrist", HumanBodyBones.RightHand);

        // === 3. Arti Inferiori ===
        MapBone("L_Hip", HumanBodyBones.LeftUpperLeg);
        MapBone("L_Knee", HumanBodyBones.LeftLowerLeg);
        MapBone("L_Ankle", HumanBodyBones.LeftFoot);
        MapBone("L_Toe", HumanBodyBones.LeftToes);

        MapBone("R_Hip", HumanBodyBones.RightUpperLeg);
        MapBone("R_Knee", HumanBodyBones.RightLowerLeg);
        MapBone("R_Ankle", HumanBodyBones.RightFoot);
        MapBone("R_Toe", HumanBodyBones.RightToes);

        // === 4. Mano Sinistra (Dita) ===
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

        // === 5. Mano Destra (Dita) ===
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

    private void MapBone(string jointName, HumanBodyBones humanoidBone)
    {
        if (animator == null) return;
        Transform boneTransform = animator.GetBoneTransform(humanoidBone);
        if (boneTransform != null)
        {
            boneMap[jointName] = boneTransform;
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

### 4. Script di Simulazione Full-Body: `UDPSimulator.cs`
```csharp
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Collections.Generic;
using Newtonsoft.Json;

public class UDPSimulator : MonoBehaviour
{
    private UdpClient client;
    public int port = 5065;
    public string ipAddress = "127.0.0.1";

    void Start()
    {
        client = new UdpClient();
        Debug.Log("Simulatore UDP Full-Body avviato verso " + ipAddress + ":" + port);
    }

    void Update()
    {
        float braccio = Mathf.Sin(Time.time * 3f) * 45f;
        float testa = Mathf.Cos(Time.time * 2f) * 30f;
        float dito = Mathf.Sin(Time.time * 6f) * 20f;

        var payload = new
        {
            unity_rotations_deg = new Dictionary<string, object>
            {
                { "L_Shoulder", new { x = 0f, y = 0f, z = braccio } },
                { "R_Shoulder", new { x = 0f, y = 0f, z = -braccio } },
                { "Head",       new { x = 0f, y = testa,  z = 0f } },
                { "L_Index1",   new { x = dito, y = 0f,   z = 0f } }
            },
            root_position = new { x = 0f, y = 0f, z = 0f }
        };

        string json = JsonConvert.SerializeObject(payload);
        byte[] data = Encoding.UTF8.GetBytes(json);

        try
        {
            client.Send(data, data.Length, ipAddress, port);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Errore invio UDP simulato: " + e.Message);
        }
    }

    void OnDisable()
    {
        if (client != null) client.Close();
    }
}
```