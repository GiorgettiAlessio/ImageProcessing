# SISTEMA DI MOTION CAPTURE IN TEMPO REALE
### Integrazione tra Computer Vision (Python/C++) e Grafica 3D (Unity)

---
#### Valutazione del progetto e scelta dell'approccio:

- **AlphaPose**: Fornisce coordinate spaziali 2D/3D (X, Y, Z) dei giunti.
Come si usa in Unity: Richiede il sistema di Cinematica Inversa (HybridIK). Sposti i punti di destinazione (targets) nello spazio 3D e l'IK calcola come devono piegarsi gomiti e ginocchia.
- **SAM3DBody-cpp (Modello MHR / SMPL)**: Calcola direttamente le rotazioni relative delle articolazioni (Quaternioni/Euler Angles) o file .bvh.
Come si usa in Unity: È l'approccio più pulito ed efficiente. Applicate direttamente le rotazioni alle ossa dell'avatar senza bisogno di IK, eliminando i problemi di snodo innaturale degli arti.

#### Quale approccio scegliere per il tuo lavoro in Unity?
Pianifica una doppia gestione nello script C# Unity:
- Approccio Primario (Rotazioni Dirette / SAM3DBody): Se ti arrivano angoli di rotazione (o dati MHR/SMPL), applicali direttamente ai giunti dell'Avatar Humanoid.
- Approccio Secondario (Coordinate Posizionali + HybridIK): Se ti inviano coordinate X,Y,Z (da AlphaPose), aggiorna dei GameObject vuoti e usa HybridIK per guidare gli arti dell'avatar.

---

## 1. PREPARAZIONE DELL'AMBIENTE E PIPELINE DEL MODELLO 3D IN UNITY

Prima di poter ricevere dati di movimento, è fondamentale strutturare correttamente l'asset tridimensionale all'interno del motore grafico. Lo sviluppo segue una pipeline precisa che va dalla creazione della mesh fino alla configurazione della cinematica.

`[ Creazione in Blender ]`&rarr;`[ Importazione in Unity ]`&rarr;`[ Configurazione Humanoid ]`&rarr;`[ Script C# Ricevitore ]`

### Creazione del Modello: Blender vs Soluzioni Pronte
Non è strettamente necessario modellare un personaggio da zero su Blender, a meno che non si richieda un design altamente personalizzato. Le opzioni principali sono:

* **Blender (Approccio Custom):** Se si crea il modello in Blender, la mesh deve essere sottoposta a *Rigging* (creazione dello scheletro) e *Skinning* (assegnazione dei pesi dei vertici alle ossa). Lo scheletro deve seguire una gerarchia standard (Sito -> Bacino -> Spina -> Spalle -> Arti). Il modello va esportato in formato `.fbx`.
* **Generatori Automatici (Consigliato):** Strumenti come **Ready Player Me** o **MakeHuman** generano istantaneamente modelli 3D completi di texture e scheletri standard perfettamente ottimizzati per i motori di gioco.

### Importazione e Configurazione dell'Umanoide in Unity
Una volta importato il file `.fbx` nell' `Asset Project` di Unity, il motore grafico deve riconoscere l'anatomia dello scheletro:

1. Selezionare il file del modello nella finestra *Project*.
2. Nell' *Inspector*, aprire la scheda **Rig**.
3. Impostare il *Animation Type* su **Humanoid** e cliccare su *Apply*.
4. Unity avvierà una mappatura automatica (Avatar). Cliccando su *Configure*, è possibile verificare che ogni osso virtuale (es. *LeftUpperArm*, *Spine*) corrisponda all'osso corretto del modello 3D. Questa astrazione è fondamentale: permette a qualsiasi script di controllare il modello indipendentemente dalle proporzioni specifiche della mesh.

### Predisposizione della Scena per la Ricezione Dati
Per collegare i dati esterni all'avatar, si crea una struttura a nodi nella scena di Unity:

* **Il Modello 3D (Target):** L'avatar umanoide posizionato nella scena.
* **I Target di Riferimento (GameObject Vuoti):** Si creano dei punti vuoti nella scena (es. `Target_Mano_DX`, `Target_Gomito_DX`) organizzati in modo speculare ai giunti rilevati dal software di visione. Lo script C# aggiornerà le coordinate di questi punti vuoti, e il sistema di animazione guiderà lo scheletro verso di essi.



## 2. ANALISI COMPARATIVA DEI FRAMEWORK DI COMPUTER VISION

La scelta dell'algoritmo di stima della posa determina l'efficacia del tracciamento. Di seguito vengono analizzati i tre approcci principali evidenziandone vantaggi e limiti tecnologici.

### 2.1 AlphaPose
AlphaPose è un sistema di tracciamento multi-persona *Top-Down*. Rileva prima le persone nella scena (Bounding Box) e successivamente stima i punti chiave di ciascun individuo.

* **Architettura Tecnica:** Sfrutta reti neurali profonde (come i moduli di attenzione e i classificatori regionali) per massimizzare la precisione anche in presenza di sovrapposizioni o posture complesse.
* **Limitazioni Real-Time:** L'elevata accuratezza richiede un costo computazionale massiccio. Per girare in tempo reale (30+ FPS), necessita obbligatoriamente di una GPU NVIDIA di fascia alta con supporto CUDA. L'output nativo è spesso bidimensionale, richiedendo algoritmi di *lifting* per proiettare i punti nello spazio 3D.

### 2.2 Framework Basati su Modelli Parametrici (SMPL / Sam3dbody)
Questi modelli non si limitano a trovare i nodi dello scheletro, ma tentano di ricostruire l'intera superficie corporea partendo da una singola immagine.

* **Architettura Tecnica:** Basandosi sul modello matematico SMPL (Skinned Multi-Person Linear Model), l'algoritmo calcola i parametri di forma del corpo e, soprattutto, i vettori di **rotazione relativa delle giunzioni**.
* **Limitazioni Real-Time:** Estrarre una mesh intera a 60 FPS è proibitivo per la maggior parte dei sistemi consumer. Tuttavia, il vantaggio teorico è enorme: inviando a Unity direttamente le *rotazioni* (angoli) invece delle *posizioni*, si elimina la necessità di calcolare la cinematica inversa in Unity, poiché le ossa sanno già come ruotare.



## 3. APPLICAZIONE DELLA CINEMATICA INVERSA (HYBRIDIK) IN UNITY

Quando da Python riceviamo soltanto coordinate spaziali (X, Y, Z), il modello 3D non sa come orientare i propri arti. Se muoviamo solo il polso virtuale, il gomito e la spalla rimarrebbero immobili, distorcendo l'avatar. Per risolvere questo problema si utilizza la **Cinematica Inversa (IK)**, nello specifico il pacchetto **HybridIK**.

### Meccanismo di Funzionamento
A differenza della cinematica diretta (dove si ruota la spalla per muovere la mano), l'IK fa il contrario:

1. Python comunica la posizione 3D della mano.
2. Unity sposta il `Target_Mano_DX` in quella esatta coordinata.
3. Il solutore **HybridIK**, agganciato al braccio dell'avatar, calcola istantaneamente le rotazioni algebriche necessarie per la spalla e il gomito affinché la mano tocchi il target.

### Vantaggi di HybridIK rispetto all'IK Standard
HybridIK combina algoritmi analitici e geometrici per garantire:

* **Vincoli Anatomici:** Impedisce alle articolazioni di compiere rotazioni impossibili per un essere umano (es. l'iperestensione del gomito).
* **Fluidità e Risparmio Computazionale:** Risolve le equazioni matematiche dello scheletro in pochi millisecondi, mantenendo il framerate di Unity stabile.



## 4. ARCHITETTURA DI COMUNICAZIONE E PROTOCOLLO DI STREAMING UDP

Il collegamento tra l'ambiente di computazione (Python) e l'ambiente di rendering (Unity) deve avvenire senza accumulare ritardi. Il protocollo scelto per questa pipeline è **UDP (User Datagram Protocol)**.

|          LATO PYTHON              |       |            LATO UNITY             |
|:----------------------------------|:------|:----------------------------------|
| 1. Cattura Frame (OpenCV)         ||1. Thread UDP in ascolto (Porta)   |
| 2. Inferenza (AlphaPose)          || 2. Parsing stringa (JSON/Byte)    |
| 3. Serializzazione Coordinate     || 3. Assegnazione coordinate nodi   |
| 4. Invio Socket UDP               | --(Rete)&rarr; | 4. Esecuzione HybridIK & Render   |


### Perché UDP e non TCP?
Il protocollo TCP garantisce la consegna di ogni singolo pacchetto tramite meccanismi di controllo e ritrasmissione. Se un pacchetto viene perso, la comunicazione si blocca in attesa del recupero. Nel motion capture in tempo reale, un frame vecchio non ha valore; serve solo l'ultimo dato generato. UDP invia i pacchetti "a flusso continuo" senza verificare la ricezione: se un pacchetto si perde, viene semplicemente scartato a favore di quello successivo, garantendo una **latenza vicina allo zero**.

### Struttura del Flusso Dati (Data Packet)
I dati vengono generalmente serializzati in stringhe strutturate o array di byte per ridurre il peso del pacchetto. Un esempio di payload JSON inviato tramite UDP per un singolo giunto si presenta così:

```json
{
  "joint": "left_wrist",
  "x": 0.452,
  "y": 1.231,
  "z": -0.114
}
```

In Unity, un thread parallelo (Background Worker) rimane in ascolto sulla porta locale impostata (es. `127.0.0.1:7000`), deserializza la stringa e passa i valori numerici al ciclo di aggiornamento grafico principale (`Update`), completando il ciclo di animazione in tempo reale.

---

#### Guida Dettagliata per l'Implementazione su Unity

### PASSO 1: Configurazione dell'Avatar Humanoid in Unity
*già visto sopra &uarr;*

Per fare in modo che Unity sappia muovere il personaggio indipendentemente dalla sua forma specifica:

1. Trascina il file `.fbx` dell'avatar nella finestra **Project** (nella cartella `Assets`).
2. Seleziona il file nell'Assets e apri la scheda **Rig** nella finestra **Inspector** (in alto a destra).
3. Cambia **Animation Type** da *Generic* a **Humanoid**.
4. Clicca su **Apply**.
5. Clicca sul pulsante **Configure...** che appare subito sotto per verificare la mappatura delle ossa (dovrebbero apparire tutte in verde). Se qualche osso manca, trascinalo manualmente nello schema.



### PASSO 2: Creazione dello Script Ricevitore UDP in C#

In Unity, l'interfaccia grafica (UI e posizionamento oggetti) gira sul **Main Thread**. I dati UDP arrivano invece in continuo su un **Thread secondario**. Lo script deve ricevere i dati via UDP su un thread separato e salvarli in una variabile sicura, che verrà poi letta nell'evento `Update()` per muovere il personaggio.

Crea uno script C# chiamato `UDPReceiver.cs` e assegnalo a un GameObject vuoto nella scena (chiamalo `[UDP_Manager]`):

```csharp
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UDPReceiver : MonoBehaviour
{
    [Header("Configurazione Rete")]
    public int port = 7000;
    
    private UdpClient client;
    private Thread receiveThread;
    
    // Stringa JSON grezza ricevuta
    [HideInInspector]
    public string latestData = "";
    private readonly object lockObject = new object();

    void Start()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);

                lock (lockObject)
                {
                    latestData = text;
                }
            }
            catch (Exception err)
            {
                Debug.LogError("Errore UDP: " + err.ToString());
            }
        }
    }

    public string GetLatestData()
    {
        lock (lockObject)
        {
            return latestData;
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Abort();
        if (client != null)
            client.Close();
    }
}
```



### PASSO 3: Mappatura dei Dati sull'Avatar (Rotazioni o IK)

Crea una struttura dati compatibile con il formato JSON. Ad esempio, se inviano un pacchetto contenente posizioni/rotazioni dei giunti:

#### Definizione delle classi per la Deserializzazione JSON:

```csharp
[System.Serializable]
public class JointData
{
    public string joint_name;
    public float x;
    public float y;
    public float z;
    public float rw; // Rotazione quaternione (opzionale)
    public float rx;
    public float ry;
    public float rz;
}

[System.Serializable]
public class PosePacket
{
    public int person_id;
    public JointData[] joints;
}
```

#### Script di Controllo Avatar (`AvatarPoseController.cs`):
Collega questo script direttamente al Game Object del tuo **Avatar 3D**.

```csharp
using UnityEngine;

public class AvatarPoseController : MonoBehaviour
{
    public UDPReceiver udpReceiver;
    public Animator animator; // Componente Animator dell'Avatar Humanoid

    void Start()
    {
        if (animator == null)
            animator = GetComponent<Animator>();
    }

    void Update()
    {
        string json = udpReceiver.GetLatestData();
        if (string.IsNullOrEmpty(json)) return;

        try
        {
            PosePacket packet = JsonUtility.FromJson<PosePacket>(json);
            ApplyPose(packet);
        }
        catch (System.Exception e)
        {
            // Gestione eventuali errori di parsing
        }
    }

    void ApplyPose(PosePacket packet)
    {
        foreach (var j in packet.joints)
        {
            // Converti il nome del giunto nell'osso dell'Avatar Humanoid di Unity
            HumanBodyBones bone = MapJointToBone(j.joint_name);
            if (bone != HumanBodyBones.LastBone)
            {
                Transform boneTransform = animator.GetBoneTransform(bone);
                if (boneTransform != null)
                {
                    // Se ricevi rotazioni (Quaternioni):
                    Quaternion targetRotation = new Quaternion(j.rx, j.ry, j.rz, j.rw);
                    boneTransform.localRotation = Quaternion.Slerp(boneTransform.localRotation, targetRotation, Time.deltaTime * 15f);
                }
            }
        }
    }

    HumanBodyBones MapJointToBone(string jointName)
    {
        switch (jointName)
        {
            case "left_shoulder": return HumanBodyBones.LeftUpperArm;
            case "left_elbow": return HumanBodyBones.LeftLowerArm;
            case "left_wrist": return HumanBodyBones.LeftHand;
            case "right_shoulder": return HumanBodyBones.RightUpperArm;
            case "right_elbow": return HumanBodyBones.RightLowerArm;
            case "right_wrist": return HumanBodyBones.RightHand;
            case "hip": return HumanBodyBones.Hips;
            // Aggiungi le altre mappature in base al dataset scelto
            default: return HumanBodyBones.LastBone;
        }
    }
}
```


### PASSO 4: Integrazione di HybridIK (Se si usano le Posizioni X, Y, Z)

Se il modello Python invia solo posizioni X, Y, Z anziché rotazioni:

1. **Importa il package HybridIK** dentro Unity (`Assets -> Import Package -> Custom Package`).
2. Crea dei **GameObject Vuoti** nella scena che rappresentino i target (es. `Target_LeftHand`, `Target_RightHand`).
3. Applica il componente **HybridIK Solver** sul braccio dell'Avatar:
   * Imposta come **Root** la spalla (`LeftUpperArm`).
   * Imposta come **Tip** la mano (`LeftHand`).
   * Assegna il GameObject `Target_LeftHand` come **Target**.
4. Nello script C#, anziché ruotare direttamente l'osso, aggiorna semplicemente la posizione del `Target_LeftHand`:

```csharp
targetLeftHand.transform.position = Vector3.Lerp(
    targetLeftHand.transform.position, 
    new Vector3(j.x, j.y, j.z), 
    Time.deltaTime * 15f
);
```

---

# Suggerimenti sulle estensioni del progetto

1. **Gestione di più persone nella scena:**
   * In `UDPReceiver`, quando ricevi un pacchetto con un `person_id` nuovo, fai uno `Instantiate()` dinamico di un nuovo Prefab dell'Avatar nella scena.
   * Mantieni un `Dictionary<int, GameObject>` in C# per collegare ogni `person_id` al suo specifico avatar 3D.
2. **Interazione dell'avatar con oggetti e ambiente:**
   * Usa il sistema di **RigidBody** e **Colliders** di Unity sugli arti dell'Avatar.
   * Sfrutta **HybridIK con Physics Interaction** per fare in modo che quando la mano dell'avatar si avvicina a un oggetto (es. una palla), l'IK si blocchi sulla superficie dell'oggetto senza compenetrarlo.

---
# Approfondimento Estensioni: Multi-Person Tracking e Interazione Fisica


## ESTENSIONE 1: Gestione di più persone nella scena (Multi-Person Tracking)

Quando la telecamera (YOLO + AlphaPose/SAM3DBody) rileva più persone, lo stream JSON invierà pacchetti con `person_id` diversi (es. 0, 1, 2...). In Unity, è necessario **generare (Instantiate) dinamicamente** gli avatar quando entra una nuova persona nell'inquadratura e **distruggerli** quando escono.

### La Logica:
Utilizzeremo un `Dictionary<int, GameObject>`. Il dizionario assocerà ogni `person_id` (la chiave) al suo corrispondente Avatar 3D instanziato (il valore). 

### Come configurare Unity:
1. Prendi il tuo Avatar 3D già configurato (con il suo `AvatarPoseController` collegato) dalla scena e trascinalo nella cartella **Project**. Questo creerà un **Prefab** (un modello base riutilizzabile).
2. Elimina l'Avatar dalla scena (verrà generato via codice).
3. Crea un nuovo script chiamato `MultiAvatarManager.cs` e assegnalo al GameObject `[UDP_Manager]`.

### Lo Script `MultiAvatarManager.cs`:

```csharp
using System.Collections.Generic;
using UnityEngine;

public class MultiAvatarManager : MonoBehaviour
{
    [Header("Riferimenti")]
    public UDPReceiver udpReceiver;
    public GameObject avatarPrefab; // Trascina qui il Prefab del tuo Avatar

    // Dizionario: collega il person_id (int) al GameObject dell'Avatar instanziato
    private Dictionary<int, GameObject> avatarsInScene = new Dictionary<int, GameObject>();
    
    // Dizionario per tenere traccia dell'ultimo aggiornamento (per rimuovere chi esce dall'inquadratura)
    private Dictionary<int, float> lastUpdateTime = new Dictionary<int, float>();
    public float timeoutSeconds = 2.0f; // Se non ricevo dati per 2 secondi, distruggo l'avatar

    void Update()
    {
        string json = udpReceiver.GetLatestData();
        if (string.IsNullOrEmpty(json)) return;

        try
        {
            PosePacket packet = JsonUtility.FromJson<PosePacket>(json);
            ManageAvatar(packet);
        }
        catch { /* Ignora errori di parsing incompleti */ }

        CheckTimeouts();
    }

    void ManageAvatar(PosePacket packet)
    {
        int id = packet.person_id;
        lastUpdateTime[id] = Time.time;

        // Se è una persona nuova (ID non presente nel dizionario), crea un nuovo Avatar
        if (!avatarsInScene.ContainsKey(id))
        {
            GameObject newAvatar = Instantiate(avatarPrefab, Vector3.zero, Quaternion.identity);
            newAvatar.name = "Avatar_Person_" + id;
            avatarsInScene.Add(id, newAvatar);
            Debug.Log("Nuova persona rilevata! ID: " + id);
        }

        // Passa i dati di posa allo script dell'Avatar corretto
        AvatarPoseController controller = avatarsInScene[id].GetComponent<AvatarPoseController>();
        if (controller != null)
        {
            // Nota: devi modificare AvatarPoseController affinché abbia un metodo pubblico
            // UpdatePose(PosePacket p) invece di leggere lui stesso l'UDP.
            controller.UpdatePose(packet); 
        }
    }

    void CheckTimeouts()
    {
        List<int> idsToRemove = new List<int>();

        foreach (var kvp in lastUpdateTime)
        {
            if (Time.time - kvp.Value > timeoutSeconds)
            {
                idsToRemove.Add(kvp.Key);
            }
        }

        foreach (int id in idsToRemove)
        {
            Destroy(avatarsInScene[id]);
            avatarsInScene.Remove(id);
            lastUpdateTime.Remove(id);
            Debug.Log("Persona " + id + " persa. Avatar distrutto.");
        }
    }
}
```
*(Ricorda: con questo approccio, dovrai togliere l'`Update()` dallo script `AvatarPoseController` precedentemente creato e sostituirlo con un metodo pubblico `public void UpdatePose(PosePacket packet)` che viene chiamato dal Manager).*



## ESTENSIONE 2: Interazione Fisica con Oggetti e Ambiente

Se l'avatar si muove tramite script (forzando la posizione/rotazione delle ossa), **attraverserà i muri e gli oggetti** come un fantasma. Unity muove la mesh, ma il motore fisico non percepisce le collisioni. Per permettere all'avatar di interagire con scatole, palloni o pareti, dobbiamo trasformare le sue ossa in componenti fisici (Rigidbodies + Colliders).

### Passo A: Configurazione Fisica (Ragdoll)
Unity ha uno strumento integrato per farlo in pochi secondi:
1. Seleziona il tuo Prefab dell'Avatar.
2. Vai nel menu in alto: **GameObject > 3D Object > Ragdoll...**
3. Si aprirà una finestra: trascina le ossa corrispondenti dal tuo Avatar nei campi vuoti (Pelvis, Left Femur, ecc.) e clicca **Create**.
4. Unity aggiungerà automaticamente dei `Capsule Collider` e dei `Rigidbody` su ogni arto.
5. **FONDAMENTALE:** Seleziona tutti i giunti dell'Avatar che ora hanno un `Rigidbody` e spunta la casella **Is Kinematic** nell'Inspector. 
   * *Perché?* Un Rigidbody "Kinematic" non cade per la gravità e non viene mosso dagli impatti, ma **può spingere** gli altri oggetti fisici (es. scatole) quando si muove. Dato che i movimenti sono dettati dai dati MoCap, le ossa devono essere Kinematic.

### Passo B: Afferrare oggetti in modo intelligente (Parenting & Trigger)
Supponiamo di voler far afferrare un oggetto all'avatar. L'algoritmo di Python vede la mano chiudersi a 40cm dalla telecamera, ma non sa dove si trova l'oggetto in Unity. 

1. Aggiungi uno `Sphere Collider` sulla mano dell'Avatar e spunta **Is Trigger**.
2. Quando la mano entra in collisione con un oggetto etichettato (Tag) come "Grabbable", fai scattare uno script.
3. Se l'utente chiude la mano (deducibile dai dati dei *fingers* inviati da SAM3DBody), imposta l'oggetto come "figlio" (parenting) della mano.

Esempio concettuale di script (`HandInteraction.cs`) da attaccare all'osso della mano:

```csharp
using UnityEngine;

public class HandInteraction : MonoBehaviour
{
    private bool isGrabbing = false;
    private GameObject grabbedObject = null;

    void OnTriggerStay(Collider other)
    {
        if (other.CompareTag("Grabbable") && !isGrabbing)
        {
            // Controlla se il sistema Python sta dicendo che la mano è chiusa
            // (Richiede l'invio di questo dato nel pacchetto UDP)
            bool handIsClosed = ControllaSeManoChiusaDaDatiMocap(); 

            if (handIsClosed)
            {
                isGrabbing = true;
                grabbedObject = other.gameObject;
                
                // Disabilita la fisica dell'oggetto e attaccalo alla mano
                grabbedObject.GetComponent<Rigidbody>().isKinematic = true;
                grabbedObject.transform.SetParent(this.transform);
                
                // Opzionale: Azzera la posizione per farlo aderire al palmo
                // grabbedObject.transform.localPosition = Vector3.zero; 
            }
        }
    }

    void Update()
    {
        if (isGrabbing && !ControllaSeManoChiusaDaDatiMocap())
        {
            // L'utente ha aperto la mano, rilascia l'oggetto
            isGrabbing = false;
            grabbedObject.transform.SetParent(null); // Sgancialo
            grabbedObject.GetComponent<Rigidbody>().isKinematic = false; // Riattiva la fisica
            grabbedObject = null;
        }
    }

    // Metodo fittizio per leggere i dati dal tuo gestore UDP
    private bool ControllaSeManoChiusaDaDatiMocap()
    {
        // Da implementare: leggere lo stato delle dita inviato da SAM3DBody
        return false; 
    }
}
```

Questa combinazione (**Ossa IsKinematic** + **Script di Afferramento via Parenting**) creerà l'illusione perfetta di un'interazione fisica reale!