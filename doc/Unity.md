# SISTEMA DI MOTION CAPTURE IN TEMPO REALE
### Integrazione tra Computer Vision (Python/C++) e Grafica 3D (Unity)

---
#### Valutazione del progetto e scelta dell'approccio:

- **SCENARIO A - L'Ideale (HybrIK / SAM3DBody)**: Il modello AI calcola e invia direttamente le rotazioni relative delle articolazioni (Quaternioni/Euler Angles) del modello SMPL.
*Come si usa in Unity:* È l'approccio più pulito ed efficiente. Applicate direttamente le rotazioni alle ossa dell'avatar senza bisogno di Cinematica Inversa (IK), eliminando i problemi di snodo innaturale degli arti.
- **SCENARIO B - Piano di Riserva (AlphaPose / Posizioni 3D)**: L'algoritmo fornisce solo le coordinate spaziali (X, Y, Z) dei giunti.
*Come si usa in Unity:* Richiede il sistema di Cinematica Inversa ufficiale di Unity (**Animation Rigging**)   . Sposti i punti di destinazione (targets) nello spazio 3D e l'IK calcola matematicamente come devono piegarsi gomiti e ginocchia.


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

### 2.1 Framework Basati su Modelli Parametrici (HybrIK / SAM3DBody) - SCENARIO A
Questi modelli tentano di ricostruire l'intera superficie corporea partendo da una singola immagine   .
*   **Vantaggio in Unity:** Calcolano direttamente i vettori di rotazione relativa delle giunzioni   . Inviando a Unity le *rotazioni* anziché le *posizioni*, si elimina la necessità di calcolare la cinematica inversa in Unity   .

### 2.2 AlphaPose - SCENARIO B
AlphaPose rileva prima le persone nella scena (Bounding Box) e successivamente stima i punti chiave (coordinate X, Y, Z)   .
*   **Limiti in Unity:** L'output nativo richiede algoritmi di *lifting* per proiettare i punti nello spazio 3D   . In Unity, sarai obbligato a usare il sistema *Animation Rigging* per far calcolare al motore grafico come piegare gli arti per raggiungere tali coordinate.




## 3. ARCHITETTURA DI COMUNICAZIONE E PROTOCOLLO DI STREAMING UDP

Il collegamento tra l'ambiente di computazione (Python) e l'ambiente di rendering (Unity) deve avvenire senza accumulare ritardi. Il protocollo scelto per questa pipeline è **UDP (User Datagram Protocol)**.

|          LATO PYTHON              |       |            LATO UNITY             |
|:----------------------------------|:------|:----------------------------------|
| 1. Inferenza (HybrIK/AlphaPose)    | | 1. Thread UDP in ascolto    |
| 2. Estrazione Rotazioni/Posizioni | | 2. Parsing JSON    |
| 3. Invio Socket UDP    | --(Rete)&rarr; | 3. Rotazione Ossa o Calcolo IK    |



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
Se ricevessi le rotazioni anziché le semplici posizioni (XYZ), il formato JSON dovrebbe includere i dati per orientare l'osso nello spazio tridimensionale.
Il formato più utilizzato e sicuro in grafica 3D per evitare problemi di blocco del gimbal (Gimbal Lock) è basato sui Quaternioni (`x, y, z, w`), oppure in alternativa sugli angoli di Eulero (`rx, ry, rz`).
Ecco come apparirebbe il payload JSON strutturato per inviare le rotazioni (o un pacchetto completo per più giunti e una persona):

```json
{
  "person_id": 0,
  "joints": [
    {
      "joint_name": "left_wrist",
      "rw": 0.923,
      "rx": 0.0,
      "ry": 0.382,
      "rz": 0.0
    }
  ]
}
```
Dettaglio dei campi per le rotazioni:
- `rw, rx, ry, rz`: Rappresentano i quattro componenti di un Quaternione, che descrivono l'orientamento puro dell'articolazione rispetto al suo osso padre.  
- Perché i Quaternioni? Unity gestisce nativamente la rotazione delle ossa (`transform.localRotation`) tramite Quaternioni. Ricevere questi quattro valori permette di applicarli direttamente all'avatar senza dover calcolare complessi algoritmi di cinematica inversa (IK).

In Unity, un thread parallelo (Background Worker) rimane in ascolto sulla porta locale impostata (es. `127.0.0.1:7000`), deserializza la stringa e passa i valori numerici al ciclo di aggiornamento grafico principale (`Update`), completando il ciclo di animazione in tempo reale.

---


## 4. GUIDA DETTAGLIATA PER L'IMPLEMENTAZIONE IN UNITY

### PASSO 1: Configurazione dell'Avatar Humanoid
Segui la configurazione della scheda **Rig -> Humanoid** descritta al punto 1   . Verifica la mappatura cliccando su **Configure...**   .

### PASSO 2: Creazione dello Script Ricevitore UDP (`UDPReceiver.cs`)
In Unity i dati UDP arrivano in continuo su un **Thread secondario**   . Crea un GameObject vuoto chiamato `[UDP_Manager]` e assegnagli questo script per ricevere i dati e salvarli in modo sicuro   :

```csharp
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UDPReceiver : MonoBehaviour
{
    public int port = 7000;
    private UdpClient client;
    private Thread receiveThread;
    
    [HideInInspector] public string latestData = "";
    private readonly object lockObject = new object();

    void Start()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData)) { IsBackground = true };
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
                lock (lockObject) { latestData = Encoding.UTF8.GetString(data); }
            }
            catch (Exception err) { Debug.LogError("Errore UDP: " + err.ToString()); }
        }
    }

    public string GetLatestData()
    {
        lock (lockObject) { return latestData; }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
        client?.Close();
    }
}
```

### PASSO 3: Mappatura Dati sull'Avatar (SCENARIO A - ROTAZIONI DIRETE)
Se Python riesce a inviare le rotazioni, lo script `AvatarPoseController.cs` le applicherà fluidamente   .

```csharp
void ApplyPose(PosePacket packet)
{
    foreach (var j in packet.joints)
    {
        HumanBodyBones bone = MapJointToBone(j.joint_name);
        if (bone != HumanBodyBones.LastBone)
        {
            Transform boneTransform = animator.GetBoneTransform(bone);
            if (boneTransform != null)
            {
                // Assegnazione diretta della rotazione senza IK
                Quaternion targetRotation = new Quaternion(j.rx, j.ry, j.rz, j.rw);
                boneTransform.localRotation = Quaternion.Slerp(boneTransform.localRotation, targetRotation, Time.deltaTime * 15f);
            }
        }
    }
}
```

### PASSO 4: Integrazione "Animation Rigging" (SCENARIO B - POSIZIONI X, Y, Z)
Se Python invia coordinate spaziali (X,Y,Z), devi usare l'IK per evitare che l'avatar si deformi muovendo solo il polso   :
1. Vai su **Window > Package Manager** in Unity e installa il pacchetto gratuito **Animation Rigging**.
2. Seleziona il tuo Avatar nella scena, vai nel menu in alto: **Animation Rigging > Rig Setup**.
3. Crea dei GameObject vuoti per i target (es. `Target_Mano_DX`).
4. Aggiungi il componente **Two Bone IK Constraint** (per le braccia/gambe).
5. Assegna nello script del Constraint le ossa (Root=Spalla, Mid=Gomito, Tip=Mano) e il tuo `Target_Mano_DX` come bersaglio finale.
6. **Nello script C#**, anziché ruotare l'osso, sposterai il target nello spazio 3D:
```csharp
targetManoDX.transform.position = Vector3.Lerp(
    targetManoDX.transform.position, 
    new Vector3(j.x, j.y, j.z), 
    Time.deltaTime * 15f
);
```
Il solutore calcolerà istantaneamente le rotazioni necessarie rispettando i vincoli anatomici   



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