# Porting di una pipeline realtime AlphaPose + HybrIK da Linux/CUDA a macOS Intel, e unificazione del lato Unity

## 1. Contesto

Il progetto si compone di due parti distinte ma collegate:

1. una **pipeline di computer vision** (Python) che cattura il video webcam, rileva le persone, stima la posa 3D e invia i dati via UDP;
2. un **progetto Unity**, dove un `AvatarController` riceve questi dati via rete e li applica a un avatar 3D in tempo reale.

Il presente capitolo documenta entrambi i fronti di lavoro: il porting della pipeline Python da Linux/CUDA a macOS Intel (CPU-only), e la successiva unificazione dell'architettura Unity, resa necessaria dal fatto che il progetto supporta **tre pipeline di detection alternative** (SAM3DBody in C++, MMDetection+HybrIK, AlphaPose+HybrIK), ciascuna con un proprio formato di output.

### 1.1 La pipeline Python oggetto del porting

Lo script di partenza (`angoliAlphaPose_HybrIK.py`, versione Linux) implementa una pipeline realtime che:

1. cattura il flusso webcam,
2. rileva le persone nel frame con un detector YOLO (AlphaPose),
3. stima la posa 3D SMPL con HybrIK (24 giunti, rotazioni + coordinate 3D),
4. converte le rotazioni in angoli di Eulero nel sistema di riferimento di Unity,
5. invia i dati via pacchetto UDP a un'applicazione esterna (es. Unity),
6. mostra opzionalmente una finestra con lo scheletro sovrapposto al video.

Lo script originale era scritto assumendo un ambiente **Linux con GPU NVIDIA/CUDA** obbligatoria: il codice legge la GPU con `torch.cuda.get_device_name(...)` senza alcun ramo alternativo, usa `torch.autocast(device_type="cuda", ...)` per l'inferenza in FP16, e si appoggia a componenti di AlphaPose (cattura webcam multithread, NMS) che dipendono da estensioni compilate nativamente per quella piattaforma.

L'obiettivo del porting era far girare la stessa pipeline su **Mac Intel, senza GPU dedicata, quindi interamente su CPU**, mantenendo intatta la logica applicativa (detection → pose → UDP).

## 2. Metodo di lavoro

Il lavoro si è svolto in due sessioni distinte, riportate rispettivamente nei capitoli 3 e 4: una prima fase di **installazione e configurazione dell'ambiente** (repository, ambiente Python, build delle estensioni native, reperimento dei pesi pre-addestrati), seguita da una fase di **debugging a runtime** una volta che lo script era finalmente eseguibile ma presentava ancora comportamenti scorretti (nessuna finestra visibile, crash, scala errata, prestazioni basse).

In entrambe le fasi il porting è stato fatto in modo incrementale: ad ogni tentativo di esecuzione, lo script falliva su un problema diverso; ogni errore è stato isolato con test minimi mirati (es. uno script cv2 standalone per capire se il problema fosse OpenCV o la pipeline), corretto, e si passava all'errore successivo. Questo approccio "a strati" si è rivelato necessario perché i problemi appartenevano a categorie molto diverse tra loro (compatibilità di sistema operativo, differenze di firma tra versioni di libreria, dipendenze binarie mancanti o non compilate, bug di logica), e risolverli "alla cieca" tutti insieme avrebbe reso impossibile capire quale fix avesse davvero effetto.

## 3. Installazione e configurazione iniziale su macOS (fase di lavoro con Gemini)

Prima ancora di arrivare ai problemi di *runtime* descritti nel capitolo 4, è stata necessaria un'intera fase di installazione e configurazione dell'ambiente su macOS, condotta in una sessione di lavoro precedente. Questa fase ha prodotto la versione "adattata per Mac" dello script (quella poi ereditata ed ulteriormente corretta nel capitolo 4) e ha richiesto modifiche dirette a file dell'installazione di AlphaPose stessa (`setup.py`, `nms_wrapper.py`), non solo allo script applicativo.

### 3.1 Rimozione del vincolo CUDA obbligatorio

Lo script originale, scritto dal compagno di gruppo per Linux con GPU NVIDIA, interrompeva l'esecuzione con un errore esplicito se CUDA non era disponibile (`raise RuntimeError("CUDA non disponibile...")`), assumendo l'accelerazione hardware come requisito. Essendo il Mac Intel privo sia di CUDA sia del backend MPS (disponibile solo su Apple Silicon), questa prima modifica ha introdotto il ramo alternativo `device = torch.device("cpu")`, con relativa disattivazione automatica di FP16/AMP (ottimizzazioni specifiche per GPU NVIDIA, prive di senso su CPU). È questa la versione da cui è partito il lavoro di debugging del capitolo 4.

### 3.2 Ambiente Python: da Conda a venv, e downgrade della versione

Il primo tentativo di creare un ambiente isolato con **Conda** è fallito perché `conda` non è installabile tramite `pip` (è un gestore di ambienti a sé stante, da scaricare come Miniconda/Anaconda). Si è quindi passati al modulo **`venv`** standard di Python, già incluso in macOS:

```bash
python3 -m venv venv_tesina
source venv_tesina/bin/activate
```

Un secondo problema è emerso con la versione di Python: il venv creato con **Python 3.13** (versione di sistema del Mac) ha causato ripetuti errori di build delle estensioni native di AlphaPose (in parte dovuti a incompatibilità tra `setuptools` recenti e il vecchio `setup.py` del progetto). La soluzione è stata ricreare l'ambiente con **Python 3.10**:

```bash
rm -rf venv_tesina
python3.10 -m venv venv_tesina
source venv_tesina/bin/activate
```

### 3.3 Build nativa di AlphaPose: conflitti di dipendenze e patch a `setup.py`

L'installazione di AlphaPose (`pip install -e .` nella cartella del repository clonato) ha richiesto diversi interventi in sequenza:

- **`requirements.txt` assente**: il repository non lo include nella forma attesa; i pacchetti sono stati installati singolarmente (`opencv-python`, `numpy`, `pyyaml`, `scipy`, `matplotlib`, `cython`, `easydict`, `json_tricks`, `tensorboard`, `shapely`, `tqdm`, ecc.).
- **`ModuleNotFoundError: No module named 'torch'`** durante la build: `setup.py` importa `torch` per definire le estensioni C++/CUDA, ma l'isolamento di build di `pip` non vede i pacchetti già installati nel venv. Risolto forzando `pip install --no-build-isolation -e .`.
- **Conflitto NumPy 2.x**: la versione più recente di NumPy rompe la compatibilità con i moduli Cython/PyTorch usati da AlphaPose. Risolto fissando `pip install "numpy<2.0"`, con conseguente necessità di allineare anche OpenCV a una versione compatibile (`pip install "opencv-python<5.0"`, dato che OpenCV 5.x richiede NumPy ≥2).
- **`OSError: CUDA_HOME environment variable is not set`**: il vero blocco principale. Il `setup.py` originale di AlphaPose tenta sempre di compilare le estensioni CUDA (incluso il modulo Cython `soft_nms_cpu.pyx`, che nonostante il nome "cpu" nel processo di build originale veniva comunque legato alla ricerca di CUDA), fallendo su un sistema privo di GPU NVIDIA. È stato quindi necessario **riscrivere `setup.py`**, rimuovendo interamente la logica di compilazione delle estensioni CUDA e mantenendo solo l'estensione Cython effettivamente utilizzabile su CPU (`soft_nms_cpu`), tramite una funzione `get_ext_modules()` semplificata e protetta da `try/except`.
- **Bug residuo nella patch**: la prima riscrittura di `setup.py` passava `extra_compile_args` come `dict` invece che come `list` su macOS, causando `TypeError: can only concatenate list (not "dict") to list` durante la compilazione. Corretto normalizzando l'argomento a una semplice lista di flag del compilatore.

Solo dopo questi interventi l'installazione si è conclusa con successo (`Successfully built alphapose`).

### 3.4 Installazione di HybrIK e dipendenza da `chumpy`

HybrIK è stato installato allo stesso modo (`pip install -e .` in modalità sviluppo), con un'installazione complessivamente meno problematica rispetto ad AlphaPose. È stata però necessaria la libreria **`chumpy`**, richiesta per caricare correttamente i file `.pkl` (formato legacy usato dai modelli statistici SMPL) inclusa tra le dipendenze installate insieme a `pycocotools`, `tqdm` ed `easydict`.

### 3.5 Prima versione del fallback NMS (`nms_wrapper.py`)

Con `soft_nms_cpu` unica estensione compilata (e non il modulo `nms_cpu` vero e proprio, usato per il Non-Maximum Suppression delle bounding box), il codice originale di AlphaPose andava in errore con `ImportError: cannot import name 'nms_cpu' from partially initialized module`. È stata quindi scritta una prima versione di `detector/nms/nms_wrapper.py` con import "protetti" in cascata (`nms_cuda` → `nms_cpu` → `None`) e una funzione `nms()` che sceglie il primo backend disponibile, sollevando un errore esplicito se nessuno dei due è compilato. Questa è esattamente la versione di `nms_wrapper.py` ereditata all'inizio del capitolo 4 (§4.5): risultava sufficiente a far *importare* il modulo senza crash immediati, ma non forniva ancora un vero fallback funzionante su CPU quando **nessuna** delle due estensioni native risultava effettivamente disponibile — problema poi risolto definitivamente con `torchvision.ops.nms`.

### 3.6 Reperimento dei modelli pre-addestrati

Il posizionamento dei pesi pre-addestrati richiesti dalla pipeline ha comportato diverse difficoltà pratiche:

- **Checkpoint HybrIK**: il file atteso (`pretrained_w_cam.pth`) non è stato reperito immediatamente; è stato inizialmente usato un file alternativo trovato online (`pretrained_hrnet.pth`), poi sostituito con il checkpoint corretto una volta reperito.
- **Modello statistico SMPL** (`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`): file mancante nella cartella `model_files/` attesa dal codice. È stato individuato tra i download precedenti dell'utente (in una posizione e con un nome leggermente diversi) e copiato/rinominato nel percorso corretto.
- **Pesi del detector YOLO** (`yolov3-spp.weights`): le indicazioni su dove scaricarli e posizionarli (`detector/yolo/data/` dentro AlphaPose) sono state fornite in questa fase, ma il download effettivo è stato completato solo più avanti, nella sessione di debugging del capitolo 4 (§4.4), quando lo script si è effettivamente bloccato per l'assenza del file.

### 3.7 Disallineamento tra checkpoint e configurazione del modello

Una volta risolti i problemi di percorso, il caricamento del modello HybrIK falliva con `RuntimeError: Error(s) in loading state_dict`, per due motivi distinti emersi in sequenza:

1. **Backbone incompatibile**: un primo tentativo con `pretrained_hrnet.pth` (basato su una backbone HRNet) è stato erroneamente abbinato a un file di configurazione `.yaml` pensato per una backbone ResNet-34 (`256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml`), con nomi e forme dei layer completamente diversi tra checkpoint e modello istanziato. Risolto usando il checkpoint corretto (`pretrained_w_cam.pth`, coerente con la configurazione ResNet-34 già in uso).
2. **Discrepanza nei parametri di forma SMPL**: anche con il checkpoint corretto, restava un errore di *size mismatch* sul tensore `smpl.shapedirs` (300 componenti nel modello SMPL usato dal codice, contro i 10 attesi dal checkpoint pre-addestrato). Risolto troncando a runtime il tensore prima del caricamento dei pesi:

```python
if hasattr(pose_model, 'smpl') and hasattr(pose_model.smpl, 'shapedirs'):
    if pose_model.smpl.shapedirs.shape[-1] == 300:
        pose_model.smpl.shapedirs = pose_model.smpl.shapedirs[:, :, :10]
```

Questa correzione è presente anche nella versione finale dello script (subito prima della chiamata a `pose_model.load_state_dict(checkpoint)`).

Solo al termine di questa fase la pipeline è arrivata a un punto di esecuzione stabile (avvio, caricamento modelli, invio pacchetti UDP), lasciando come problema residuo l'assenza della finestra di visualizzazione — il punto di partenza esatto del capitolo 4.

## 4. Problemi riscontrati e soluzioni nella fase di debugging runtime

### 4.1 La finestra della webcam non appariva mai

**Sintomo:** lo script catturava correttamente il video e inviava i pacchetti UDP, ma la finestra `cv2.imshow` non compariva mai, senza errori e senza consumo di CPU.

**Causa:** `WebCamDetectionLoader` di AlphaPose cattura i frame webcam in un `threading.Thread` separato, anche in modalità `--sp` (single process). Su Linux questo funziona senza problemi; su **macOS**, il backend di cattura video di OpenCV (AVFoundation) richiede che la cattura avvenga nel *thread principale*, dove gira il run loop di sistema (Cocoa). Se la cattura viene avviata da un thread secondario, `stream.read()` può bloccarsi indefinitamente senza generare eccezioni.

**Soluzione:** sostituzione di `WebCamDetectionLoader` con una classe equivalente (`SyncWebcamLoader`), che replica la stessa logica (stesso detector, stessa trasformazione di crop/resize per HybrIK) ma esegue cattura, detection e preprocessing **in modo sincrono nel thread principale**, eliminando il thread di cattura.

### 4.2 Incompatibilità di firma in `SimpleTransform3DSMPL`

**Sintomo:** `TypeError: SimpleTransform3DSMPL.__init__() got an unexpected keyword argument 'gpu_device'`.

**Causa:** la versione della classe installata localmente (specifica per la variante SMPL/HybrIK usata) non accetta esattamente gli stessi parametri della versione documentata online.

**Soluzione:** costruzione dinamica degli argomenti tramite `inspect.signature`, filtrando automaticamente solo i parametri effettivamente accettati dal costruttore installato, invece di forzare un elenco fisso di kwargs.

### 4.3 `args.gpus` di tipo stringa invece che lista

**Sintomo:** `TypeError: '<' not supported between instances of 'str' and 'int'` dentro il detector YOLO (`yolo_api.py`), che si aspetta `args.gpus` come lista di interi (es. `[-1]` per CPU).

**Causa:** nel ramo CPU dello script, `args.gpus` non veniva mai convertito da stringa a lista (la conversione avveniva solo nel ramo CUDA dell'originale).

**Soluzione:** normalizzazione esplicita di `args.gpus` in entrambi i rami (`[gpu_ids]` su CUDA, `[-1]` su CPU) prima di istanziare il detector.

### 4.4 Pesi YOLO mancanti

**Sintomo:** `FileNotFoundError: detector/yolo/data/yolov3-spp.weights`.

**Causa:** i pesi pre-addestrati di YOLOv3-SPP (~240 MB) non fanno parte del repository AlphaPose e vanno scaricati separatamente.

**Soluzione:** download manuale del file nella cartella attesa dal detector.

### 4.5 Estensione NMS non compilata (fallback definitivo)

**Sintomo:** `ImportError: Neither nms_cuda nor nms_cpu is available.`

**Causa:** il Non-Maximum Suppression di AlphaPose si appoggia a un'estensione C++/CUDA compilata al momento dell'installazione; su macOS senza CUDA questa compilazione non era andata a buon fine per nessuna delle due varianti (CPU e GPU).

**Soluzione:** aggiunta di un fallback puro PyTorch in `nms_wrapper.py`, che usa `torchvision.ops.nms` (già incluso in `torchvision`, nessuna compilazione richiesta) quando né `nms_cuda` né `nms_cpu` sono disponibili, mantenendo lo stesso contratto di ritorno (`dets_tenuti, indici`) atteso dal resto del codice.

### 4.6 Bug di indicizzazione nel disegno dei giunti

**Sintomo:** `IndexError: invalid index to scalar variable` durante il disegno dello scheletro.

**Causa:** l'output `pred_xyz_jts_29` di HybrIK per persona è un array **1D appiattito** (29 giunti × 3 coordinate = 87 valori), non un array `(29, 3)` come assunto inizialmente dal codice di visualizzazione.

**Soluzione:** `reshape(-1, 3)` prima di iterare sui giunti.

### 4.7 Colori invertiti (RGB/BGR)

**Causa:** `WebCamDetectionLoader`/`SyncWebcamLoader` convertono il frame da BGR a RGB (`frame[:, :, ::-1]`) per coerenza con il resto della pipeline AlphaPose, ma `cv2.imshow` si aspetta BGR.

**Soluzione:** riconversione esplicita con `cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)` prima della visualizzazione.

### 4.8 Scala errata nella visualizzazione dello scheletro

**Sintomo:** i giunti disegnati risultavano inizialmente ammassati al centro del corpo, poi (con uno scale factor fisso `box_height * 0.7`) troppo piccoli rispetto alla figura reale, anche se il movimento relativo tra i giunti era corretto.

**Causa:** `pred_xyz_jts_29` è espresso in coordinate 3D metriche root-relative (relative al bacino), in un'unità di misura che non è direttamente proporzionale ai pixel dell'immagine con un fattore fisso e "banale" da indovinare.

**Soluzione:** calcolo dinamico, frame per frame, del fattore di scala: si misura quanto i giunti si allontanano realmente dal bacino nell'output del modello (usando il 90° percentile delle distanze, per non farsi rovinare la stima da un singolo giunto rumoroso), e si mappa questo valore su una frazione ragionevole (~45%) della reale altezza in pixel del bounding box della persona rilevata. In questo modo lo scheletro si adatta automaticamente alla dimensione reale della persona nell'inquadratura, indipendentemente dall'unità di misura assoluta usata dal modello.

### 4.9 Prestazioni su CPU

Senza GPU, l'intera pipeline (detection YOLO + regressione HybrIK) gira su CPU, con un costo computazionale per frame significativamente più alto rispetto a un'esecuzione CUDA. Sono state applicate alcune ottimizzazioni mirate:

- `torch.set_num_threads(os.cpu_count())`, per assicurarsi che PyTorch sfrutti tutti i core disponibili (di default, in alcuni ambienti virtuali, ne usa solo uno);
- riduzione della risoluzione di cattura webcam (640×480), per alleggerire sia il preprocessing YOLO sia il crop per HybrIK;
- introduzione di un parametro opzionale `--det-every-n`, che consente di eseguire la detection YOLO solo ogni N frame riusando l'ultimo bounding box trovato (mentre la stima di posa HybrIK continua a girare su ogni frame), utile come ulteriore leva di compromesso qualità/velocità.

## 5. Integrazione Unity: dall'AvatarController monolitico all'architettura a Provider

### 5.1 Il problema di partenza

Il progetto prevede tre pipeline di detection alternative, sviluppate da persone diverse del gruppo:

- **Scenario A – SAM3DBody** (C++): invia rotazioni in **angoli di Eulero** (gradi), con una nomenclatura dei giunti personalizzata (es. `hip`, `lShldr`, dita numerate), tramite un `BVHWriter`.
- **Scenario B – AlphaPose + HybrIK** (Python): invia rotazioni come **quaternioni** `(x, y, z, w)` sui 24 giunti standard SMPL.
- **Scenario C – MMDetection + HybrIK** (Python): stesso formato quaternioni/SMPL dello Scenario B, ma con un detector diverso a monte.

Ogni pipeline era stata affiancata da una propria versione, evoluta indipendentemente, dello script `AvatarController.cs` originale, arrivando a **tre script non compatibili tra loro**, ciascuno con la propria logica di calibrazione automatica (azzeramento della posa iniziale dopo un timer), smoothing/dead-zone anti-jitter e gestione della root position — tutte funzionalità utili, ma duplicate e non riutilizzabili tra le pipeline.

### 5.2 Architettura a Provider (Strategy Pattern)

La soluzione adottata è il pattern **Strategy**, comune in Unity per questo tipo di situazioni: invece di tre `AvatarController` distinti, si mantiene **un solo controller**, che delega la lettura e conversione dei dati grezzi a componenti intercambiabili.

- **`IPoseProvider`** — interfaccia comune che ogni pipeline deve implementare (nella versione finale: `void ApplyPose(string jsonPayload, Animator animator)`), in modo che il controller non debba sapere *come* i dati vengono letti o convertiti, solo che riceverà rotazioni pronte da applicare.
- **Un Provider per pipeline** (`HybrIKProvider`, `SAM3DProvider`, ecc.), ciascuno responsabile di: parsing del proprio formato JSON specifico, mappatura nome-giunto → osso dell'avatar (es. `ConvertSamBoneName` per la nomenclatura personalizzata di SAM3DBody), calibrazione iniziale e smoothing.
- **`AvatarController`** unificato, che si limita a inoltrare i dati al provider correntemente attivo.

Questo isola la specificità di ogni pipeline (formato rotazioni, nomi giunti) nel proprio Provider, mantenendo la logica comune (movimento, applicazione alle ossa) in un unico punto.

### 5.3 Selezione dinamica della pipeline via rete

Avere tre Provider pronti non risolve da solo il problema pratico di dover riconfigurare manualmente la scena Unity ogni volta che si cambia pipeline di test. È stato quindi introdotto un meccanismo di **selezione automatica via rete**, su due lati:

**Lato Python — `main_launcher.py`:** uno script "cabina di regia" con un menu testuale interattivo, che permette di scegliere quale pipeline avviare. Alla selezione:
1. invia subito un pacchetto UDP di **handshake** (es. `{"pipeline_type": "alphapose-hybrik"}`), *prima* di avviare il modello di detection vero e proprio (evitando ritardi dovuti al caricamento dei pesi);
2. lancia come sottoprocesso lo script corrispondente alla pipeline scelta (`angoliAlphaPose_HybrIK.py`, lo script MMDetection+HybrIK, o il binario C++ di SAM3DBody), che da quel momento invia in streaming i pacchetti di posa sulla stessa porta UDP.

**Lato Unity — `ConnectionManager.cs`:** un componente che resta in ascolto fin dall'avvio (`Play`), con **tutti i Provider disabilitati** di default. Alla ricezione del pacchetto di handshake, individua il Provider corrispondente al `pipeline_type` ricevuto, lo abilita e disabilita gli altri. I pacchetti successivi (quelli di posa vera e propria, non handshake) vengono inoltrati automaticamente al solo Provider attivo. Cambiare pipeline in corsa (fermare lo script Python, sceglierne un'altra dal menu) fa scattare un nuovo handshake, che disattiva il Provider precedente e attiva quello nuovo "a caldo", senza dover toccare la scena Unity.

### 5.4 Separazione delle responsabilità lato Unity: `UDPReceiver` come sensore puro

Il progetto disponeva già di uno script `UDPReceiver.cs`, con un thread in background sempre in ascolto sulla porta UDP, usato dal vecchio `AvatarController` monolitico. Integrarlo con la nuova architettura senza interferenze ha richiesto di **restringerne la responsabilità**: da "riceve e decide cosa fare con i dati" a puro "sensore" che riceve i byte e li inoltra al `ConnectionManager`, che è l'unico punto che interpreta il contenuto (handshake o dati di posa).

Un dettaglio tecnico rilevante: poiché `UDPReceiver` riceve i pacchetti su un thread separato, ma Unity richiede che gli oggetti di scena (`Transform`, `Animator`) vengano toccati solo dal thread principale, è stato introdotto un componente di supporto, **`UnityMainThreadDispatcher`**, che mette in coda le azioni ricevute dal thread di rete e le esegue nel `Update()` del thread principale — evitando crash o comportamenti indefiniti nell'applicazione delle rotazioni all'avatar.

### 5.5 Flusso di esecuzione a runtime

In sintesi, l'esecuzione end-to-end segue questi passi:

1. **Play** in Unity → `ConnectionManager` in ascolto sulla porta UDP, tutti i Provider disabilitati, avatar fermo.
2. L'utente lancia `main_launcher.py` da terminale e sceglie la pipeline dal menu.
3. Python invia immediatamente l'handshake UDP con il tipo di pipeline scelto.
4. `UDPReceiver` riceve il pacchetto e lo inoltra al `ConnectionManager`, che riconosce l'handshake, abilita il Provider corrispondente e disabilita gli altri.
5. Python avvia il vero script di detection (webcam + modello); da qui in poi ogni frame produce un pacchetto JSON di posa, inviato in streaming sulla stessa porta.
6. Il `ConnectionManager` riconosce questi pacchetti come dati di posa (non handshake) e li inoltra al solo Provider attivo, che applica calibrazione, smoothing e rotazioni alle ossa dell'avatar in tempo reale.

Per cambiare pipeline basta interrompere lo script Python in esecuzione, tornare al menu di `main_launcher.py` e scegliere un'altra opzione: nessuna modifica manuale alla scena Unity è necessaria.

*(Nota metodologica: questa parte del lavoro — l'architettura Unity — è stata sviluppata in una sessione di lavoro dedicata, separata da quella del porting macOS descritto nei capitoli precedenti.)*

## 6. Sintesi delle differenze Linux (GPU) → macOS (CPU)

| Aspetto | Versione Linux/CUDA originale | Versione macOS/CPU |
|---|---|---|
| Ambiente Python | Non specificato/gestito a parte | venv (non Conda) con Python 3.10, dopo tentativi falliti con Conda e Python 3.13 |
| Build estensioni native | `setup.py` compila estensioni CUDA (incl. NMS) | `setup.py` riscritto: nessuna estensione CUDA, solo `soft_nms_cpu` via Cython |
| Cattura webcam | Thread separato (`WebCamDetectionLoader`) | Sincrona nel thread principale (`SyncWebcamLoader`), per compatibilità con AVFoundation |
| Device | CUDA obbligatoria, nessun fallback | Rilevamento automatico CUDA/CPU, `args.gpus` normalizzato in entrambi i casi |
| Precisione | FP16/AMP su GPU | Disabilitata (FP16 non ha senso su CPU standard) |
| NMS a runtime | Estensione compilata CUDA/CPU | Fallback `torchvision.ops.nms` (pura PyTorch), dopo un primo tentativo di fallback a cascata insufficiente |
| Modello SMPL | Checkpoint e config coerenti per costruzione | Allineamento manuale necessario (backbone HRNet/ResNet, troncamento `shapedirs` 300→10) |
| Prestazioni | Accelerazione hardware GPU | Multi-threading CPU esplicito, risoluzione ridotta, detection sotto-campionata opzionale |
| Visualizzazione | Non presente/non testata nello script originale | Finestra con scheletro SMPL, con scaling dinamico giunti→pixel |

## 7. Conclusioni

Il lavoro descritto ha coperto tre tipi di problemi molto diversi tra loro, entrambi tipici di un progetto che integra più componenti software eterogenee (Python/CV, C++, Unity/C#) sviluppate da persone diverse su piattaforme diverse.

La fase di **installazione e configurazione** (capitolo 3) ha richiesto di intervenire direttamente sui file dell'installazione di AlphaPose e HybrIK stessi — non solo sullo script applicativo — per aggirare l'assunzione, diffusa in questo tipo di progetti di ricerca, che una GPU NVIDIA con CUDA sia sempre disponibile: dalla riscrittura di `setup.py` per rimuovere la compilazione delle estensioni CUDA, al reperimento e allineamento manuale dei pesi pre-addestrati con le rispettive configurazioni.

La fase di **debugging runtime** (capitolo 4), una volta che la pipeline era finalmente eseguibile, ha coperto tre categorie distinte di problemi: **compatibilità di sistema operativo** (threading e cattura video con AVFoundation), **dipendenze binarie mancanti o non compilate per la piattaforma** (pesi YOLO, estensione NMS non ancora del tutto funzionante), e **bug di logica pre-esistenti o emersi solo eseguendo davvero il codice** (formato dell'output di HybrIK, scala di visualizzazione, tipizzazione di `args.gpus`).

Sul lato **Unity** (capitolo 5), il problema non era di piattaforma ma di **architettura software**: tre pipeline con formati di output incompatibili (quaternioni vs. Eulero, nomenclature dei giunti diverse) avevano prodotto tre script `AvatarController` non riconciliabili. La soluzione — pattern Strategy con interfaccia `IPoseProvider`, selezione dinamica della pipeline attiva tramite un handshake di rete, e separazione netta tra "ricezione dati" (`UDPReceiver`) e "instradamento/interpretazione" (`ConnectionManager`) — ha permesso di ottenere un singolo punto di ingresso (`main_launcher.py` lato Python, una sola scena Unity lato client) senza dover scegliere quale pipeline "vince" sulle altre, mantenendo tutte e tre disponibili e selezionabili a runtime.

Nelle tre fasi, l'approccio efficace è stato lo stesso: isolare un problema alla volta, verificarlo con un test minimo o un log mirato prima di proporre una correzione, e solo dopo passare al problema successivo — invece di provare a correggere "a blocchi" più cause potenziali insieme, il che avrebbe reso impossibile capire quale intervento avesse davvero risolto cosa.

# Relazione Tecnica: Porting della Pipeline MMDetection + HybrIK su macOS Intel

## 1. Introduzione

Nell'architettura del sistema di Motion Capture descritto nella tesina, lo **Scenario C** delega il compito di localizzazione e tracking dei soggetti nello spazio bidimensionale al framework **MMDetection**, mantenendo HybrIK per la stima della posa 3D (esattamente come nello Scenario B con AlphaPose).

Come per lo Scenario B, lo script di partenza (`angoliMMdetection_HybrIK.py`) era stato scritto per **Linux con GPU NVIDIA/CUDA obbligatoria** (percorsi `/home/alessio/...`, `device = f'cuda:{opt.gpu}'`, `raise RuntimeError("CUDA non disponibile")` in assenza di GPU, detector Faster R-CNN, checkpoint HybrIK `hybrik_hrnet.pth`). Il porting su macOS Intel ha richiesto due tipi di intervento distinti: la costruzione di un **ambiente virtuale condiviso** con AlphaPose senza rompere quest'ultimo, e la risoluzione di una catena di errori runtime specifici di macOS — in parte diversi nella natura da quelli incontrati con AlphaPose, come si vedrà nel §4.

Questa relazione è stata prodotta a partire da due fonti: la relazione tecnica precedente (che documentava la scelta architetturale di `mmcv-lite` + `torchvision.ops`) e la trascrizione integrale della sessione di lavoro con Gemini in cui l'intera pipeline è stata resa funzionante.

## 2. Creazione dell'ambiente virtuale condiviso (AlphaPose + MMDetection)

A differenza della relazione precedente, qui viene documentato per la prima volta **l'intero iter di allestimento dell'ambiente**, condiviso tra i due scenari (`venv_tesina_310`), e le precauzioni prese per non compromettere l'installazione di AlphaPose già funzionante.

### 2.1 Riuso della venv esistente

Anziché creare un secondo ambiente virtuale isolato per MMDetection, è stata riutilizzata la stessa `venv_tesina_310` (Python 3.10) già allestita per AlphaPose e descritta nel capitolo 3 della tesina principale. La scelta comporta un vantaggio (un solo ambiente da mantenere, un solo interprete da richiamare da Unity/dal launcher) ma anche un rischio concreto: **ogni dipendenza installata per MMDetection può, in linea di principio, rompere AlphaPose** se cambia una libreria condivisa (PyTorch, NumPy, OpenCV). Questo rischio si è effettivamente materializzato durante il lavoro (v. §2.3) ed è stato esplicitamente posto come vincolo in una delle richieste fatte a Gemini ("dammi la soluzione che non si scontra con il progetto fatto con AlphaPose").

### 2.2 Clonazione di MMDetection e primo tentativo di installazione di MMCV

Il framework **MMDetection (v3.3.0)** è stato clonato dal sorgente ufficiale (progetto OpenMMLab) all'interno della cartella della tesina, seguendo lo stesso schema già usato per AlphaPose e HybrIK (repository clonati localmente, non installati da PyPI, con i rispettivi path aggiunti a `sys.path` a runtime tramite il context manager `in_directory`).

Il primo ostacolo è stato l'installazione di **MMCV**, la libreria di base di OpenMMLab che fornisce le operazioni geometriche accelerate (RoI Align, NMS, ecc.) tramite estensioni compilate in C++/CUDA. Il tentativo di installare la versione completa (`mmcv`, non `mmcv-lite`) ha prodotto un fallimento di compilazione:

```
Failed building wheel for mmcv
clang failed with exit code 1
```

La causa, chiarita durante la sessione: non esisteva un pacchetto binario precompilato (`.whl`) per macOS compatibile con la versione di PyTorch installata (2.2.2), quindi `pip` tentava di **compilare MMCV da sorgente** usando il compilatore di sistema (`clang`), fallendo perché il codice C++ di quella versione di MMCV (file come `iou3d.cpp`, `nms.cpp`) conteneva costrutti incompatibili con il compilatore/header disponibili sul Mac.

### 2.3 Il conflitto NumPy/OpenCV con l'ambiente condiviso

Un secondo tentativo di risolvere il problema installando OpenCV in una versione diversa ha avuto un effetto collaterale diretto sulla venv condivisa: il comando `pip install opencv-python` ha silenziosamente aggiornato **NumPy da 1.26.4 a 2.2.6** (dipendenza automatica di OpenCV 5.x) e installato **OpenCV 5.0.0.93**, entrambe versioni potenzialmente incompatibili con i moduli Cython/PyTorch compilati per AlphaPose (v. tesina principale, §3.3, dove proprio NumPy ≥2 era stato identificato come causa di rottura). Questo è un esempio concreto del rischio descritto al §2.1: un comando pensato per risolvere un problema di MMDetection ha modificato silenziosamente due dipendenze critiche condivise con AlphaPose.

La correzione, una volta riconosciuto l'errore (l'ipotesi iniziale che il Mac fosse Apple Silicon si è rivelata sbagliata: è un **Mac Intel**, con GPU discreta AMD compatibile Metal — dettaglio rilevante anche per il §4.3), è stato un downgrade esplicito e mirato:

```bash
pip uninstall opencv-python numpy
pip install "numpy<2.0"
pip install "opencv-python==4.9.0.80"
```

riallineando l'ambiente alle versioni compatibili con AlphaPose, già fissate nel capitolo 3 della tesina principale.

### 2.4 La decisione architetturale: `mmcv-lite` + `torchvision.ops`

Per evitare di dover compilare estensioni native su un Mac privo di CUDA, la decisione finale (già presente nella relazione precedente, qui confermata come esito di questo percorso di tentativi falliti) è stata di installare **`mmcv-lite`**, che non contiene i binari nativi (`mmcv._ext`) ma conserva l'intera logica algoritmica in puro Python, e deviare le operazioni geometriche critiche su **`torchvision.ops`**, già incluso nel PyTorch della venv:

- `torchvision.ops.roi_align` al posto del RoI Align custom di MMCV;
- `torchvision.ops.nms` al posto del NMS C++ di MMCV.

Va notato che il primo utilizzo di `mmcv-lite` ha prodotto un errore prevedibile ma inizialmente non anticipato — `ModuleNotFoundError: No module named 'mmcv._ext'` — nel momento in cui una parte del codice di MMDetection (non ancora deviata su `torchvision.ops`) tentava comunque di caricare le estensioni native assenti in `mmcv-lite`. La soluzione architetturale sopra descritta è stata quindi applicata in modo sistematico a tutte le operazioni che ne avevano bisogno, non solo a NMS.

### 2.5 Bypass del controllo di versione MMCV in MMDetection

Un ulteriore ostacolo, indipendente dai precedenti: `mmdet/__init__.py` esegue un controllo rigido sulla versione di MMCV installata:

```python
assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'
```

La versione di `mmcv-lite` installata (2.2.0) risultava superiore al limite massimo atteso da quella versione di MMDetection, facendo fallire l'`assert` all'avvio, prima ancora di arrivare a un qualunque errore legato alla pipeline vera e propria. **È stato quindi necessario modificare direttamente il file sorgente di MMDetection**, alzando la soglia massima consentita (`<` → `<=`), analogamente a quanto fatto per `setup.py` di AlphaPose nella tesina principale — un altro caso di intervento diretto sui file dell'installazione, non solo sullo script applicativo.

### 2.6 Dipendenza mancante: `pytorch3d`

Un problema non presente nello Scenario B: HybrIK, nella variante usata per lo Scenario C (che carica anche il modello **SMPL-X**, non solo SMPL), richiede la libreria **`pytorch3d`** per le conversioni tra rappresentazioni di rotazione (`axis_angle_to_matrix` e simili), usata internamente da `hybrik/models/layers/smplx/lbs.py`. `pytorch3d` è notoriamente difficile da installare su macOS perché generalmente non distribuisce wheel precompilate per questa piattaforma; l'installazione da sorgente ha inizialmente fallito con lo stesso pattern di errore già visto per AlphaPose (`ModuleNotFoundError: No module named 'torch'` durante il build isolato di `pip`), risolto allo stesso modo:

```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

## 3. Architettura dei Repository e Core Framework

*(Sezione invariata rispetto alla relazione precedente.)*

**A. MMDetection (v3.3.0)** — framework principale basato su PyTorch, clonato dal sorgente originale. Fornisce l'infrastruttura per caricare i file di configurazione, istanziare la rete (BBox Head, Backbone) ed eseguire l'inferenza sui fotogrammi della webcam.

**B. MMEngine (v1.x)** — libreria di base di OpenMMLab che gestisce il ciclo di vita di training/testing (Runner), il caricamento dei checkpoint e la mappatura dei registri (Registry Tree).

## 4. Correzione dei modelli e dei percorsi

Un primo tentativo di avvio ha rivelato che lo script puntava ancora ai percorsi e ai nomi file della versione Linux originale, non aggiornati per lo Scenario C effettivamente documentato nella relazione precedente:

- `DET_CONFIG`/`DET_CKPT` puntavano a **Faster R-CNN** (`faster_rcnn_r50_fpn_1x_coco.py`), non al **RTMDet-Tiny** già scelto e motivato nella relazione precedente per l'inferenza leggera su CPU — corretto puntando ai file corretti in `configs/rtmdet/` e `checkpoints/rtmdet_tiny.pth`.
- `HYBRIK_CKPT` puntava a un file `hybrik_hrnet.pth` mai scaricato; il checkpoint effettivamente reperito si chiamava **`hybrik_hrnet48_wo3dpw.pth`** (HRNet-W48) — corretto aggiornando il percorso nello script anziché rinominare il file scaricato, per mantenere tracciabile la provenienza del checkpoint.

## 5. Stabilizzazione della cattura webcam su macOS: un problema diverso da quello di AlphaPose

Con la pipeline finalmente in grado di partire, è emerso lo stesso sintomo osservato con AlphaPose ("la webcam si apre e si richiude subito"), ma con una **causa radice diversa** — un dettaglio interessante perché mostra che lo stesso sintomo, su macOS, può derivare da problemi non correlati.

Il test diagnostico isolato (`check_cam.py`, la stessa tecnica usata nella tesina principale per lo Scenario B) ha dato:

```
aperta: True
0 ret = False
```

A differenza dello Scenario B — dove la causa era la cattura eseguita in un thread secondario, incompatibile con AVFoundation — qui lo script **catturava già correttamente nel thread principale** (non usa alcun loader threadizzato come `WebCamDetectionLoader`). Il problema era invece che **il primo `cap.read()` immediatamente dopo l'apertura di `VideoCapture` restituisce sistematicamente `ret = False`** su questo Mac: AVFoundation non ha ancora inizializzato il buffer hardware nell'istante in cui `VideoCapture` risulta già "aperto" (`isOpened() == True`).

La correzione applicata è stata un **"riscaldamento" (warm-up) esplicito** subito dopo l'apertura della webcam: una breve pausa più lo scarto dei primi fotogrammi (tipicamente vuoti/non validi):

```python
cap = cv2.VideoCapture(opt.webcam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
assert cap.isOpened(), 'Impossibile aprire la webcam'

time.sleep(1.0)               # lascia inizializzare AVFoundation
for _ in range(5):            # scarta i primi frame, spesso non validi
    cap.read()
```

Questa tecnica è **complementare, non alternativa**, a quella usata per AlphaPose: se la cattura fosse stata (come in AlphaPose) eseguita in un thread separato, il warm-up da solo non sarebbe bastato a risolvere il blocco descritto nella tesina principale (§4.1); viceversa, nello script MMDetection, dove la cattura è già nel thread principale, il problema non era il thread ma solo il "tempo di aggancio" iniziale del driver.

## 6. Selezione del device di calcolo: la cascata CUDA → MPS → CPU e l'instabilità di MPS

Una differenza sostanziale rispetto allo Scenario B: questo script implementa una selezione automatica del device a tre livelli:

```python
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

**Dettaglio non ovvio**: nonostante il Mac in uso sia **Intel** (non Apple Silicon), `torch.backends.mps.is_available()` restituiva `True`. Questo perché il backend **MPS di PyTorch si appoggia all'API Metal**, disponibile anche su Mac Intel dotati di GPU discreta compatibile (in questo caso una AMD Radeon Pro, tipica dei MacBook Pro 16" Intel) — MPS non è quindi sinonimo di "Apple Silicon" come si potrebbe erroneamente assumere. Lo script ha di conseguenza selezionato `device = 'mps'`, innescando una serie di problemi di stabilità:

1. **`NotImplementedError: torchvision::nms non implementato per MPS`** — l'operatore NMS di `torchvision`, su cui si basa il fallback deciso al §2.4, non è disponibile sul backend MPS. Risolto (temporaneamente) impostando la variabile d'ambiente `PYTORCH_ENABLE_MPS_FALLBACK=1`, che forza silenziosamente quell'operazione specifica a girare su CPU pur restando "dentro" un flusso MPS.
2. **`AssertionError` dentro `InstanceData.__getitem__` di MMEngine** — dopo il fallback CPU per NMS, gli indici filtrati (`keep_idxs`) tornavano in un formato/tipo non atteso dal codice di MMDetection quando si rientrava nel contesto MPS. Risolto forzando l'intero **detector** (non solo l'operazione NMS) su CPU esplicitamente, lasciando HybrIK su MPS.
3. **`TypeError: Cannot convert a MPS Tensor to float64`** — MPS non supporta la doppia precisione; gli array NumPy passati a HybrIK (`bbox`, `img_center`) erano in `float64` per default. Risolto forzando `dtype=np.float32` alla creazione dei tensori, prima del trasferimento su device.
4. **Crash irrecuperabile del driver Metal** (`MPSNDArrayDescriptor ... failed assertion`, terminazione del processo con `zsh: abort`) — un'operazione di slicing dentro i layer SMPL di HybrIK (`torch.det(...)` in `lbs.py`), eseguita su MPS, produceva un errore a livello di driver grafico, non intercettabile né correggibile lato Python.

Il punto 4 ha reso evidente che **MPS non è una piattaforma pienamente affidabile per questa pipeline**: HybrIK, sviluppato e testato storicamente solo su CUDA, contiene operazioni la cui implementazione MPS in PyTorch è ancora incompleta o instabile (non un problema del codice della tesina, ma dell'ecosistema PyTorch/MPS per questo tipo di operazioni geometriche). La soluzione finale, definitiva, è stata **abbandonare MPS del tutto** e forzare l'intera pipeline (detector e HybrIK) su CPU:

```python
# --- FORZIAMO LA CPU PER EVITARE I BUG DI MPS SU HYBRIK ---
device = torch.device('cpu')
```

Questo rende di fatto irrilevante, per questo specifico script, la selezione automatica CUDA→MPS→CPU inizialmente implementata: resta nel codice come logica "difensiva" per altri ambienti (es. un Mac Apple Silicon con operazioni pienamente supportate, o una macchina CUDA), ma sul Mac Intel in uso la CPU è l'unica scelta stabile.

## 7. Schema della Pipeline di Esecuzione Locale

*(Invariato rispetto alla relazione precedente.)*

```
[ Webcam frame ]
       │
       ▼
┌───────────────────────────┐
│     MMDetection Core      │ ──► RTMDet-Tiny su CPU (bbox persona, ogni 10 frame)
└───────────────────────────┘
       │
       ▼
┌───────────────────────────┐
│      Torchvision CPU      │ ──► roi_align & NMS (fallback puro PyTorch)
└───────────────────────────┘
       │
       ▼
┌───────────────────────────┐
│   HybrIK (HRNet-W48 cam)  │ ──► CPU (MPS abbandonato per instabilità)
└───────────────────────────┘
       │
       ▼
[ Pacchetto JSON via UDP → Unity ConnectionManager ]
```

## 8. Sintesi delle differenze Linux (GPU) → macOS (CPU), Scenario C

| Aspetto | Versione Linux/CUDA originale | Versione macOS |
|---|---|---|
| Ambiente Python | Non specificato/gestito a parte | `venv_tesina_310` **condivisa** con AlphaPose, con rischio concreto di conflitto NumPy/OpenCV già verificatosi |
| Detector | Faster R-CNN (`faster_rcnn_r50_fpn_1x_coco`) | RTMDet-Tiny (più leggero, adatto a CPU) |
| MMCV | Versione completa con estensioni CUDA | `mmcv-lite` + fallback `torchvision.ops` (roi_align, nms) |
| Controllo versione MMCV in MMDetection | Passa senza intervento | `mmdet/__init__.py` patchato per accettare la versione installata |
| `pytorch3d` | Assunta disponibile/compilabile | Installata da sorgente con `--no-build-isolation` |
| Device | `cuda:{gpu}` obbligatoria, `RuntimeError` se assente | Cascata CUDA→MPS→CPU, MPS scartata per instabilità, forzato CPU |
| Cattura webcam | Diretta, nessun problema noto | `ret=False` al primo frame: risolto con warm-up (`sleep` + scarto frame), **non** con la ristrutturazione a thread singolo usata per AlphaPose |
| Precisione tensori | Non un problema su CUDA | `float64`→`float32` esplicito per compatibilità MPS (poi comunque bypassato forzando CPU) |
| Checkpoint | `hybrik_hrnet.pth` (nome/percorso assunti) | `hybrik_hrnet48_wo3dpw.pth`, percorso corretto nello script |

## 9. Conclusioni

Rispetto allo Scenario B (AlphaPose), il porting dello Scenario C ha richiesto una superficie di intervento diversa. La componente di **ambiente condiviso** ha introdotto un rischio nuovo — non presente quando ogni scenario ha la propria venv isolata — cioè che una dipendenza installata per uno scenario rompa silenziosamente l'altro; questo si è verificato concretamente con NumPy/OpenCV ed è stato containeuto tornando esplicitamente alle versioni già fissate per AlphaPose. La componente di **selezione del device** ha rivelato un problema strutturale specifico di questa pipeline: la disponibilità apparente di MPS anche su hardware Intel (grazie alla GPU discreta compatibile Metal) ha inizialmente portato a un percorso di esecuzione instabile, con crash progressivamente più profondi (da un semplice operatore mancante fino a un crash del driver grafico), risolto solo abbandonando l'accelerazione hardware e restando su CPU pura — la stessa conclusione architetturale già raggiunta per AlphaPose, ma qui raggiunta passando per un problema (MPS) che nello Scenario B non si era mai posto.