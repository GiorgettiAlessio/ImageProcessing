# Estensione della Tesina

## Porting e unificazione delle tre pipeline (SAM3DBody-cpp, AlphaPose+HybrIK, MMDetection+HybrIK) su un'unica macchina macOS Intel

*Capitoli aggiuntivi rispetto alla tesina principale*

---

# 10. Perché un'unica macchina: dal confronto teorico al testing simultaneo

Nella tesina principale i tre scenari — **A** (SAM3DBody-cpp), **B** (AlphaPose + HybrIK) e **C** (MMDetection + HybrIK) — sono presentati e confrontati sul piano architetturale (capitolo 3) e implementativo (capitolo 4), come tre alternative equivalenti per la stima della posa. Nella pratica, però, erano nati in condizioni tutt'altro che equivalenti: gli script Python di AlphaPose e di MMDetection erano stati scritti da upersone separate assumendo **Linux con GPU NVIDIA/CUDA obbligatoria**; il motore C++ di SAM3DBody-cpp era stato pensato a sua volta per l'accelerazione CUDA, con `--cuda DEVICE` come parametro di default. Ogni scenario, insomma, poteva essere mostrato **solo separatamente**, sulla macchina o sull'ambiente in cui era stato originariamente sviluppato da chi se n'era occupato.

Questa frammentazione rendeva impossibile un confronto diretto "alla pari" fra le tre pipeline e, soprattutto, impossibile una demo unica del progetto: per passare da uno scenario all'altro sarebbe stato necessario cambiare macchina, sistema operativo e, in pratica, persona.

Da qui la decisione di tentare di portare **tutti e tre gli scenari sulla stessa macchina**, con un duplice obiettivo:

1. verificare concretamente se fosse possibile costruire un **programma/launcher unico**, in grado di avviare a scelta uno qualsiasi dei tre modelli di detection/pose/mesh-recovery e di collegarlo allo **stesso progetto Unity** di destinazione, senza dover riconfigurare nulla manualmente a ogni cambio di pipeline (v. capitolo 16);
2. poter **testare, confrontare e mostrare** le tre pipeline nella stessa sessione di lavoro, sulla stessa webcam, con lo stesso avatar, invece di dover ricorrere a macchine diverse o a registrazioni separate fatte da persone diverse.

La macchina scelta per l'unificazione è un **MacBook Pro 16" Intel** (privo quindi sia di CUDA). Questa scelta ha imposto un vincolo trasversale a tutti e tre gli scenari, il filo conduttore dell'intero lavoro descritto in questa estensione: nessuno dei tre poteva più contare sull'accelerazione hardware NVIDIA per cui era stato originariamente scritto, e ciascuno ha dovuto essere adattato per funzionare **in modo stabile esclusivamente su CPU**.

Il tentativo di unificazione ha avuto **successo dal punto di vista funzionale**: le tre pipeline girano, ciascuna, sulla stessa macchina, nello stesso ambiente Python condiviso dove applicabile, ed espongono verso Unity lo stesso protocollo UDP con payload JSON già descritto nella tesina principale (capitolo 5). Ha però anche reso evidente un limite pratico non trascurabile, in particolare per lo **Scenario A**: il costo computazionale della pipeline SAM3DBody-cpp — un Vision Transformer pesante, derivato da DINOv2, pensato per GPU di fascia alta — privo di qualsiasi accelerazione hardware si è rivelato tale da rendere l'esecuzione realtime sostanzialmente impraticabile su questa macchina (v. §13.8). Questo è di per sé un risultato di confronto rilevante fra i tre scenari, che integra quanto già discusso nelle conclusioni della tesina principale (capitolo 9).

---

# 11. Organizzazione del progetto sulla macchina di sviluppo

Per rendere possibile la convivenza delle tre pipeline, il progetto è stato organizzato attorno a un'unica cartella radice (di seguito **`Tesina/`**), che raccoglie sia l'ambiente Python condiviso sia i repository clonati dei tre modelli, tenuti distinti dal repository applicativo vero e proprio del gruppo.

```text
Tesina/
│
├── venv_tesina_310/              # ambiente virtuale Python 3.10, condiviso da AlphaPose e MMDetection
│                                  # (e usato anche per gli script Python ausiliari di SAM3DBody-cpp:
│                                  #  download modelli, quantizzazione)
│
├── AlphaPose/                    # repository clonato — Scenario B (detection + pose 2D/3D)
│   └── detector/yolo/data/
│       └── yolov3-spp.weights     # pesi YOLO (~240 MB, scaricati a parte)
│
├── HybrIK/                       # repository clonato — condiviso dagli Scenari B e C
│   ├── pretrained_models/
│   │   └── hybrik_hrnet48_wo3dpw.pth
│   └── model_files/
│       └── smplx/SMPLX_NEUTRAL.npz
│
├── mmdetection/                  # repository clonato — Scenario C (detector)
│   ├── checkpoints/
│   │   └── rtmdet_tiny.pth
│   └── configs/rtmdet/
│       └── rtmdet_tiny_8xb32-300e_coco.py
│
├── SAM3DBody-cpp/                # repository clonato — Scenario A (motore C++ autonomo)
│   ├── CMakeLists.txt
│   ├── build/                    # cartella di build CMake, contiene l'eseguibile
│   │   └── fast_sam_3dbody_run
│   ├── src/                      # cartella dei file sorgenti
│   │   └── main.cpp.             # script Scenario A, adattato per macOS/CPU
│   └── onnx/                     # pesi del modello (backbone, decoder, YOLO, pipeline.gguf, body_model.lbs)
│
└── ImageProcessing/               # repository Git del progetto applicativo del gruppo
    └── test_vale/                 # cartella di lavoro/test personale dell'autrice di questa estensione
        ├── angoliAlphaPose_HybrIK.py       # script Scenario B, adattato per macOS/CPU
        ├── angoliMMdetection_HybrIK.py     # script Scenario C, adattato per macOS/CPU
        └── main_launcher.py                # menu di scelta pipeline (v. cap. 16)
```

Alcune note su questa organizzazione:

- I tre repository dei modelli (`AlphaPose/`, `HybrIK/`, `mmdetection/`, `SAM3DBody-cpp/`) sono stati **clonati localmente**, con i rispettivi percorsi aggiunti a `sys.path` a runtime tramite un context manager dedicato (`in_directory`, v. §15.2) per gli scenari Python, mentre `SAM3DBody-cpp` viene compilato in loco con CMake e produce un eseguibile nativo indipendente dalla venv.
- La cartella `venv_tesina_310/` è l'unico ambiente Python dell'intero progetto: la scelta di condividerla fra AlphaPose e MMDetection (invece di isolare ciascuno scenario nella propria venv) è il punto di partenza del capitolo 12 e la fonte dei principali conflitti di dipendenze incontrati.
- `ImageProcessing/` è il repository applicativo vero e proprio del gruppo (quello che contiene gli script "finali" da lanciare, non le librerie di terze parti), con `test_vale/` come sottocartella di lavoro dedicata ai test della singola autrice.
- Il progetto Unity (contenente `AvatarController.cs`, `IPoseProvider.cs`, i vari Provider, `ConnectionManager.cs`, `UDPReceiver.cs`; v. capitolo 16) è gestito come repository/progetto separato, sviluppato e versionato indipendentemente dai colleghi che si occupano della parte C#; il collegamento fra le due parti avviene esclusivamente a runtime, via socket UDP sulla porta `5065`, e non richiede che i due progetti condividano una cartella comune.

---

# 12. Un'unica venv per tre pipeline: costruzione dell'ambiente condiviso

Il problema di fondo, comune a tutto questo capitolo, è che ospitare **tre pipeline eterogenee** (due basate su PyTorch e installate come pacchetti Python in modalità sviluppo, una scritta in C++ e compilata nativamente) sulla stessa macchina significa **condividere risorse di sistema** che, se toccate per uno scenario, possono rompere silenziosamente gli altri due.

## 12.1 venv Python 3.10 condivisa

```bash
python3.10 -m venv venv_tesina_310
source venv_tesina_310/bin/activate
```

Questa è la venv che, da qui in avanti, ospita **sia** AlphaPose sia MMDetection (Scenario C), oltre agli script Python ausiliari usati per preparare i modelli dello Scenario A (v. §12.3).

## 12.2 Il rischio del riuso: una dipendenza per uno scenario può rompere l'altro

Anziché creare una seconda venv isolata per MMDetection, si è scelto di **riutilizzare `venv_tesina_310`**, già allestita per AlphaPose. Il vantaggio è un solo ambiente da mantenere e un solo interprete da richiamare dal launcher/da Unity; il rischio, esplicito fin dall'inizio, è che **ogni dipendenza installata per uno scenario possa rompere l'altro** se cambia una libreria condivisa (PyTorch, NumPy, OpenCV). Questo rischio è stato posto esplicitamente come vincolo in una delle richieste fatte durante il lavoro e si è **effettivamente materializzato** più volte:

- **Conflitto NumPy/OpenCV fra MMDetection e AlphaPose.** Un tentativo di risolvere un problema di MMDetection reinstallando OpenCV (`pip install opencv-python`) ha aggiornato silenziosamente **NumPy da 1.26.4 a 2.2.6** (dipendenza automatica di OpenCV 5.x) e installato **OpenCV 5.0.0.93**, entrambe versioni potenzialmente incompatibili con i moduli Cython/PyTorch compilati per AlphaPose, dove proprio NumPy ≥2 era già stato identificato come causa di rottura (v. §14.3). La correzione è stato un downgrade esplicito e mirato:

```bash
pip uninstall opencv-python numpy
pip install "numpy<2.0"
pip install "opencv-python==4.9.0.80"
```

- **Compilazione fallita di MMCV.** Il tentativo di installare la versione completa di MMCV (non `mmcv-lite`) ha prodotto `Failed building wheel for mmcv` / `clang failed with exit code 1`: non esisteva un pacchetto binario precompilato (`.whl`) per macOS compatibile con PyTorch 2.2.2, quindi `pip` tentava di **compilare MMCV da sorgente** con `clang`, fallendo perché il codice C++ (`iou3d.cpp`, `nms.cpp`, ecc.) conteneva costrutti incompatibili col compilatore/header disponibili sul Mac.

- **Decisione architetturale: `mmcv-lite` + `torchvision.ops`.** Per evitare del tutto la compilazione di estensioni native su un Mac privo di CUDA, si è deciso di installare **`mmcv-lite`** (niente binari `mmcv._ext`, ma l'intera logica algoritmica in puro Python) e di deviare le operazioni geometriche critiche su **`torchvision.ops`**, già incluso nel PyTorch della venv: `torchvision.ops.roi_align` al posto del RoI Align custom di MMCV, `torchvision.ops.nms` al posto del NMS C++ di MMCV. Il primo utilizzo di `mmcv-lite` ha comunque prodotto un errore prevedibile ma non anticipato — `ModuleNotFoundError: No module named 'mmcv._ext'` — nel punto in cui una parte del codice di MMDetection tentava ancora di caricare le estensioni native assenti; la deviazione su `torchvision.ops` è stata quindi applicata sistematicamente a tutte le operazioni interessate.

- **Bypass del controllo di versione MMCV in MMDetection.** `mmdet/__init__.py` esegue un controllo rigido sulla versione di MMCV installata (`assert mmcv_version >= min and mmcv_version < max`); la versione di `mmcv-lite` installata (2.2.0) risultava superiore al limite massimo atteso, facendo fallire l'`assert` prima ancora di arrivare a un errore legato alla pipeline vera e propria. È stato quindi necessario modificare direttamente il file sorgente dell'installazione di MMDetection, alzando la soglia massima consentita (`<` → `<=`) — un altro caso, dopo `setup.py` di AlphaPose (v. §14.2), di intervento diretto sui file dell'installazione di libreria e non solo sullo script applicativo.

- **Dipendenza mancante: `pytorch3d`.** Non presente nello Scenario B: HybrIK, nella variante usata per lo Scenario C (che carica anche il modello **SMPL-X**), richiede `pytorch3d` per le conversioni fra rappresentazioni di rotazione. `pytorch3d` non distribuisce generalmente wheel precompilate per macOS; l'installazione da sorgente falliva inizialmente con lo stesso pattern già visto per AlphaPose (`ModuleNotFoundError: No module named 'torch'` durante il build isolato di `pip`), risolto allo stesso modo:

```bash
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

## 12.3 Il terzo inquilino: SAM3DBody-cpp e la sua toolchain separata

A differenza di AlphaPose e MMDetection, **SAM3DBody-cpp non è un pacchetto Python**: è un progetto C++ compilato nativamente con **CMake** e collegato a librerie di sistema installate tramite **Homebrew** (`cmake`, `opencv`), completamente al di fuori della venv. Questo ha in parte protetto il progetto da un quarto potenziale conflitto di dipendenze Python — ma ha comunque richiesto attenzione a due punti di contatto con l'ambiente condiviso:

1. **Due OpenCV distinti e non in conflitto fra loro.** La compilazione C++ di SAM3DBody-cpp usa l'**OpenCV di sistema installato via Homebrew** (collegato a CMake tramite `OpenCV_DIR`), mentre gli script Python degli Scenari B e C usano il pacchetto **`opencv-python` installato nella venv**. Si tratta di due installazioni fisicamente separate (una a livello di libreria dinamica di sistema, una a livello di `site-packages` della venv), che infatti non si sono mai scontrate: un caso in cui tenere gli ambienti fisicamente distinti ha evitato a monte lo stesso tipo di conflitto NumPy/OpenCV già sperimentato fra AlphaPose e MMDetection (v. §12.2).
2. **Script Python ausiliari eseguiti dentro `venv_tesina_310`.** Il reperimento e la preparazione dei modelli ONNX di SAM3DBody-cpp (download da HuggingFace, tentativo di quantizzazione) sono stati comunque svolti con script Python lanciati **dentro** la venv condivisa, installandovi pacchetti aggiuntivi (`huggingface_hub`, `onnxruntime`, `onnx`) — scelta di comodo (un solo interprete Python attivo sulla macchina) che non ha creato conflitti evidenti con AlphaPose/MMDetection, trattandosi di pacchetti non condivisi da nessuno dei due scenari Python.

In sintesi, la convivenza dei tre modelli sulla stessa macchina si è retta su **due ambienti distinti tenuti volutamente separati** — la venv Python condivisa (AlphaPose + MMDetection + script ausiliari SAM3D) e la toolchain di sistema Homebrew/CMake (SAM3DBody-cpp) — con un solo vero punto di attrito interno: la condivisione della venv fra AlphaPose e MMDetection, discussa al §12.2.

---

# 13. Scenario A — SAM3DBody-cpp: installazione e adattamento per Mac Intel CPU

Diversamente dagli Scenari B e C, lo Scenario A non richiede l'adattamento di uno script Python preesistente, ma la **compilazione ex novo di un progetto C++** (motore ottimizzato ONNX/GGML, `Fast-SAM-3D-Body`) e la sua successiva configurazione per funzionare su una macchina priva di CUDA. Il lavoro descritto in questo capitolo è stato l'ultimo dei tre a essere completato, a valle di quanto già consolidato per gli Scenari B e C.


## 13.1 Linking fallito: `librt` non esiste su macOS

Con OpenCV finalmente risolto, la compilazione arrivava fino alla fase di **link** della libreria condivisa `libfast_sam_3dbody.dylib`, fallendo lì. La causa era nel `CMakeLists.txt` radice del progetto, che imposta automaticamente `RT_LIB` a `rt` per qualsiasi sistema non Windows:

```cmake
if(WIN32)
    set(MATH_LIB "")
    set(RT_LIB    "")
else()
    set(MATH_LIB m)
    set(RT_LIB    rt)
endif()
```

Su Linux questo linka correttamente `librt` (funzioni POSIX real-time); su **macOS** una libreria `librt` separata non esiste, perché quelle funzioni sono incorporate nella libreria di sistema — il tentativo di collegarla comunque fa fallire il link. La correzione, diretta sul file sorgente del progetto (analogamente a quanto fatto per `setup.py` di AlphaPose e per `mmdet/__init__.py` di MMDetection, v. capitoli 12 e 15), è stata trattare macOS come Windows su questo specifico punto:

```cmake
if(WIN32 OR APPLE)
    set(MATH_LIB "")   # libm è incorporata nelle librerie di sistema su macOS/Windows
    set(RT_LIB    "")  # librt non serve su macOS/Windows
else()
    set(MATH_LIB m)
    set(RT_LIB    rt)
endif()
```

## 13.2 ONNX Runtime: download automatico per la piattaforma sbagliata

Il sistema di build del progetto tenta di scaricare automaticamente ONNX Runtime durante la configurazione CMake; nei log di questa fase compariva però `Downloading ONNX Runtime 1.20.1 (linux-x64)` — un binario per **Linux**, inutilizzabile su macOS. La soluzione è stata scaricare manualmente la build corretta per **macOS Intel (x86_64)** dalla pagina release ufficiale di ONNX Runtime su GitHub e passarne il percorso esplicitamente a CMake:

```bash
curl -L -o onnxruntime-osx.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-x86_64-1.20.1.tgz
tar xf onnxruntime-osx.tgz
cmake .. -DONNX_RUNTIME_DIR=$(pwd)/onnxruntime-osx-x86_64-1.20.1
```

Con questi due interventi (CMakeLists.txt corretto e ONNX Runtime per macOS), il progetto ha finalmente completato la compilazione dell'eseguibile `fast_sam_3dbody_run`.

## 13.3 Pesi del modello: file "segnaposto" Git LFS invece dei binari reali

Il `CMakeLists.txt` stesso avvisa, in fase di configurazione, se i file dei modelli ONNX non sono presenti nella cartella del repository, indicando dove scaricarli (un archivio su HuggingFace, `AmmarkoV/SAM3DBody-cpp-onnx-models`). 
La soluzione è stata abbandonare usare il **client ufficiale HuggingFace Hub**, che risolve correttamente i puntatori LFS scaricando il contenuto binario reale:

```bash
pip install -U huggingface_hub
huggingface-cli download AmmarkoV/SAM3DBody-cpp-onnx-models --local-dir ../onnx
```


## 13.4 File `.data` esterni e nome del file YOLO diverso da quello atteso

Due dettagli pratici, non evidenti dai soli messaggi di errore, hanno richiesto ulteriori iterazioni:

- ogni file `.onnx` di grandi dimensioni (backbone, decoder) contiene **solo la struttura della rete**; i pesi effettivi risiedono nel file `.data` associato, che deve essere scaricato e posizionato accanto all'`.onnx` corrispondente — omettendolo produce errori di lettura fuori dai limiti del file (v. §13.5);
- il file del detector YOLO pubblicato su HuggingFace si chiama `libreyolo9.onnx`, non `yolo.onnx` come assunto per default dal comando di lancio — risolto rinominando il file oppure passando esplicitamente `--yolo ../onnx/libreyolo9.onnx` e, più avanti, `--detector libreyolo` per selezionare correttamente il tipo di detector lato codice.

I file intermedi generati nei tentativi falliti (varianti INT8, "clean", ecc., ciascuna da diversi GB) sono stati infine ripuliti dalla cartella `onnx/` per evitare che il programma ne raccogliesse per errore una versione vecchia o corrotta.

## 13.5 Webcam su macOS: permessi, indice della periferica e falso "freeze"

Una volta risolti i problemi di caricamento dei modelli, lanciare il programma con la webcam sembrava non produrre alcun effetto (nessuna finestra, nessun output, terminale apparentemente bloccato). La diagnosi, per esclusione, ha isolato **tre cause distinte e concorrenti**, in linea con la stessa strategia diagnostica già usata per gli Scenari B e C (script minimale `test_cam.py`/`check_cam.py` con la sola `cv2.VideoCapture`, per capire se il problema fosse la webcam o la pipeline AI):

1. **permessi di Privacy & Sicurezza di macOS** per l'accesso alla fotocamera da parte del Terminale, con la necessità di riavviare completamente il terminale dopo aver concesso il permesso perché diventi effettivo;
2. **indice della periferica**: su questo Mac l'indice `1` (non `0`) corrispondeva alla webcam integrata, un dettaglio emerso solo confrontando log di sessioni diverse ("prima la leggeva anche se era `--from 0`" vs. "rimane fregato anche con `--from 0`" — la differenza vera era altrove, v. punto successivo);
3. **il "freeze" non era affatto un blocco**: nel ciclo principale di `main.cpp`, la finestra `cv::imshow()` viene aperta **solo dopo** che il fotogramma è stato elaborato dalla pipeline AI (`pipeline.process_bgr(...)`); su CPU, senza alcuna accelerazione hardware, un singolo fotogramma attraverso il Vision Transformer pesante poteva richiedere **15-30 secondi o più**. Il programma non era bloccato: stava semplicemente elaborando, con la CPU al 400-500% (verificato da Monitoraggio Attività), per un tempo lunghissimo prima che la finestra potesse comparire.

## 13.6 Ottimizzazione delle prestazioni su CPU

Compreso che il collo di bottiglia reale era il costo computazionale per singolo fotogramma, sono state applicate due categorie di intervento.

**Quantizzazione dinamica mirata.** Un primo tentativo di quantizzazione dinamica generica (`quantize_dynamic` su tutti i layer) ha prodotto un errore diverso da quelli precedenti — `Could not find an implementation for ConvInteger` — perché aveva quantizzato anche i layer convoluzionali iniziali (`patch_embed`), generando nodi interi che il motore CPU di ONNX Runtime non implementa per quell'operazione. La correzione è stata restringere la quantizzazione ai soli layer `MatMul` (i moltiplicatori di matrice, dominanti in un Vision Transformer), lasciando le convoluzioni in piena precisione:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="onnx/backbone_fp32.onnx",
    model_output="onnx/backbone_int8_dynamic.onnx",
    weight_type=QuantType.QInt8,
    op_types_to_quantize=['MatMul'],   # fondamentale per evitare nodi ConvInteger non supportati
)
```

Anche dopo la quantizzazione l'esecuzione è rimasta molto lenta: il modello di base (un Vision Transformer derivato da DINOv2) è progettato per GPU NVIDIA di fascia alta — gli autori del progetto lo dichiarano benchmark a circa 65 ms/frame su una RTX 5090 — e nessuna ottimizzazione software su CPU può avvicinarsi a quel regime.

**Modifiche dirette a `main.cpp`.** Due interventi mirati sul codice sorgente, applicati nel ciclo principale subito dopo la cattura del fotogramma:

```cpp
// Ridimensionamento software forzato: i driver della webcam su macOS
// spesso ignorano cv::VideoCapture::set() per la risoluzione, quindi
// il ridimensionamento viene fatto esplicitamente subito dopo la lettura
if (c.cap_w > 0 && c.cap_h > 0)
{
    cv::resize(frame, frame, cv::Size(c.cap_w, c.cap_h));
}
```

così da poter passare risoluzioni ridotte (`--size 320 240`) e alleggerire realmente il carico sul Transformer, senza che la webcam si rifiuti di aprirsi a quella risoluzione nativa. La finestra `cv::imshow()` con lo scheletro 2D sovraimpresso è stata **deliberatamente mantenuta attiva** (a differenza di quanto inizialmente proposto, cioè disattivarla per risparmiare cicli CPU): serviva per la demo della tesina, e il ridimensionamento software raggiunge comunque l'obiettivo di ridurre il carico di calcolo senza sacrificare il riscontro visivo. In aggiunta, l'esecuzione è stata lanciata con `OMP_NUM_THREADS=4` per limitare/parallelizzare esplicitamente i thread usati da ONNX Runtime.

## 13.7 Sintesi delle differenze Linux/CUDA (assunto) → macOS Intel CPU, Scenario A

| Aspetto | Versione originale (CUDA assunta) | Versione macOS Intel |
|---|---|---|
| Toolchain di build | `cmake`/`make`, `--cuda DEVICE` di default | Stessa toolchain; `cuda_device = -1` di default nel `Config`; `RT_LIB`/`MATH_LIB` svuotati anche per `APPLE` |
| OpenCV | non specificato | Homebrew `opencv@4` (non la 5.x di default, per compatibilità col modulo `calib3d` richiesto da `multiview`) |
| ONNX Runtime | scaricato automaticamente dal build system (`linux-x64`) | scaricato manualmente (`osx-x86_64`) e passato via `-DONNX_RUNTIME_DIR` |
| Pesi del modello | presumibilmente presenti/scaricati correttamente | puntatori Git LFS scaricati per errore con `urllib`, poi corretti con `huggingface-cli` |
| Precisione backbone | non specificata (presumibile fp16/int8 su GPU) | fp32 per compatibilità CPU, poi quantizzazione dinamica `MatMul`-only |
| Webcam | presumibilmente diretta | permessi macOS, indice periferica non banale, apparente "freeze" dovuto al solo tempo di inferenza |
| Prestazioni | GPU (fino a RTX di fascia alta), tempo reale dichiarato dagli autori | CPU, diversi secondi per frame anche dopo quantizzazione e resize software |

---

# 14. Scenario B — AlphaPose + HybrIK: adattamento per Mac Intel CPU

Lo script di partenza (`angoliAlphaPose_HybrIK.py`, versione Linux) implementa una pipeline realtime che cattura il flusso webcam, rileva le persone con un detector YOLO (AlphaPose), stima la posa 3D SMPL con HybrIK (24 giunti, rotazioni + coordinate 3D), converte le rotazioni in angoli utilizzabili da Unity e invia i dati via UDP, mostrando opzionalmente una finestra con lo scheletro sovrapposto al video. Era scritto assumendo un ambiente **Linux con GPU NVIDIA/CUDA** obbligatoria: legge la GPU con `torch.cuda.get_device_name(...)` senza alcun ramo alternativo, usa `torch.autocast(device_type="cuda", ...)` per l'inferenza in FP16, e si appoggia a componenti di AlphaPose (cattura webcam multithread, NMS) dipendenti da estensioni compilate nativamente per quella piattaforma.

Il lavoro si è svolto in due fasi distinte: prima l'**installazione e configurazione dell'ambiente** (questo capitolo, §14.1-14.4), poi il **debugging a runtime** una volta che lo script era finalmente eseguibile ma presentava ancora comportamenti scorretti (§14.5-14.8). In entrambe le fasi il porting è stato incrementale: a ogni tentativo di esecuzione lo script falliva su un problema diverso; ogni errore è stato isolato con test minimi mirati, corretto, e si passava a quello successivo — un approccio "a strati" necessario perché i problemi appartenevano a categorie molto diverse (compatibilità di sistema operativo, differenze di firma fra versioni di libreria, dipendenze binarie mancanti, bug di logica).

## 14.1 Rimozione del vincolo CUDA obbligatorio

Lo script originale interrompeva l'esecuzione con un errore esplicito se CUDA non era disponibile (`raise RuntimeError("CUDA non disponibile...")`). Essendo il Mac Intel privo sia di CUDA sia del backend MPS (disponibile solo su Apple Silicon — anche se, come si vedrà per lo Scenario C al §15.6, questa assunzione va poi ridiscussa), è stato introdotto il ramo alternativo `device = torch.device("cpu")`, con relativa disattivazione automatica di FP16/AMP (ottimizzazioni specifiche GPU, prive di senso su CPU).

## 14.2 Build nativa di AlphaPose: conflitti di dipendenze e patch a `setup.py`

L'installazione di AlphaPose (`pip install -e .` nel repository clonato) ha richiesto diversi interventi in sequenza:

- **`requirements.txt` assente**: i pacchetti sono stati installati singolarmente (`opencv-python`, `numpy`, `pyyaml`, `scipy`, `matplotlib`, `cython`, `easydict`, `json_tricks`, `tensorboard`, `shapely`, `tqdm`, ecc.).
- **`ModuleNotFoundError: No module named 'torch'`** durante la build: `setup.py` importa `torch` per definire le estensioni C++/CUDA, ma l'isolamento di build di `pip` non vede i pacchetti già installati nel venv. Risolto forzando `pip install --no-build-isolation -e .`.
- **Conflitto NumPy 2.x**: risolto fissando `pip install "numpy<2.0"`, con conseguente allineamento anche di OpenCV (`pip install "opencv-python<5.0"`, dato che OpenCV 5.x richiede NumPy ≥2).
- **`OSError: CUDA_HOME environment variable is not set`**: il vero blocco principale. `setup.py` tentava sempre di compilare le estensioni CUDA (incluso il modulo Cython `soft_nms_cpu.pyx`, che nonostante il nome "cpu" veniva comunque legato alla ricerca di CUDA nel processo di build originale), fallendo su un sistema privo di GPU NVIDIA. È stato necessario **riscrivere `setup.py`**, rimuovendo la logica di compilazione CUDA e mantenendo solo l'estensione Cython utilizzabile su CPU (`soft_nms_cpu`), tramite una funzione `get_ext_modules()` semplificata e protetta da `try/except`.
- **Bug residuo nella patch**: la prima riscrittura di `setup.py` passava `extra_compile_args` come `dict` invece che come `list` su macOS (`TypeError: can only concatenate list (not "dict") to list`), corretto normalizzando l'argomento a una lista di flag del compilatore.

## 14.3 Installazione di HybrIK e dipendenza da `chumpy`

HybrIK è stato installato allo stesso modo (`pip install -e .`), con un'installazione complessivamente meno problematica rispetto ad AlphaPose. È stata necessaria la libreria **`chumpy`**, richiesta per caricare correttamente i file `.pkl` (formato legacy dei modelli statistici SMPL), installata insieme a `pycocotools`, `tqdm` ed `easydict`.

## 14.4 Prima versione del fallback NMS (`nms_wrapper.py`)

Con `soft_nms_cpu` come unica estensione compilata (non `nms_cpu`), il codice originale andava in errore con `ImportError: cannot import name 'nms_cpu' from partially initialized module`. È stata scritta una prima versione di `detector/nms/nms_wrapper.py` con import "protetti" in cascata (`nms_cuda` → `nms_cpu` → `None`) e una funzione `nms()` che sceglie il primo backend disponibile — sufficiente a far *importare* il modulo senza crash immediati, ma non ancora un vero fallback funzionante quando **nessuna** delle due estensioni native risultava disponibile, problema risolto definitivamente al §14.6 con `torchvision.ops.nms`.

## 14.5 Reperimento dei modelli pre-addestrati e disallineamento checkpoint/configurazione

Il posizionamento dei pesi pre-addestrati ha comportato diverse difficoltà pratiche: il checkpoint HybrIK atteso (`pretrained_w_cam.pth`) non è stato reperito immediatamente (usato inizialmente un file alternativo, poi sostituito); il modello statistico SMPL (`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) è stato individuato fra i download precedenti e copiato/rinominato nel percorso corretto; i pesi del detector YOLO (`yolov3-spp.weights`, ~240 MB) sono stati scaricati manualmente nella cartella attesa (`detector/yolo/data/`).

Una volta risolti i problemi di percorso, il caricamento del modello falliva con `RuntimeError: Error(s) in loading state_dict`, per due motivi distinti: (1) un primo tentativo con un checkpoint HRNet era stato erroneamente abbinato a una configurazione pensata per una backbone ResNet-34, con nomi/forme dei layer completamente diversi — risolto usando il checkpoint corretto, coerente con la configurazione già in uso; (2) anche col checkpoint corretto restava un *size mismatch* sul tensore `smpl.shapedirs` (300 componenti nel modello usato dal codice, contro i 10 attesi dal checkpoint), risolto troncando a runtime il tensore prima del caricamento dei pesi:

```python
if hasattr(pose_model, 'smpl') and hasattr(pose_model.smpl, 'shapedirs'):
    if pose_model.smpl.shapedirs.shape[-1] == 300:
        pose_model.smpl.shapedirs = pose_model.smpl.shapedirs[:, :, :10]
```

## 14.6 La finestra della webcam non appariva mai

**Sintomo:** lo script catturava correttamente il video e inviava i pacchetti UDP, ma la finestra `cv2.imshow` non compariva mai, senza errori e senza consumo di CPU.

**Causa:** `WebCamDetectionLoader` di AlphaPose cattura i frame webcam in un `threading.Thread` separato, anche in modalità `--sp` (single process). Su Linux questo funziona senza problemi; su **macOS**, il backend di cattura video di OpenCV (AVFoundation) richiede che la cattura avvenga nel *thread principale*, dove gira il run loop di sistema (Cocoa). Se la cattura viene avviata da un thread secondario, `stream.read()` può bloccarsi indefinitamente senza generare eccezioni.

**Soluzione:** sostituzione di `WebCamDetectionLoader` con una classe equivalente (`SyncWebcamLoader`), che replica la stessa logica (stesso detector, stessa trasformazione crop/resize per HybrIK) ma esegue cattura, detection e preprocessing **in modo sincrono nel thread principale**, eliminando il thread di cattura.

## 14.7 Altri bug di runtime

- **Incompatibilità di firma in `SimpleTransform3DSMPL`**: `TypeError: got an unexpected keyword argument 'gpu_device'`, dovuto a una versione della classe che non accetta gli stessi parametri documentati online. Risolto costruendo dinamicamente gli argomenti tramite `inspect.signature`, filtrando solo i parametri effettivamente accettati.
- **`args.gpus` di tipo stringa invece che lista**: `TypeError: '<' not supported between instances of 'str' and 'int'` nel detector YOLO, che si aspetta `args.gpus` come lista di interi. Risolto normalizzando esplicitamente `args.gpus` in entrambi i rami (`[gpu_ids]` su CUDA, `[-1]` su CPU).
- **Estensione NMS non compilata (fallback definitivo)**: `ImportError: Neither nms_cuda nor nms_cpu is available`, risolto aggiungendo un fallback puro PyTorch in `nms_wrapper.py` con `torchvision.ops.nms` (nessuna compilazione richiesta), mantenendo lo stesso contratto di ritorno atteso dal resto del codice.
- **Bug di indicizzazione nel disegno dei giunti**: `IndexError: invalid index to scalar variable`, perché l'output `pred_xyz_jts_29` di HybrIK è un array 1D appiattito (29 giunti × 3 = 87 valori), non `(29, 3)` come assunto. Risolto con `reshape(-1, 3)`.
- **Colori invertiti (RGB/BGR)**: `SyncWebcamLoader` converte il frame da BGR a RGB per coerenza con la pipeline AlphaPose, ma `cv2.imshow` si aspetta BGR — risolto con `cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)` prima della visualizzazione.
- **Scala errata nella visualizzazione dello scheletro**: `pred_xyz_jts_29` è espresso in coordinate 3D metriche root-relative, senza un fattore di conversione fisso e ovvio verso i pixel dell'immagine. Risolto calcolando dinamicamente, frame per frame, un fattore di scala (90° percentile delle distanze dei giunti dal bacino, mappato su ~45% dell'altezza in pixel del bounding box rilevato).

## 14.8 Prestazioni su CPU

Senza GPU, l'intera pipeline (detection YOLO + regressione HybrIK) gira su CPU, con un costo per frame significativamente più alto rispetto a CUDA. Ottimizzazioni applicate: `torch.set_num_threads(os.cpu_count())` (per default, in alcuni ambienti virtuali, PyTorch ne usa solo uno); riduzione della risoluzione di cattura webcam (640×480); parametro opzionale `--det-every-n`, che esegue la detection YOLO solo ogni N frame riusando l'ultimo bounding box, mentre HybrIK continua a girare su ogni frame.

## 14.9 Sintesi delle differenze Linux/CUDA → macOS Intel CPU, Scenario B

| Aspetto | Versione Linux/CUDA originale | Versione macOS/CPU |
|---|---|---|
| Ambiente Python | non gestito a parte | venv (non Conda) con Python 3.10, dopo tentativi falliti con Conda e Python 3.13 |
| Build estensioni native | `setup.py` compila estensioni CUDA (incl. NMS) | `setup.py` riscritto: nessuna estensione CUDA, solo `soft_nms_cpu` via Cython |
| Cattura webcam | thread separato (`WebCamDetectionLoader`) | sincrona nel thread principale (`SyncWebcamLoader`) per compatibilità AVFoundation |
| Device | CUDA obbligatoria, nessun fallback | rilevamento automatico CUDA/CPU, `args.gpus` normalizzato in entrambi i casi |
| Precisione | FP16/AMP su GPU | disabilitata (priva di senso su CPU standard) |
| NMS a runtime | estensione compilata CUDA/CPU | fallback `torchvision.ops.nms`, dopo un primo fallback a cascata insufficiente |
| Modello SMPL | checkpoint e config coerenti per costruzione | allineamento manuale (backbone HRNet/ResNet, troncamento `shapedirs` 300→10) |
| Prestazioni | accelerazione hardware GPU | multi-threading CPU esplicito, risoluzione ridotta, detection sotto-campionata opzionale |

---

# 15. Scenario C — MMDetection + HybrIK: adattamento per Mac Intel CPU

Come per lo Scenario B, lo script di partenza (`angoliMMdetection_HybrIK.py`) era stato scritto per **Linux con GPU NVIDIA/CUDA obbligatoria** (percorsi `/home/alessio/...`, `device = f'cuda:{opt.gpu}'`, `raise RuntimeError("CUDA non disponibile")` in assenza di GPU, detector Faster R-CNN, checkpoint HybrIK `hybrik_hrnet.pth`). Il porting su macOS Intel ha richiesto, oltre alla costruzione dell'ambiente virtuale condiviso con AlphaPose già discussa al capitolo 12, la risoluzione di una catena di errori runtime in parte diversi, nella natura, da quelli dello Scenario B.

## 15.1 Architettura dei repository e core framework

**MMDetection (v3.3.0)** — framework principale basato su PyTorch, clonato dal sorgente originale. Fornisce l'infrastruttura per caricare i file di configurazione, istanziare la rete (BBox Head, Backbone) ed eseguire l'inferenza sui fotogrammi della webcam.

**MMEngine (v1.x)** — libreria di base di OpenMMLab che gestisce il ciclo di vita di training/testing (Runner), il caricamento dei checkpoint e la mappatura dei registri (Registry Tree).

## 15.2 Correzione dei modelli e dei percorsi

Un primo tentativo di avvio ha rivelato che lo script puntava ancora ai percorsi e ai nomi file della versione Linux originale:

- `DET_CONFIG`/`DET_CKPT` puntavano a **Faster R-CNN** (`faster_rcnn_r50_fpn_1x_coco.py`), non al **RTMDet-Tiny** scelto per l'inferenza leggera su CPU — corretto puntando ai file corretti in `configs/rtmdet/` e `checkpoints/rtmdet_tiny.pth`.
- `HYBRIK_CKPT` puntava a un file `hybrik_hrnet.pth` mai scaricato; il checkpoint effettivamente reperito si chiamava **`hybrik_hrnet48_wo3dpw.pth`** (HRNet-W48) — corretto aggiornando il percorso nello script anziché rinominare il file scaricato, per mantenere tracciabile la provenienza del checkpoint.

## 15.3 Stabilizzazione della cattura webcam: un problema diverso da quello di AlphaPose

Con la pipeline finalmente in grado di partire, è emerso lo stesso sintomo dello Scenario B ("la webcam si apre e si richiude subito"), ma con una **causa radice diversa** — un dettaglio interessante perché mostra che lo stesso sintomo, su macOS, può derivare da problemi non correlati. Il test diagnostico isolato (`check_cam.py`) ha dato:

```text
aperta: True
0 ret = False
```

A differenza dello Scenario B — dove la causa era la cattura eseguita in un thread secondario — qui lo script catturava già correttamente nel thread principale (nessun loader threadizzato). Il problema era invece che **il primo `cap.read()` immediatamente dopo l'apertura di `VideoCapture` restituisce sistematicamente `ret = False`**: AVFoundation non ha ancora inizializzato il buffer hardware nell'istante in cui `VideoCapture` risulta già "aperto" (`isOpened() == True`). La correzione applicata è stata un **"riscaldamento" (warm-up) esplicito**: una breve pausa più lo scarto dei primi fotogrammi (tipicamente non validi):

```python
cap = cv2.VideoCapture(opt.webcam_id)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
assert cap.isOpened(), 'Impossibile aprire la webcam'

time.sleep(1.0)               # lascia inizializzare AVFoundation
for _ in range(5):            # scarta i primi frame, spesso non validi
    cap.read()
```

Questa tecnica è **complementare, non alternativa**, a quella usata per AlphaPose: se la cattura fosse stata eseguita in un thread separato, il warm-up da solo non sarebbe bastato a risolvere il blocco (v. §14.6); viceversa, qui, dove la cattura è già nel thread principale, il problema non era il thread ma solo il tempo di aggancio iniziale del driver.

## 15.4 Selezione del device di calcolo: la cascata CUDA → MPS → CPU e l'instabilità di MPS

Una differenza sostanziale rispetto allo Scenario B: questo script implementa una selezione automatica del device a tre livelli:

```python
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```

**Dettaglio non ovvio**: nonostante il Mac in uso sia **Intel** (non Apple Silicon), `torch.backends.mps.is_available()` restituiva `True`. Questo perché il backend **MPS di PyTorch si appoggia all'API Metal**, disponibile anche su Mac Intel dotati di GPU discreta compatibile (in questo caso una AMD Radeon Pro, tipica dei MacBook Pro 16" Intel) — MPS non è quindi sinonimo di "Apple Silicon" come si potrebbe erroneamente assumere (v. anche §14.1, dove questa stessa assunzione era stata inizialmente data per scontata). Lo script ha di conseguenza selezionato `device = 'mps'`, innescando una serie di problemi di stabilità:

1. **`NotImplementedError: torchvision::nms non implementato per MPS`** — risolto (temporaneamente) impostando `PYTORCH_ENABLE_MPS_FALLBACK=1`, che forza silenziosamente quell'operazione a girare su CPU pur restando "dentro" un flusso MPS.
2. **`AssertionError` dentro `InstanceData.__getitem__` di MMEngine** — dopo il fallback CPU per NMS, gli indici filtrati (`keep_idxs`) tornavano in un formato non atteso quando si rientrava nel contesto MPS. Risolto forzando l'intero **detector** (non solo l'operazione NMS) su CPU esplicitamente, lasciando HybrIK su MPS.
3. **`TypeError: Cannot convert a MPS Tensor to float64`** — MPS non supporta la doppia precisione; gli array NumPy passati a HybrIK (`bbox`, `img_center`) erano in `float64` per default. Risolto forzando `dtype=np.float32` alla creazione dei tensori.
4. **Crash irrecuperabile del driver Metal** (`MPSNDArrayDescriptor ... failed assertion`, terminazione del processo con `zsh: abort`) — un'operazione di slicing dentro i layer SMPL di HybrIK (`torch.det(...)` in `lbs.py`), eseguita su MPS, produceva un errore a livello di driver grafico, non intercettabile né correggibile lato Python.

Il punto 4 ha reso evidente che **MPS non è una piattaforma pienamente affidabile per questa pipeline**: HybrIK, sviluppato e testato storicamente solo su CUDA, contiene operazioni la cui implementazione MPS in PyTorch è ancora incompleta o instabile. La soluzione finale è stata **abbandonare MPS del tutto** e forzare l'intera pipeline (detector e HybrIK) su CPU:

```python
# --- FORZIAMO LA CPU PER EVITARE I BUG DI MPS SU HYBRIK ---
device = torch.device('cpu')
```

Questo rende di fatto irrilevante, su questa macchina, la selezione automatica CUDA→MPS→CPU inizialmente implementata: resta nel codice come logica "difensiva" per altri ambienti (un Mac Apple Silicon con operazioni pienamente supportate, o una macchina CUDA), ma sul Mac Intel in uso la CPU è l'unica scelta stabile.

## 15.5 Schema della pipeline di esecuzione locale

```text
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

## 15.6 Sintesi delle differenze Linux/CUDA → macOS Intel CPU, Scenario C

| Aspetto | Versione Linux/CUDA originale | Versione macOS |
|---|---|---|
| Ambiente Python | non gestito a parte | `venv_tesina_310` **condivisa** con AlphaPose, con rischio concreto di conflitto NumPy/OpenCV già verificatosi (v. cap. 12) |
| Detector | Faster R-CNN (`faster_rcnn_r50_fpn_1x_coco`) | RTMDet-Tiny (più leggero, adatto a CPU) |
| MMCV | versione completa con estensioni CUDA | `mmcv-lite` + fallback `torchvision.ops` (roi_align, nms) |
| Controllo versione MMCV in MMDetection | passa senza intervento | `mmdet/__init__.py` patchato per accettare la versione installata |
| `pytorch3d` | assunta disponibile/compilabile | installata da sorgente con `--no-build-isolation` |
| Device | `cuda:{gpu}` obbligatoria, `RuntimeError` se assente | cascata CUDA→MPS→CPU, MPS scartata per instabilità, forzato CPU |
| Cattura webcam | diretta, nessun problema noto | `ret=False` al primo frame: risolto con warm-up (`sleep` + scarto frame), **non** con la ristrutturazione a thread singolo usata per AlphaPose |
| Precisione tensori | non un problema su CUDA | `float64`→`float32` esplicito per compatibilità MPS (poi comunque bypassato forzando CPU) |
| Checkpoint | `hybrik_hrnet.pth` (nome/percorso assunti) | `hybrik_hrnet48_wo3dpw.pth`, percorso corretto nello script |

---

# 16. Il main launcher e l'integrazione Unity

## 16.1 Il problema di partenza

Con tutti e tre gli scenari finalmente funzionanti sulla stessa macchina (capitoli 13-15), restava il problema pratico segnalato fin dall'inizio di questo lavoro (v. capitolo 10): avere **tre script/eseguibili separati da far partire manualmente**, ciascuno affiancato da una propria versione, evoluta indipendentemente, dello script Unity `AvatarController.cs`:

- **Scenario A – SAM3DBody-cpp** (C++): invia rotazioni in **angoli di Eulero** (gradi), con nomenclatura dei giunti personalizzata (es. `hip`, `lShldr`, dita numerate), tramite un `BVHWriter`.
- **Scenario B – AlphaPose + HybrIK** (Python): invia rotazioni come **quaternioni** `(x, y, z, w)` sui 24 giunti standard SMPL.
- **Scenario C – MMDetection + HybrIK** (Python): stesso formato quaternioni/SMPL dello Scenario B, ma con un detector diverso a monte.

Questo aveva portato a **tre script `AvatarController.cs` non compatibili tra loro**, ciascuno con la propria logica di calibrazione automatica, smoothing/dead-zone anti-jitter e gestione della root position — funzionalità utili, ma duplicate e non riutilizzabili fra le pipeline.

## 16.2 Architettura a Provider (Strategy Pattern)

La soluzione adottata è il pattern **Strategy**: invece di tre `AvatarController` distinti, si mantiene **un solo controller**, che delega la lettura e conversione dei dati grezzi a componenti intercambiabili.

- **`IPoseProvider`** — interfaccia comune che ogni pipeline deve implementare (`void ApplyPose(string jsonPayload, Animator animator)`), così che il controller non debba sapere *come* i dati vengono letti o convertiti, solo che riceverà rotazioni pronte da applicare.
- **Un Provider per pipeline** (`SAM3DProvider`, `HybrIKProvider` — condiviso dagli Scenari B e C, che condividono lo stesso formato quaternioni/SMPL), ciascuno responsabile di: parsing del proprio formato JSON specifico, mappatura nome-giunto → osso dell'avatar (es. `ConvertSamBoneName` per la nomenclatura personalizzata di SAM3DBody), calibrazione iniziale e smoothing.
- **`AvatarController`** unificato, che si limita a inoltrare i dati al Provider correntemente attivo.

## 16.3 Selezione dinamica della pipeline via rete

Avere tre Provider pronti non risolve da solo il problema pratico di dover riconfigurare manualmente la scena Unity a ogni cambio di pipeline di test. È stato quindi introdotto un meccanismo di **selezione automatica via rete**, su due lati:

**Lato Python — `main_launcher.py`** (v. capitolo 11 per la sua collocazione in `ImageProcessing/test_vale/`): uno script "cabina di regia" con un menu testuale interattivo, che permette di scegliere quale pipeline avviare. Alla selezione:

1. invia subito un pacchetto UDP di **handshake** (es. `{"pipeline_type": "sam3dbody"}` / `"alphapose-hybrik"` / `"mmdetection-hybrik"`), *prima* di avviare il modello di detection vero e proprio, evitando ritardi dovuti al caricamento dei pesi;
2. lancia come sottoprocesso lo script/eseguibile corrispondente alla pipeline scelta — `angoliAlphaPose_HybrIK.py`, `angoliMMdetection_HybrIK.py`, oppure il binario C++ `SAM3DBody-cpp/build/fast_sam_3dbody_run` con i parametri messi a punto al capitolo 13 (`--cuda -1 --backbone .../backbone_fp32.onnx --detector libreyolo ...`) — che da quel momento invia in streaming i pacchetti di posa sulla stessa porta UDP.

**Lato Unity — `ConnectionManager.cs`:** un componente che resta in ascolto fin dall'avvio (`Play`), con **tutti i Provider disabilitati** di default. Alla ricezione del pacchetto di handshake, individua il Provider corrispondente al `pipeline_type` ricevuto, lo abilita e disabilita gli altri. I pacchetti successivi (dati di posa, non handshake) vengono inoltrati automaticamente al solo Provider attivo. Cambiare pipeline in corsa (fermare lo script/eseguibile Python o C++ in esecuzione, sceglierne un'altra dal menu) fa scattare un nuovo handshake, che disattiva il Provider precedente e attiva quello nuovo "a caldo", senza dover toccare la scena Unity.

## 16.4 Separazione delle responsabilità lato Unity: `UDPReceiver` come sensore puro

Il progetto disponeva già di uno script `UDPReceiver.cs`, con un thread in background sempre in ascolto sulla porta UDP, usato dal vecchio `AvatarController` monolitico. Integrarlo con la nuova architettura ha richiesto di **restringerne la responsabilità**: da "riceve e decide cosa fare con i dati" a puro "sensore" che riceve i byte e li inoltra al `ConnectionManager`, unico punto che interpreta il contenuto (handshake o dati di posa). Poiché `UDPReceiver` riceve i pacchetti su un thread separato, ma Unity richiede che gli oggetti di scena (`Transform`, `Animator`) vengano toccati solo dal thread principale, è stato introdotto un componente di supporto, **`UnityMainThreadDispatcher`**, che mette in coda le azioni ricevute dal thread di rete e le esegue nell'`Update()` del thread principale.

## 16.5 Flusso di esecuzione a runtime

1. **Play** in Unity → `ConnectionManager` in ascolto sulla porta UDP, tutti i Provider disabilitati, avatar fermo.
2. L'utente lancia `main_launcher.py` da terminale e sceglie la pipeline dal menu (SAM3DBody-cpp / AlphaPose+HybrIK / MMDetection+HybrIK).
3. Python invia immediatamente l'handshake UDP con il tipo di pipeline scelto.
4. `UDPReceiver` riceve il pacchetto e lo inoltra al `ConnectionManager`, che riconosce l'handshake, abilita il Provider corrispondente e disabilita gli altri.
5. Viene avviato il vero processo di detection (script Python + webcam, o eseguibile C++ + webcam); da qui in poi ogni frame produce un pacchetto JSON di posa, inviato in streaming sulla stessa porta.
6. Il `ConnectionManager` riconosce questi pacchetti come dati di posa e li inoltra al solo Provider attivo, che applica calibrazione, smoothing e rotazioni alle ossa dell'avatar in tempo reale.

Per cambiare pipeline basta interrompere il processo in esecuzione, tornare al menu di `main_launcher.py` e sceglierne un'altra: nessuna modifica manuale alla scena Unity è necessaria. Con l'aggiunta dello Scenario A a questo meccanismo (i cui Provider e handshake erano già previsti nell'architettura fin dalla sua progettazione, v. §16.2), l'obiettivo posto al capitolo 10 — un unico punto di ingresso in grado di far girare, confrontare e mostrare tutte e tre le pipeline sulla stessa macchina e sullo stesso progetto Unity — è stato raggiunto, con la riserva discussa al capitolo 13 sulle prestazioni dello Scenario A su CPU.

---

# 17. Conclusioni dell'estensione

Il lavoro descritto in questi capitoli ha affrontato tre categorie di problemi molto diverse fra loro, tutte tipiche di un progetto che integra più componenti software eterogenee (Python/CV, C++, Unity/C#) sviluppate da persone diverse, su piattaforme diverse, e qui portate a convivere su un'unica macchina.

La **costruzione dell'ambiente condiviso** (capitolo 12) ha introdotto un rischio non presente quando ogni scenario dispone della propria venv isolata: che una dipendenza installata per uno scenario rompa silenziosamente l'altro. Questo si è verificato concretamente con NumPy/OpenCV fra AlphaPose e MMDetection, ed è stato contenuto tornando esplicitamente alle versioni già fissate; è stato invece evitato a monte fra la venv Python e la toolchain C++ di SAM3DBody-cpp, tenute volutamente separate.

Le fasi di **installazione e configurazione** dei tre scenari (capitoli 13-15) hanno richiesto, in tutti e tre i casi, di intervenire direttamente sui file sorgente delle installazioni di libreria stesse — non solo sugli script applicativi — per aggirare l'assunzione, diffusa in questo tipo di progetti di ricerca, che una GPU NVIDIA con CUDA sia sempre disponibile: dalla riscrittura di `setup.py` di AlphaPose, al patch di `mmdet/__init__.py`, fino alla correzione di `CMakeLists.txt` di SAM3DBody-cpp per l'assenza di `librt` su macOS.

Le fasi di **debugging runtime** hanno coperto categorie distinte di problemi ricorrenti su tutti e tre gli scenari: compatibilità di sistema operativo (threading e cattura video con AVFoundation, permessi della webcam), dipendenze binarie mancanti o scaricate in modo errato (pesi YOLO, estensioni NMS non compilate, puntatori Git LFS al posto dei pesi ONNX reali), e bug di logica pre-esistenti o emersi solo eseguendo davvero il codice (formato dell'output di HybrIK, scala di visualizzazione, tipizzazione di `args.gpus`, nodi ONNX non supportati dal motore CPU).

Sul lato **prestazioni**, i tre scenari mostrano un gradiente netto una volta forzati su CPU: le pipeline basate su HybrIK (Scenari B e C), pur rallentate, restano utilizzabili in tempo quasi reale con le ottimizzazioni descritte (multi-threading, risoluzione ridotta, detection sotto-campionata); lo Scenario A, basato su un Vision Transformer pensato per GPU di fascia alta, resta invece pesantemente sotto la soglia del tempo reale anche dopo quantizzazione dinamica e ridimensionamento software — un risultato che integra concretamente il confronto qualitativo fra i tre approcci già presentato nella tesina principale (capitolo 9) con un dato quantitativo sul costo computazionale in assenza di accelerazione hardware.

Sul lato **Unity** (capitolo 16), il problema non era di piattaforma ma di architettura software: tre pipeline con formati di output incompatibili (quaternioni vs. Eulero, nomenclature dei giunti diverse) avevano prodotto tre script `AvatarController` non riconciliabili. La soluzione — pattern Strategy con interfaccia `IPoseProvider`, selezione dinamica della pipeline attiva tramite handshake di rete, separazione netta fra "ricezione dati" (`UDPReceiver`) e "instradamento/interpretazione" (`ConnectionManager`) — ha permesso di ottenere un singolo punto di ingresso (`main_launcher.py` lato Python, una sola scena Unity lato client) senza dover scegliere quale pipeline "vince" sulle altre, mantenendo tutte e tre disponibili e selezionabili a runtime: l'obiettivo di partenza posto al capitolo 10.

In tutte le fasi, l'approccio efficace è stato lo stesso: isolare un problema alla volta, verificarlo con un test minimo o un log mirato prima di proporre una correzione, e solo dopo passare al problema successivo — invece di provare a correggere "a blocchi" più cause potenziali insieme, il che avrebbe reso impossibile capire quale intervento avesse davvero risolto cosa.