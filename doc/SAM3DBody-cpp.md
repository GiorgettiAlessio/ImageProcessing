# SAM3DBody-cpp

**SAM3DBody-cpp** è un software open-source sviluppato interamente in C++. Questo programma elimina del tutto Python durante l'esecuzione e combina modelli di Intelligenza Artificiale ottimizzati su scheda video (GPU) con calcoli matematici veloci su processore (CPU). Il risultato è un sistema capace di tracciare più persone contemporaneamente, creando un "manichino 3D" in tempo reale pronto per essere usato in videogiochi o programmi di animazione.

## 1. Introduzione

Il progetto SAM3DBody-cpp prende un modello AI molto potente (chiamato SAM-3D-Body) e lo trasforma in un programma C++ ultra-veloce capace di calcolare:

1. Dove si trova la persona nella stanza (distanza dalla fotocamera).

2. Come sono piegate le sue articolazioni (braccia, gambe, colonna vertebrale).

3. I movimenti dettagliati delle mani e le espressioni del viso.

4. La forma esatta del corpo (se la persona è alta, magra o robusta).

## 2. Come funziona la Pipeline

```text
[ Video / Foto della Webcam ]
              │
              ▼
┌────────────────────────────────────────────────────────┐
│ PASSO 1: Il Rilevatore (YOLO11)                        │
│ • Riconosce dove sono le persone nella foto            │
└────────────────────────┬───────────────────────────────┘
                         │ Ritaglio della persona
                         ▼
┌────────────────────────────────────────────────────────┐
│ PASSO 2: Il Cervello Visivo (DINOv3 / ViT)            │
│ • Analizza i dettagli della foto (ombre, forme, panni) │
└────────────────────────┬───────────────────────────────┘
                         │ Mappa di informazioni
                         ▼
┌────────────────────────────────────────────────────────┐
│ PASSO 3: Il Traduttore di Posa (Transformer)           │
│ • Trasforma la mappa visiva in 519 numeri di posa      │
└────────────────────────┬───────────────────────────────┘
                         │ Angoli di rotazione e coordinate
                         ▼
┌────────────────────────────────────────────────────────┐
│ PASSO 4: Il Costruttore 3D (Linear Blend Skinning)     │
│ • Muove lo scheletro e applica la "pelle 3D"           │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
[ Output: Avatar 3D / File di Animazione per Blender o Unity ]
```

### Passo 1: Il Rilevatore (YOLO11)

La prima cosa da fare è capire dove sono le persone nell'immagine. Il modulo YOLO analizza la foto e disegna un rettangolo attorno a ogni persona trovata. Se nella stanza ci sono 3 persone, il sistema creerà 3 rettangoli distinti.

### Passo 2: Il Cervello Visivo (DINOv3)

Ogni rettangolo estratto viene inviato a un modello di Intelligenza Artificiale chiamato Vision Transformer (ViT). Questo modello agisce come l'occhio umano: analizza i pixel, capisce la prospettiva, la piega dei vestiti e l'orientamento del busto, e trasforma la foto in una "mappa di indizi visivi".

### Passo 3: Il Traduttore di Posa

Un terzo modulo prende questa "mappa di indizi" e, tramite MHR (Momentum Human Rig) la traduce in:

- 18.439 vertici 3D che compongono la "pelle" del manichino.

- 70 / 127 keypoints usati per creare il file .bvh pronto per l'animazione.

- 519 numeri precisi. Questi numeri rappresentano:
  - La posizione e la rotazione di 127 articolazioni del corpo.
  - Tutti i movimenti delle dita delle mani (54 punti per mano).
  - La forma del corpo e le espressioni del viso.
  - La distanza stimata della persona rispetto alla telecamera.

### Passo 4: Il Costruttore 3D (LBS Engine)

Prende i 519 numeri e li applica a un manichino tridimensionale virtuale composto da 18.439 piccoli triangoli (vertici). Attraverso una tecnica matematica nota come Linear Blend Skinning (LBS), muove le ossa virtuali e "stira" la pelle del modello 3D esattamente come farebbe un software come Maya o Blender.

## 3. Struttura del repository

```
SAM3DBody-cpp/
├── CMakeLists.txt
├── body_mesh.tri             Mesh corporea in stile SMPL per il renderer GL
├── fast_sam_3dbody_frontend.py        Frontend Python leggero (ctypes, nessuna dipendenza extra)
├── fast_sam_3dbody_frontend-3D.py     Frontend Python 3D (ctypes + modello corporeo Python)
├── fast_sam_3dbody_dump_csv.py        Esportatore CSV Python – 70 keypoint MHR per frame
├── two_pass.py                        Lissatore temporale a doppio passaggio
├── ros_demo_webcam.py                 Demo per ROS
├── onnx/                              File di modello per il runtime – scaricare da HuggingFace (vedi sopra)
│   ├── backbone.onnx + .data         ~4.8 GB  Encoder DINOv3-ViT-H/14+
│   ├── decoder.onnx                  ~93 MB   PromptableDecoder a 6 strati
│   ├── pipeline.gguf                 ~5 MB    Head MHR + fotocamera
│   ├── yolo.onnx                     ~81 MB   YOLO11m-pose
│   ├── body_model.lbs                ~27 MB   Dati LBS C nativo
│   ├── correctives.bin               ~33 MB   Blend shape correttivi della posa
│   └── keypoint_mapping.bin          ~8 KB    Mappa indici keypoint MHR-70
├── GraphicsEngine/
│   ├── System/glx3.{h,c}             Gestione finestre GLX
│   └── ModelLoader/                  Caricatore di mesh .tri + trasformazione articolazioni LBS
├── AmMatrix/                         Libreria C leggera per matrici / quaternioni
├── render/
│   ├── fast_sam_3dbody_render.cpp    Renderer overlay mesh OpenGL
│   └── mhr_pose_driver.h             Driver LBS (matrici fotocamera, aggiornamento vertici)
├── scripts/
│   └── build.sh / setup.sh / webcam.sh / video.sh / offline_video.sh
└── src/
    ├── fast_sam_3dbody.h             API pubblica C++
    ├── fast_sam_3dbody.cpp           Implementazione della pipeline
    ├── fast_sam_3dbody_capi.h        API C pura (per ctypes)
    ├── fast_sam_3dbody_capi.cpp
    ├── preprocess.hpp                Crop, normalizzazione, ray_cond, NMS, conversione pose
    ├── bvh_writer.h / bvh_writer.cpp Esportatore motion-capture BVH (multi-persona, mappato alle articolazioni MHR)
    ├── mhr_joint_table.h             Generato da scripts/build_joint_table.py — Nomi articolazioni MHR + genitori
    └── main.cpp                      Eseguibile CLI (--bvh, --out CSV, finestra di overlay dal vivo)
```

## 4. Perché questo progetto è veloce?

La vera innovazione di questo software non è solo l'Intelligenza Artificiale, ma come è stato programmato. Gli sviluppatori hanno usato tre strategie intelligenti:

- Addio a Python, benvenuto C++: Python è un linguaggio "interpretato" (il computer deve tradurlo mentre lo esegue, il che lo rende più lento). Il C++ viene invece tradotto direttamente nel linguaggio macchina del computer prima dell'esecuzione, garantendo la massima velocità.

- ONNX Runtime per la Scheda Video: L'Intelligenza Artificiale pesante (i modelli che analizzano le foto) viene inviata direttamente alla scheda video (GPU NVIDIA) tramite una tecnologia chiamata ONNX. La GPU può fare miliardi di calcoli in parallelo in pochi millisecondi.

- Divisione dei compiti tra Scheda Video e Processore:
  - La GPU fa il lavoro visivo pesante (trovare le persone e analizzare i pixel).

  - La CPU fa i calcoli matematici finali più leggeri (muovere le ossa dello scheletro 3D).

Cosa ci si può fare?

- Esportare File BVH: Il programma salva un file con i movimenti (formato .bvh) che può essere caricato in Blender, Unreal Engine o Unity per far muovere qualsiasi personaggio 3D (come nei videogiochi o nei film d'animazione).

- Controllo Robotico: Invia i dati a un robot per fargli imitare i movimenti umani in tempo reale.

- Fitness e Sport: Analizza la postura di un atleta mentre esegue un esercizio per verificare se la posizione è corretta.

## 5. Perché è importante il runtime C++

Il modello originale gira così:

```text
Python

↓

PyTorch

↓

CUDA

↓

Output
```

SAM3DBody-cpp invece:

```text
C++

↓

ONNX Runtime

↓

ggml

↓

CUDA o CPU

↓

Output
```

## 6. Perché usare ONNX?

Perché ONNX è il formato standard dei modelli.
Una volta esportato il modello non serve più Python.
Questo rende possibile integrare il sistema in Unity

## 7. Diffrenza fra MHR e SMPL.

Sono due modelli per la ricostruzione del corpo umano in 3D.
Il modello più diffuso è SMPL (Skinned Multi-Person Linear Model), che rappresenta il corpo attraverso una mesh tridimensionale deformabile e uno scheletro di 24 articolazioni.

La maggior parte dei moderni algoritmi di Human Mesh Recovery, come HMR2, 4D-Humans, CameraHMR e VIBE, si basa su questo modello o sulle sue estensioni (SMPL-H e SMPL-X), che aggiungono rispettivamente mani e volto.

Una delle principali limitazioni di SMPL è che la mesh e lo scheletro sono fortemente accoppiati: la deformazione della mesh dipende direttamente dallo scheletro e l'intero modello è stato progettato principalmente per la ricostruzione del corpo piuttosto che per applicazioni di animazione. Inoltre, il numero limitato di articolazioni rende meno accurata la rappresentazione di mani, dita e piedi.

Per superare questi limiti, Meta ha introdotto MHR (Momentum Human Rig) all'interno del progetto SAM 3D Body. In questo modello il rig (ossia la struttura scheletrica utilizzata per l'animazione) costituisce l'elemento centrale, mentre scheletro e mesh sono trattati come componenti più indipendenti. MHR utilizza un numero maggiore di articolazioni (circa 70), consentendo una rappresentazione più dettagliata del corpo, delle mani e dei piedi e facilitando operazioni di motion capture, retargeting e animazione.

MHR rappresenta un'evoluzione progettata per offrire maggiore flessibilità e una migliore integrazione con le moderne pipeline di animazione e produzione, risultando particolarmente adatto ad applicazioni in Blender, Unity, Unreal Engine e sistemi di motion capture.
