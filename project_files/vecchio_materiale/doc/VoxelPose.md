# VoxelPose

VoxelPose prende immagini provenienti da più telecamere, le trasforma in una rappresentazione tridimensionale (voxel grid) e da questa ricostruisce direttamente lo scheletro 3D di ogni persona presente nella scena.
Non utilizza depth camera.
Non utilizza LiDAR.
Non utilizza mesh 3D.
Lavora esclusivamente con immagini RGB provenienti da più telecamere sincronizzate.

## 1. Introduzione

In ambienti affollati, con occlusioni severe o sovrapposizioni visive, l'associazione 2D-su-2D fallisce frequentemente. Se l'associazione sbaglia abbinamento, la triangolazione calcola coordinate 3D errate.

La soluzione VoxelPose: VoxelPose sposta il problema dal 2D al 3D. Anziché associare persone 2D tra viste diverse, proietta i segnali visivi di tutte le fotocamere dentro uno spazio 3D comune. Le occlusioni si risolvono "fondendo" i raggi visivi nel punto esatto dello spazio in cui la persona si trova realmente.

## 2. Come funziona la Pipeline

```text
[ Telecamera 1 ]     [ Telecamera 2 ]     [ Telecamera N ]
       │                    │                    │
       ▼                    ▼                    ▼
┌────────────────────────────────────────────────────────┐
│ PASSO 1: Estrattore Feature 2D (HRNet / ResNet)        │
│ • Genera le heatmap 2D delle articolazioni per vista   │
└────────────────────────┬───────────────────────────────┘
                         │ Heatmap 2D + Calibrazione (K, R, T)
                         ▼
┌────────────────────────────────────────────────────────┐
│ PASSO 2: Unproiezione Volumetrica (CUDA Voxel Kernel)  │
│ • Proietta le heatmap 2D nel Voxel Grid 3D comune      │
│ • Accumula e media le confidenze nello spazio (X,Y,Z)  │
└────────────────────────┬───────────────────────────────┘
                         │ Volume 3D Feature Map
                         ▼
┌────────────────────────────────────────────────────────┐
│ PASSO 3: Localizzazione 3D (Cuboid Proposal Net - CPN) │
│ • 3D CNN individua i centri delle persone nello spazio │
│ • Genera Bounding Cuboids 3D per ogni persona trovata  │
└────────────────────────┬───────────────────────────────┘
                         │ Cuboidi 3D centrati sulle persone
                         ▼
┌─────────────────────────────────────────────────────────┐
│ PASSO 4: Stima della Posa 3D (Pose Regression Net - PRN)│
│ • Estrae sotto-volumi ad alta risoluzione               │
│ • Calcola le coordinate 3D esatte (X, Y, Z) dei joint   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
[ Output: Traiettorie e Scheletri 3D nello spazio metrico globale ]
```

### Passo 1: Estrazione delle Heatmap 2D

Ogni vista $N$ viene elaborata in parallelo da una rete 2D. L'output non sono semplici punti, ma mappe di probabilità (heatmap) che indicano la presenza di ciascuna articolazione nei pixel del fotogramma.

### Passo 2: Unproiezione Volumetrica

Sfruttando le matrici di calibrazione intrinseche ($K$) ed estrinseche ($R, T$), il sistema traccia i raggi nello spazio 3D. Ogni Voxel della griglia $V \in \mathbb{R}^{X \times Y \times Z}$ campiona il valore corrispondente dalle heatmap di tutte le viste disponibili e ne calcola la media.

### Passo 3: Localizzazione 3D

Una rete convoluzionale 3D analizza il volume compresso. Anziché cercare punti isolati, rileva i centri di massa dei corpi umani e genera dei "cuboidi 3D" (scatole tridimensionali) attorno a ogni persona presente nella scena.

### Passo 4: Stima della Posa 3D

Per ogni cuboide identificato, la PRN ritaglia un sotto-volume a risoluzione più elevata. Una seconda 3D CNN calcola con precisione millimetrica la posizione 3D $(X, Y, Z)$ di tutte le 15/17 articolazioni dello scheletro.

## 3. Struttura del Repository

```
voxelpose-pytorch/
├── configs/                     File di configurazione dei dataset e dei modelli
├── data/                        Dataset (Panoptic, Shelf, Campus...)
├── dataset/                     Caricamento e preprocessing dei dataset
├── lib/
│   ├── core/                    Training, validazione e funzioni principali
│   ├── dataset/                 Gestione dei dataset multicamera
│   ├── models/
│   │   ├── pose_resnet.py       Backbone CNN
│   │   ├── voxelpose.py         Implementazione principale di VoxelPose
│   │   ├── cuboid_proposal_net.py
│   │   ├── pose_regression_net.py
│   │   └── project_layer.py     Proiezione delle feature nello spazio 3D
│   ├── utils/                   Funzioni di supporto
│   └── visualization/           Visualizzazione dei risultati
├── output/                      Modelli addestrati e risultati
├── tools/
│   ├── train.py                 Addestramento
│   ├── test.py                  Valutazione
│   └── inference.py             Inferenza
├── requirements.txt
└── README.md

```

## 4. Perché questo progetto è innovativo?

La vera innovazione di VoxelPose consiste nell'abbandonare completamente la classica triangolazione delle pose 2D.

Nei sistemi tradizionali la pipeline è:

- rilevamento delle pose 2D;
- associazione delle persone tra le telecamere;
- triangolazione dei keypoint.

VoxelPose invece:

- costruisce direttamente un volume tridimensionale;
- fonde le informazioni provenienti da tutte le telecamere;
- ricerca le persone direttamente nello spazio 3D;
- stima la posa tridimensionale senza effettuare il matching tra le viste.

Questa strategia rende il modello molto più robusto in presenza di occlusioni e scene affollate.

## 5. Perché usa una Voxel Grid?

La Voxel Grid rappresenta l'intera scena come un volume tridimensionale suddiviso in piccoli cubi.

Ogni voxel raccoglie le informazioni provenienti da tutte le telecamere che osservano quella regione dello spazio.

In questo modo il modello:

- combina automaticamente le diverse prospettive;
- riduce gli errori dovuti alle occlusioni;
- evita il problema dell'associazione tra persone osservate da telecamere differenti.

## 6. Perché non utilizza la triangolazione?

I metodi tradizionali funzionano così:

```
Pose 2D

↓

Matching tra le telecamere

↓

Triangolazione

↓

Pose 3D
```

VoxelPose segue invece una strategia completamente diversa:

```
Feature Maps

↓

Voxel Grid

↓

Cuboid Proposal Network

↓

Pose Regression Network

↓

Pose 3D
```

Eliminando il matching tra viste, il modello riduce una delle principali fonti di errore
della ricostruzione tridimensionale.

## 7. Pregi

I principali vantaggi sono:

molto robusto alle occlusioni;
gestisce bene scene con molte persone;
non richiede il matching tra viste;
produce pose 3D accurate grazie alla fusione delle informazioni nello spazio 3D.

## 8. Limiti

Ha anche diversi limiti importanti:

richiede telecamere calibrate;
necessita di più telecamere sincronizzate;
il volume di voxel richiede molta memoria GPU;
l'inferenza è relativamente lenta rispetto ai metodi più recenti basati su transformer o lifting monoculare.
