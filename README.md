# ImageProcessing - Tesina
_Alessio Giorgetti s345637_
_Martina Pasquinelli s363700_
_Valentina Giordano s364229_



---

In questo Repository si è cercato di caricare una prima analisi di progetto e una prima stesura della tesina (_"Un sistema di motion capture markerless basato su computer vision"_).

Il lavoro è stato svolto nella seguente maniera:
- ricerca e analisi dei modelli di _human detection_, sia forniti tramite mail, sia ricercati in maniera aggiuntiva sul web.
- selezione dei modelli più efficienti e "fattibili" per essere scaricati e testati.
- scelta della pipeline di lavoro sulla condivisione dei dati di output per animare un avatar su Unity in tempo reale.
- scrittura del codice per i 3 modelli scelti.
- creazione del progetto su Unity e scrittura del codice C# per la ricezione dei dati tramite rete.
- scrittura della tesina e testing.

Al momento la tesina si trova in uno stato iniziale, sono stati scritti tutti i passaggi svolti e analizzati, con l'aggiunta di alcune estensioni proposte per un'eventuale prolungamento del progetto.

Inoltre, è stato testato un tentativo di unificazione del progetto stesso, per creare un'unico programma che permettesse l'analisi ed il confronto dei tre modelli scaricati su un unico sistema operativo, per valutare prestazioni e caratteristiche a confronto a partire dalla stessa architettura.

Il Repository è organizzato in questo modo:
```text
ImageProcessing/projects_files
│
├── 1_alphapose+Hybrik/            # codice per lo sviluppo del primo modello
│                                  
│
├── 2_mmDec+Hybrik+Unity/          # codice per lo sviluppo del secondo modello
│   
│
├── 3_sam+Unity/                   # codice per lo sviluppo del terzo modello
│   
│       
├── materiale/                     # tutto il materiale utile alla nostra ricerca
│                                  # non necessariamente utile per la ricerca finale
│
├── test_vale/                     # cartella con i file di estensione della tesina
│   
│
├── Unity/                         # file di progetto Unity
│   
│
└── video/                         # video di test          
```

```
ImageProcessing
│
├── tesina.md                     # Documento per la consegna
│   
│
├── estensione_tesina.md          # Estensione della tesina
│   
│
└── README.md   
```
