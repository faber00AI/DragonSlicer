# DragonSlicer

Slicer 3D per stampanti a resina (MSLA), ispirato al workflow di ELEGOO SatelLite.
Python 3.11 + PySide6 + OpenGL.

    pip install -r requirements.txt
    python main.py

## Struttura
- `core/machine.py`   profili stampante (`machines/*.json`) e parametri di stampa (velocità in mm/min)
- `tools/import_machines.py` importa i profili da `officialPlatform.ini` di SatelLite (19 ELEGOO; `--all` per tutte le marche)
- `core/scene.py`     oggetti, trasformazioni, posa sul piatto, snapshot per annulla
- `core/modeling.py`  primitive, booleani (manifold), taglio con piano (anche inclinato) e tappo, separa parti, ripara, svuota (`hollow`)
- `core/supports.py`  supporti automatici a pilastro (overhang + minimi locali), geometria vettorizzata
- `core/slicer.py`    mesh -> maschere 8 bit per layer (anteprima) e -> intervalli pieni per l'export, senza costruire l'immagine. Taglio vettorizzato: tocca solo i triangoli che attraversano un piano; riempimento a *winding number* (gusci sovrapposti = unione, guscio interno invertito = cavità), con ripiego sul pari/dispari per mesh aperte
- `core/goo.py`       formato di stampa nativo ELEGOO `.goo` V3.0 (lettura header, RLE, scrittura)
- `core/export.py`    export `.goo` (nativo) e ZIP di PNG + job.json
- `ui/scale_dialog.py` domanda di scala all'import se il modello è < 5 mm (cm/pollici/metri)
- `ui/`               viewer 3D (orbita/pan/zoom), anteprima layer, schede Scena / Modella / Supporti e slicing
- `presets/supports.json` preset supporti (Tree_Small/Medium/Large, valori nostri)

### Slider dei layer nella vista 3D
A destra della vista 3D c'è uno slider verticale: taglia il modello allo strato scelto (con il piano usato dallo slicer per quello strato) e
colora la sezione in giallo, così si vede subito l'interno: pareti di un pezzo svuotato, fori, riempimenti. In alto («Tutto») nessun taglio.
Segue la scena e l'altezza layer. La sezione è disegnata con lo stencil buffer, solo guardando da sopra il piano di taglio.

### Svuotare un oggetto
Tab *Modella* → *Mesh* → **Svuota: parete / fori** (parete in mm, diametro dei fori; 0 = nessun foro). Il guscio interno è la superficie a distanza
`parete` dal bordo (griglia di voxel + marching cubes, precisione sotto-voxel, parete ≥ richiesta) aggiunta come guscio invertito. Per ogni cavità si
apre un foro di drenaggio nel punto più basso e uno di sfiato nel più alto, dalla parte con la parete più sottile: una cavità chiusa trattiene
resina non indurita e può scoppiare con la forza di distacco. Cavità più piccole di 4 mm restano piene. Prima dei supporti (i supporti vengono rigenerati).
Da agente/CLI: `hollow_object`.

### Raft dei supporti
Sotto i pilastri che arrivano al piatto c'è un raft (`raft_height`, 0 = nessuno). Può essere **pieno** (con smusso sul bordo) oppure
**forato** (`raft_hole` > 0, default nei preset): una griglia di fori quadrati tra i pilastri, con una cornice (`raft_bar`) lungo il bordo e
attorno alle basi dei pilastri, così il raft resta un pezzo unico. Perché forato: meno resina e meno forza di distacco/ventosa sul FEP,
la resina drena, e a fine stampa il raft si stacca dal piatto e si spezza dai supporti più facilmente. Fori e barre sono arrotondati a
0,5 mm (griglia `RAFT_CELL`); il raft forato non ha smusso. I pilastri poggiano esattamente sul raft: lo slicer riempie even-odd, quindi
gusci sovrapposti bucherebbero lo strato.
Il pulsante **Salva impostazioni supporti** memorizza preset e valori (anche raft e fori); vengono ripristinati a ogni avvio. Aprendo un `.dsl` valgono quelli del progetto.
- `core/resins.py`    profili resina: libreria SatelLite (`presets/resin_library.json`, 1999 profili su 19 stampanti, generata da `tools/import_resins.py`) + profili utente per stampante in `presets/resins_user.json`
- `core/library.py`   libreria personale di modelli (`library/*.obj`)
- `ui/resin_panel.py` tab «Resina»: scelta profilo, tutti i parametri normali/base, Salva / Salva con nome / Elimina

## Da fare / noto
- Export `.goo`: ~1 ms/layer di taglio su una sfera da 330k facce (1020 layer in ~1 s, era ~11 s); non serve il multiprocesso (provato: più lento per l'avvio dei processi)
- `.goo`: token RLE di tipo 1/2 (livelli di grigio/antialiasing) non implementati. L'encoder di SatelLite sta in `EleImageApp.dll`, classe `GOOEncodeWorker` (`ENCODE_FORMAT`, `gooEncodeProcessAvx2`, impostazioni `antiAliasingMaxGray/MixGray`); la DLL conosce anche GOO V4 e V5. Serve un `.goo` esportato da SatelLite con antialiasing attivo per ricavare e verificare il formato byte per byte
- `.goo`: movimenti Z (lift/retract) scritti a 0 come fa SatelLite
- Preset originali `.satsup` (binario Qt) non ancora letti; i nostri sono approssimati
- Supporti ad albero veri (fusione dei fusti), antialiasing

## Formato .goo (ricavato da EleImageApp.dll + 3 file reali, verificato)
Tutto big-endian dopo i campi di testo. Layout: `V3.0` + tag `07 00 00 00 DLP 00`, testi (software 32, versione 24, data 24,
macchina 32, tipo 32, resina 32), aa/grey/blur u16, anteprima 116x116 RGB565 BE + `0D0A`, anteprima 290x290 + `0D0A`,
campi numerici (vedi `HEADER_FIELDS`), valuta[8], indirizzo layer-def u32, grigio u8, layer di transizione u16.
Per layer: def da 64 byte + `0D0A` + lunghezza u32 + blocco `0x55 | token RLE | ~somma(token)` + `0D0A`.
Token RLE: byte = tipo(2 bit: 0 vuoto, 3 pieno) | nByte extra(2 bit) | nibble basso; lunghezza = (extra BE << 4) | nibble.
Piè di pagina: `00 00 00` + tag. Prova: ricostruzione di un file reale identica byte per byte tranne 30 byte di padding dei testi.

## Generare il file
Un solo pulsante verde **GENERA GOO** in cima al pannello laterale (sempre visibile, anche Ctrl+E): apre una finestra dove scegli stampante, resina
(o «Impostazioni attuali» per tenere i valori modificati nel tab Resina) e la destinazione: salvare il file, **inviarlo alla stampante Wi-Fi** (menu delle stampanti trovate, «Cerca in rete», «Avvia la stampa dopo l'invio»), o entrambi. Il tab *Stampante* ha il menu «Resina» (solo il nome del profilo,
sincronizzato col tab Resina) e il suo «Esporta la scena e invia» taglia e spedisce via Wi-Fi. «Anteprima layer» sta nella vista *Layer*.
Caselle numeriche e menu a tendina ignorano la rotella del mouse (`ui/no_wheel.py`): si cambiano solo con le freccette o da tastiera.

## Libreria personale di modelli
Gruppo **Libreria personale** nel tab *Modella*, sotto «Aggiungi oggetto»: i tuoi modelli salvati come `.obj` nella cartella `library/` accanto al programma
(si possono anche copiare lì a mano). Doppio clic o *Aggiungi al piatto* li mette sul piatto accanto alle altre parti; *Salva selezione…* salva l'oggetto
selezionato così com'è posizionato (rotazione e scala incluse, centrato e appoggiato a Z=0); *Importa file…* accetta OBJ/STL/3MF/PLY/OFF e li converte in OBJ;
*Rinomina…* ed *Elimina*. Niente riparazione né ricalcolo di scala: un modello torna dalla libreria com'è entrato. Se esiste già un nome, chiede prima di sovrascrivere.
Agente: `list_library`, `save_to_library` (da file o da un oggetto del progetto), `add_from_library`, `delete_from_library`.

## Profili resina
Stessi parametri dei profili di SatelLite (`resources/resin/material/*.data`): esposizione, attese (prima/dopo esposizione, dopo sollevamento),
ritardo spegnimento luce, sollevamento in due fasi, ritrazione con tratto lento, PWM, ciascuno per layer normali e layer base; più layer base,
transizione, tipo, densità e costo al kg. Velocità in mm/min.
- Tab **Resina**: il menu elenca prima i profili utente (★) e poi la libreria della stampante scelta; modificando un valore il profilo risulta
  «modificato». *Salva* sovrascrive un profilo utente, *Salva con nome…* ne crea uno (anche partendo da uno della libreria, che non si modifica né si cancella).
  Il profilo scelto viene salvato anche nel `.dsl`, con tutti i valori.
- **Piatti di stampa**: le schede sopra il viewer («+» per aggiungerne, tasto destro per rinominare / eliminare) sono piatti dello stesso progetto, ciascuno con i suoi oggetti, supporti e annulla; stampante, resina e parametri dei supporti sono comuni. *Modifica > Sposta selezione su piatto* (o tasto destro sulla scheda di destinazione) sposta gli oggetti selezionati restando nella stessa posizione (Annulla rimette tutto a posto su entrambi i piatti; i supporti dei due piatti vanno rigenerati). **GENERA GOO** e l'invio alla stampante lavorano sul piatto attivo, il cui nome compare sul pulsante. *Salva progetto* salva tutti i piatti nel `.dsl`: il piatto 1 resta nelle chiavi di sempre (CLI/MCP lo vedono come prima, `pack`/`unpack` conservano gli altri), gli altri stanno in `extra_plates` (formato versione 2, scritto solo se i piatti sono più di uno).
- **Testo 3D**: in *Modella → Aggiungi oggetto* scegli la forma **Testo**: scrivi (anche più righe), scegli il font tra quelli installati, altezza delle lettere, **spessore** (estrusione), spaziatura, allineamento e, se vuoi, la **curvatura**: un raggio > 0 avvolge il testo su un cilindro con asse lungo Y (lettere all'esterno o all'interno di un anello). Il risultato è una mesh chiusa come le altre forme (si può unire/sottrarre con i booleani). Da CLI/MCP: `add_text` e `list_fonts`. Serve `fonttools`.
- **Menu tasto destro** (nel 3D, senza trascinare, e sulla lista oggetti): su un oggetto offre Duplica, Rimuovi, Sposta su piatto, Posa/Appoggia sul piatto, Ripara mesh, Specchia, Disponi tutti, Esporta selezione, Nuovo piatto; nello spazio vuoto Importa, Seleziona tutto, Disponi tutti, Nuovo piatto. *Disponi tutti* usa lo stesso algoritmo del CLI (`arrange`) ed è annullabile. Il pulsante **Sposta ▾** accanto a Importa/Duplica/Rimuovi/Annulla fa lo spostamento tra piatti.
- **Progetti SatelLite (`.goo.3mf`)**: *File > Apri progetto*, trascinamento sulla finestra o *Importa* li aprono come progetto (`core/satellite.py`): una parte = un oggetto, con la posizione sul piatto; stampante e profilo resina attivo (esposizioni, layer base, attese, PWM; sollevamento/ritrazione solo se ≠ 0). I supporti di SatelLite (blob `.bms`) non vengono letti: se il file ne aveva, si attivano quelli di DragonSlicer (Tree_Medium) e vengono rigenerati. Il file non viene mai sovrascritto, *Salva* chiede un nuovo `.dsl`. Agente: `import_project` accetta anche il `.goo.3mf`. Il lettore usa solo la libreria standard (nessun `lxml`) e legge in streaming anche file da GB.
- **Movimenti nel file**: come SatelLite, di default il `.goo` porta 0 in sollevamento/ritrazione e vale il profilo della stampante. Spuntando
  «Scrivi sollevamento e ritrazione nel file» (o `embed_motion=true`) i valori vengono scritti. Esposizione, attese, PWM, densità e costo vanno sempre nel file.
- Agente: `list_resins`, `apply_resin`, `save_resin`, `delete_resin`, e `set_print` con tutti i parametri.
- `python tools/import_resins.py` rigenera la libreria da SatelLite.

## Selezione col mouse e gizmo 3D

- **Clic** su un oggetto nel 3D lo seleziona (Ctrl/Maiusc+clic per aggiungere/togliere, clic nel vuoto deseleziona). Il picking è un raggio dal cursore contro i triangoli (Möller–Trumbore, vettorizzato).
- **Modalità** (pulsanti in *Scena › Trasformazione* oppure tasti nel 3D): `Q` seleziona, `W` sposta/ruota, `R` scala.
  - *Sposta / Ruota* (insieme, come Tinkercad): frecce su X (rosso), Y (verde), Z (blu) per spostare e archi di rotazione (un quarto di cerchio che unisce le frecce degli altri due assi) per ruotare attorno agli assi del mondo; Z non scende sotto il piatto.
  - *Scala*: cubetti sugli assi locali dell'oggetto. Con **Scala non uniforme** attivo scala solo l'asse trascinato, altrimenti tutti e tre.
- Accanto a Rot X/Y/Z i pulsanti **45** e **90** impostano l'angolo a quel valore esatto (non lo sommano).
- **Prestazioni del 3D**: la mesh di ogni oggetto va sulla scheda grafica una volta sola (VBO); muovere, ruotare o scalare cambia solo una matrice di trasformazione. Ingombri e quote vengono dai punti del guscio convesso (esatti, in millisecondi). Il picking col mouse lavora in coordinate locali dell'oggetto.
- **Dimensioni finali** (Trasformazione): larghezza X, profondità Y, altezza Z in mm, sempre allineate all'oggetto. Scrivi la misura e premi Invio. Con *Proporzionale* (default) le altre misure si adeguano; spento cambia solo quell'asse (attiva la scala non uniforme).
- Pannello **Allinea** (2+ oggetti): sinistra/centro/destra (X), fronte/centro/retro (Y), base/centro/cima (Z) rispetto all'ingombro del gruppo; **Distribuisci X/Y** (3+ oggetti) per spazi uguali; **Metti A sopra B** (A = oggetto corrente, B = l'altro, opzione *Centra A su B*).
- Trascinando un tirante l'orbita è sospesa; trascinando nel vuoto si orbita come prima. Con più oggetti selezionati lo spostamento vale per tutti; rotazione e scala agiscono su ciascuno attorno al proprio centro.
- Ogni trascinamento è un solo passo di **Annulla**.

## Stampante in Wi-Fi (tab «Stampante»)
Protocollo ricavato da `EleNetManager.dll` di SatelLite (SDCP v3, solo rete locale):
ricerca con UDP `M99999` sulla porta 3000, comandi WebSocket `ws://IP:3030/websocket`, invio file con
`POST http://IP:3030/uploadFile/upload` (pezzi da 1 MiB, campi `TotalSize, Uuid, Offset, Check=0, S-File-MD5, File`),
avvio con il comando 128. Dal tab: cerca la stampante (o scrivi l'IP), «Esporta la scena e invia» (con
«Avvia la stampa dopo l'invio» opzionale), stato, pausa, riprendi, stop, elenco file. L'avvio chiede sempre conferma.
Il tab chiede conferma anche se il modello della stampante è diverso da quello scelto per il taglio.
`python tools/check_goo.py file.goo [altro.goo]`
valida un `.goo` (struttura, checksum, copertura pixel di ogni layer, piè di pagina) e confronta le intestazioni.

## Per l'agente: CLI e server MCP
Stessa pipeline dell'interfaccia (importa → ripara → dimensioni/rotazione → disponi → supporti → `.goo` → stampante), senza UI,
su una cartella di progetto (`project.json` + `meshes/*.ply`). Un solo elenco di funzioni (`core/agent_api.py`) alimenta entrambi:

- **CLI**: `python dragonslide_cli.py --help` (ogni comando stampa un JSON; l'avanzamento va su stderr). Esempio:
  `new_project --project job1` → `add_model modello.stl --project job1` → `set_object 0 --project job1 --size_z 60` →
  `arrange` → `set_supports --preset Tree_Medium` → `check_project` → `send_project 192.168.1.49 --project job1 --name pezzo.goo`.
- **MCP** (stdio): `claude mcp add dragonslide -- python C:\percorso\dragonslide_mcp.py`. 28 strumenti, errori restituiti come `{"error": ...}`.
- **Sicurezza**: avvio, pausa, ripresa, stop e cancellazione file sulla stampante rifiutano senza `confirm=true`; l'agente deve prima chiedere
  all'utente. `send_project` / `printer_upload` non avviano mai la stampa e rifiutano una stampante di modello diverso dal progetto.
  `printer_start` rifiuta anche se la stampante è occupata o il file non c'è.


## Versione portatile (.exe)
`build_exe.ps1` (serve `pip install pyinstaller`) crea in `dist\`:
- `DragonSlicer.exe` — l'interfaccia grafica, un solo file (~330 MB, il primo avvio impiega qualche secondo a decomprimersi)
- `DragonSlicer.exe` (stesso file della GUI) — `DragonSlicer.exe cli <strumento> ...` (stessi comandi di `dragonslide_cli.py`) e `DragonSlicer.exe mcp` (server MCP su stdio)

Macchine e preset sono dentro l'exe; i dati che cambiano stanno accanto all'exe: `library\` (copiata al primo avvio), `presets\resins_user.json`,
`dragonslide.ini` (impostazioni, anche i supporti salvati) e la cartella di progetto predefinita `dragonslide_project`. Basta copiare la cartella per spostare tutto.

## Aggiornamenti automatici
Il menu **Aiuto → Cerca aggiornamenti** (e un controllo all'avvio) legge l'ultima *Release* di GitHub (`faber00AI/DragonSlicer`).
Se la versione è più recente propone il download di `DragonSlicer.exe`, chiude l'app, sostituisce l'exe e la riavvia.
I dati utente (`library\`, `presets\resins_user.json`, `dragonslide.ini`) non vengono toccati.

**Pubblicare una versione:** alza `__version__` in `core/version.py`, ricompila con `build_exe.ps1`, poi crea una Release con tag `vX.Y.Z` allegando `DragonSlicer.exe` (nome esatto).
