# FPS-Optimierungsnotizen

### Aktuelle Erkenntnisse
- **Release-Tick ohne Render:** Deadlocks verschwinden und PyBoy tickt schneller, wenn die Release-Phase headless läuft (`debug_release_tick_render_off: true`).
- **Logfrequenz senken:** Hohe `stats_flush_freq` und seltenes Speichern (`logging.save_freq`, TensorBoard) halten I/O niedrig und vermeiden fps-Einbrüche.
- **Map/Send reduzieren:** `send_map_to_agent: false`, `pack_bits: true` verringern das Observation-Volumen; nur bei Visualisierungsruns aktivieren.

### Stellschrauben ohne Trainingslogik zu ändern
- **Worker-Anzahl optimieren:** `num_cpu` an physische Kerne koppeln (z. B. 20 auf dem i7‑14700). Hyperthreading erhöht die FPS kaum und kann Scheduling-Kollisionen provozieren.
- **CPU-Affinität setzen:** Parent/Policy auf einen Kern, Worker auf dedizierte Kerne (`taskset`, `os.sched_setaffinity`) → weniger Kontextwechsel.
- **Emissionen bündeln:** Logging asynchron puffern oder nur alle N Episoden schreiben; JSONL und TensorBoard mit gröberen Intervallen speichern.
- **Visualisierung sampeln:** Beobachtungen/Frames nur alle N Schritte per `env_method("get_last_observation")` abfragen statt jede Episode voll mitzuschneiden.

### Optionale Ideen
- **Release/Press-Ticks anpassen:** Kürzere Release-Phase (z. B. 6/12 Frames) erhöht Aktionsfrequenz, muss aber bewusst dokumentiert werden.
- **Turbo-/Null-Renderer:** PyBoy im Turbo-Modus (`set_emulation_speed(0)`) lassen, Videoaufnahmen nur in Diagnose-Runs aktivieren.
- **Tools später prüfen:** PyPy/Cython bringen hier voraussichtlich keinen Vorteil (NumPy/PyBoy sind C-gebunden); Fokus auf Scheduling und I/O lohnt mehr.
