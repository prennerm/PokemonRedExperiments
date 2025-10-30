# Pipeline V2 Learnings

## Deadlock Ursache & Fix
- Schritt 1: Instrumentierung von `run_action_on_emulator` zeigte, dass der Worker immer dann hängen blieb, wenn der Release-Tick (`pyboy.tick(self.act_freq - press_step - 1, render=True)`) ausgeführt wurde.
- Schritt 2: Läufe mit deaktiviertem Render in der Release-Phase (`debug_release_tick_render_off: true`) liefen stabil bis 1 Mio Schritte, ebenso Läufe mit verkürzten Tick-Längen (Press=6, Release=12). Fazit: Rendern beim Loslassen ladet den SDL-/PyBoy-Pfad, auf unserer Umgebung (PyBoy 2.6.0, Spawn-Prozesse) blockiert er.
- Fix für Produktion: Release-Tick ohne Render laufen lassen (Code in `src/environments/base_env.py` → `self.pyboy.tick(release_ticks, render=False)`), optional finaler Tick ebenfalls headless. Trainingslogik bleibt unverändert.

## Shared Memory & Datenfluss
- Motivation: Beobachtungen (Screens, Map) als Referenzen statt als Pickle über Pipes bewegen. `SharedMemoryVecEnv` schreibt Schrittresultate in gemeinsame Buffers; `terminal_obs_ref` verweist beim Done auf bereits vorhandene Daten.
- Resultat: Pipe-Überläufe verschwinden, besonders wichtig in Multi-Worker-Szenarien (>16 Prozesse).
- Fazit: Shared Memory steigert Stabilität/Performance bei hohen Workerzahlen, ersetzt aber nicht den Release-Render-Fix.

## Logging & Sampling
- `log_stats_to_file`: JSONL pro Worker. Praktisch für Debugging, aber I/O-intensiv. Für produktive Läufe deaktivieren oder selten flushen (`stats_flush_freq` groß).
- `logging.save_freq` & `TensorBoard`: Größere Intervalle = weniger Overhead. Für Visualisierungen reichen Stichproben.
- Sampling-Idee: Nur alle N Schritte per `env_method("get_last_observation")` Screens/Heatmaps sammeln, statt jede Episode vollständig mitzuschreiben.

## `send_map_to_agent`
- Fairness: Alle Agent-Varianten (v1–v4) sollen identische Inputs bekommen. `send_map_to_agent` bleibt `false`, damit niemand eine zusätzliche Karte sieht.
- Debugging: Wenn Map-Daten benötigt werden, per Shared Memory oder punktuell abrufen – nicht in die Policy-Observation gemischt.

## Heatmap-Notizen
- Kartenaktualisierung/Heatmap-Logik liegt in `src/utils/map_utils.py` sowie in den Environment-Klassen (`update_explore_map`). Für Visualisierung Runs mit niedriger Frequenz speichern, bei Debug Bedarf hochschalten.

## Performance-Hinweise
- Workeranzahl an physische Kerne koppeln (z. B. 20 auf i7‑14700). Hyperthreading nur nutzen, wenn getestet.
- CPU-Affinität (Parent/Worker auf feste Kerne) reduziert Kontextwechsel.
- Video-/Soundaufnahme deaktivieren (`save_video=False`, `fast_video irrelevant`).

## Zusammenfassung
- Hauptursache des Deadlocks: Rendern im Release-Tick.
- Hauptfix: Release-Tick headless laufen lassen, Logging/I/O schlanker halten.
- Weitere Verbesserungen (Shared Memory, Sampling, CPU-Affinität) steigern Stabilität und Performance, wirken aber nur ergänzend.
