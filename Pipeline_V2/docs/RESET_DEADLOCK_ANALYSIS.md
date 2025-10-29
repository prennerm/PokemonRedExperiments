# Reset Deadlock Analysis

**Hinweis:** Thema wird aktuell nicht weiterverfolgt (geparkt bis nach Linux-Umstieg).

## Cleanup Notes

- 2025-10-08: Removed LSTM-specific first-episode shortening and staggered delays; BaseRedGymEnv/LSTMEnv now reuse the original synchronous reset pattern.
Last updated: 2025-10-06

This document tracks how we reproduced, instrumented, and resolved the reset deadlock in Pipeline V2. Only the key findings and confirmed facts remain; obsolete hypotheses and intermediate detours have been removed for clarity.

---

## 1. Baseline Facts
- Deadlock appears only with multiple parallel environments (SubprocVecEnv). Single-worker runs progress indefinitely.
- Reset()/PyBoy state loading are healthy: every worker loads in <30 ms and executes at least 100 steps in the next episode before any freeze.
- StatsCallback is not the trigger. Even with logging disabled, the hang recurs at the same step counts.

---

## 2. Instrumented Runs

| Date & Run ID | Config Summary | Outcome | Notes |
|---------------|----------------|---------|-------|
| 20251006_093827 | 16 envs, full payload | Freeze after reset=5 | Workers stop producing `step_progress` immediately after reset.
| 20251006_100157 | 16 envs, full payload + `step_progress` logging | Freeze after reset=5 | Confirms all workers complete reset and first action.
| 20251006_105243 | 16 envs, DebugSubprocVecEnv (send logs) | Freeze at `step_count=198` | `worker_send_end` missing for workers 14/15 → block inside `remote.send`.
| 20251006_111716 | 16 envs, DebugSubprocVecEnv | Freeze at `step_count=198` | Payload per send ~22.7 KB; zwei Worker blockieren gleichzeitig.
| 20251006_115425 | 16 envs, DebugSubprocVecEnv | Freeze at `step_count=70` | Gleiches Verhalten, bestätigt konstante Payload-Größe.
| 20251006_120203 | 16 envs, DebugSubprocVecEnv + parent recv timing | Freeze bei `step_count=70`; Parent wartet auf blockierten Worker.
| 20251006_141016 | 16 envs, map=ZeroArray | **Completed** (`total_timesteps=5.0e5`) | Keine Deadlocks → Payload-Größe ursächlich.

### 2.1 Comparison with Original Baseline

| Aspect | Pre-fork (`red_gym_env_v2.py`) | Pipeline V2 (`FrameStackEnv` + `BaseRedGymEnv`) | Impact |
|--------|--------------------------------|-----------------------------------------------|--------|
| Observation dict | `screens`, `health`, `level`, `badges`, `events`, `map`, `recent_actions` | identisch übernommen | kein Unterschied |
| Map size / format | 48×48 `uint8` (repeat-upscaled) | identisch | kein Unterschied |
| Env INFO payload | leer | leer (Debug-Felder nur bei `debug_reset_timing`) | kein Unterschied |
| VecEnv implementation | `SubprocVecEnv` (Linux, fork/forkserver) | `SubprocVecEnv` (Windows, spawn) | **Pipe-Buffer auf Windows kleiner (32 KB)** |
| Parallelism | 64 Envs | z. B. 16 Envs | kleinere Pipe → anfälliger |

Der Hauptunterschied liegt also nicht in der Beobachtung selbst, sondern in Kombination aus Plattform (win32 Pipes), geringerer Buffergröße und gleichzeitigen Resets. Peter Whiddens Code lief auf Linux mit größeren Pipe-Buffern; unser Refactoring plus Windows-Spawns führt zum Überlauf.

---

## 3. Confirmed Root Cause
- Die Beobachtung (insbesondere `map`) erzeugt Pakete von 22–45 KB. Wenn mehrere Worker gleichzeitig senden, füllt sich der Windows-Pipe-Buffer (32 KB) → `remote.send` blockiert.
- Debug-Logs zeigen wiederholt, dass einzelne Sends keinen Abschluss mehr erreichen, obwohl der Parent noch auf sie wartet.
- Sobald wir die Map auf Null reduzieren, verschwinden die Deadlocks → Transport-Volumen ist entscheidend.

---

## 4. Mitigation Plan
1. **Payload optionalisieren / komprimieren**
   - `send_map`-Flag (Default `False`) einführen; bei Bedarf echte Map über separaten Kanal oder komprimiert bereitstellen.
   - Prüfen, ob Events/Felder bit-gepackt werden können, ohne Trainingsinformationen zu verlieren.
2. **Transport optimieren (Langfristig)**
   - Shared Memory oder hierarchische VecEnv, um große Daten effizienter zu verteilen.
   - Asynchrone/gekapselte Map-Updates nur noch in DIagnostic-Modi.
3. **Validierung**
   - Stress-Test mit neuer Konfiguration (`send_map=False` / optional) über ≥5e5 Schritte.
   - Einzel-Env-Läufe mit `send_map=True`, um Visualisierung/Debugging zu ermöglichen.

---

## 5. Remaining Work
- Implement `send_map` + Konfigurationshook.
- Entferne den temporären Nullarray-Hack nach der sauberen Lösung.
- Dokumentiere, wie man Map/Events bei Bedarf sicher einschaltet (Single-Worker oder periodisch).

Test artefacts:
```
experiments/v1_reset_test/20251006_093827/
experiments/v1_reset_test/20251006_100157/
experiments/v1_reset_test/20251006_105243/
experiments/v1_reset_test/20251006_111716/
experiments/v1_reset_test/20251006_115425/
experiments/v1_reset_test/20251006_120203/
experiments/v1_reset_test/20251006_141016/
```

## 6. Shared-Memory Terminal Observation (2025-10-23)

- **Status:** Implementiert (`SharedMemoryVecEnv`, `BaseTrainer`).  
- **Mechanik:** Statt `info["terminal_observation"]` direkt über die Pipe zu schicken, legt jeder Worker die finale Observation in einen separaten Shared-Memory-Puffer und sendet nur noch `terminal_obs_ref = {"version": <counter>}`. Der Parent baut `info["terminal_observation"]` anschließend lokal mit `_gather_terminal_observation()` wieder zusammen.  
- **Vorteil:** Pipe-Bursts fallen weg (nur wenige Bytes pro Reset). Selbst wenn alle 16 Envs gleichzeitig fertig werden, bleibt die Pipe leer.  
- **Cleanup:** `BaseTrainer.run()` schließt die VecEnv jetzt auch bei `KeyboardInterrupt`, damit Shared-Memory-Segmente nach manuellen Abbrüchen korrekt freigegeben werden.  
- **Restproblem:** Trotz entlasteter Pipe frierten Stresstests weiter ein (z. B. `experiments/v4_vecenv_stress/20251024_092118/`). DebugSubprocVecEnv zeigte, dass einzelne Worker (`rank=11`) während `env.step()` blockierten, bevor die Reset-Antwort gesendet wurde – Ursache im Emulator/Action-Handling.
- **Fix (2025-10-24):** Der Deadlock tritt auf, wenn `run_action_on_emulator()` beim Release-Tick rendert. Läufe mit deaktiviertem Release-Render (`experiments/v4_tick_test/20251024_160148/`) bzw. mit angepasster Tick-Länge (`…/20251025_073630/`) laufen stabil bis 1 M Schritte. Produktion wird so angepasst, dass Release-Ticks ohne Render laufen (kein Trainings-Impact, verhindert Hang).

## 7. Archivierte Experimente

- **2025-10-22 – Terminal Observation Slimming (SharedMemoryVecEnv)**  
  - Ziel: Pipe-Bursts bei gleichzeitigen Resets vermeiden, indem `terminal_observation` aus SharedMemoryVecEnv stark verkleinert wird.  
  - Umsetzung: In `SharedMemoryVecEnv` wurde statt der kompletten Beobachtung nur noch ein Summary (`total_timesteps`, `step`, LD-Kennzahlen) zurückgegeben. Folgeversuch: Null-Arrays für alle erwarteten Keys (`screens`, `events`, …), damit SB3s `VecTransposeImage` weiter funktioniert.  
  - Ergebnis: Konzept verworfen. SB3 erwartet weiterhin vollständige Schlüsselstrukturen; Null-Arrays gleicher Form sparen kaum Volumen. Stärker verkleinerte Arrays führten zu `KeyError: 'screens'`.  
  - Status: Änderugen rückgängig gemacht. Nächste Maßnahmen: Statistiken per Worker-Datei bzw. Frequenzreduktion, um Pipe-Last nachhaltig zu senken.
- **2025-10-22 – Per-Worker Stat-Logging (experimentell)**  
  - Ziel: `agent_stats` nicht mehr als große Listen über Pipes schicken, sondern direkt pro Worker auf Disk schreiben.  
  - Ansatz: `BaseRedGymEnv` erhält Flags `log_stats_to_file` und `stats_flush_freq`. Bei aktivem Modus puffern Worker ihre Stats und hängen sie als JSONL in `logs/worker_<rank>_stats.jsonl`. `StatsCallback` prüft das Flag; wenn aktiv, ruft er nur `env_method("flush_stats_buffer")` und verzichtet auf die bisherigen Pipe-Transfers.  
  - Status: Implementiert, Tests stehen noch aus. Untersuchung läuft, ob dies Deadlocks nachhaltig verhindert und trotzdem genug Daten für Analysen liefert.
- **2025-10-22 – Terminal Observation Placeholder (SharedMemoryVecEnv)**  
  - Ziel: `terminal_observation` beim Reset nur noch als minimale Platzhalter-Observation übertragen, um Pipe-Bursts weiter zu reduzieren.  
  - Umsetzung: `SharedMemoryVecEnv.step_wait()` ersetzte `terminal_observation` durch Null-Arrays plus `_summary`.  
  - Status: Verworfen zugunsten der Shared-Memory-Referenz (Abschnitt 6); Platzhalter lösten das Pipen-Problem nicht.

## 8. Offene Ideen

- **Generalisiertes Terminal-Ref-Interface**  
  - Ziel: Das Shared-Memory-Konzept auch für alternative VecEnvs (z. B. Debug-/Packed-Varianten) sowie Upstream-SB3 nutzbar machen.  
  - Fragezeichen: Wie gehen wir mit Plattformen ohne Shared Memory um (reiner Fork/Posix-Shared-Memory)? Möglicherweise Fallback auf komprimierte Dicts erforderlich.  
  - Status: Konzeptphase – Abschnitt 6 dokumentiert den Ist-Zustand für `SharedMemoryVecEnv`.
EOS
