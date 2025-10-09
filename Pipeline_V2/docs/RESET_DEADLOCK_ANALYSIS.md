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
| 20251006_105243 | 16 envs, DebugSubprocVecEnv (send logs) | Freeze at `step_count=198` | `worker_send_end` missing for workers 14/15 â†’ block inside `remote.send`.
| 20251006_111716 | 16 envs, DebugSubprocVecEnv | Freeze at `step_count=198` | Payload per send ~22.7 KB; zwei Worker blockieren gleichzeitig.
| 20251006_115425 | 16 envs, DebugSubprocVecEnv | Freeze at `step_count=70` | Gleiches Verhalten, bestÃ¤tigt konstante Payload-GrÃ¶ÃŸe.
| 20251006_120203 | 16 envs, DebugSubprocVecEnv + parent recv timing | Freeze bei `step_count=70`; Parent wartet auf blockierten Worker.
| 20251006_141016 | 16 envs, map=ZeroArray | **Completed** (`total_timesteps=5.0e5`) | Keine Deadlocks â†’ Payload-GrÃ¶ÃŸe ursÃ¤chlich.

### 2.1 Comparison with Original Baseline

| Aspect | Pre-fork (`red_gym_env_v2.py`) | Pipeline V2 (`FrameStackEnv` + `BaseRedGymEnv`) | Impact |
|--------|--------------------------------|-----------------------------------------------|--------|
| Observation dict | `screens`, `health`, `level`, `badges`, `events`, `map`, `recent_actions` | identisch Ã¼bernommen | kein Unterschied |
| Map size / format | 48Ã—48 `uint8` (repeat-upscaled) | identisch | kein Unterschied |
| Env INFO payload | leer | leer (Debug-Felder nur bei `debug_reset_timing`) | kein Unterschied |
| VecEnv implementation | `SubprocVecEnv` (Linux, fork/forkserver) | `SubprocVecEnv` (Windows, spawn) | **Pipe-Buffer auf Windows kleiner (32 KB)** |
| Parallelism | 64 Envs | z.â€¯B. 16 Envs | kleinere Pipe â†’ anfÃ¤lliger |

Der Hauptunterschied liegt also nicht in der Beobachtung selbst, sondern in Kombination aus Plattform (win32 Pipes), geringerer BuffergrÃ¶ÃŸe und gleichzeitigen Resets. Peter Whiddens Code lief auf Linux mit grÃ¶ÃŸeren Pipe-Buffern; unser Refactoring plus Windows-Spawns fÃ¼hrt zum Ãœberlauf.

---

## 3. Confirmed Root Cause
- Die Beobachtung (insbesondere `map`) erzeugt Pakete von 22â€“45 KB. Wenn mehrere Worker gleichzeitig senden, fÃ¼llt sich der Windows-Pipe-Buffer (32 KB) â†’ `remote.send` blockiert.
- Debug-Logs zeigen wiederholt, dass einzelne Sends keinen Abschluss mehr erreichen, obwohl der Parent noch auf sie wartet.
- Sobald wir die Map auf Null reduzieren, verschwinden die Deadlocks â†’ Transport-Volumen ist entscheidend.

---

## 4. Mitigation Plan
1. **Payload optionalisieren / komprimieren**
   - `send_map`-Flag (Default `False`) einfÃ¼hren; bei Bedarf echte Map Ã¼ber separaten Kanal oder komprimiert bereitstellen.
   - PrÃ¼fen, ob Events/Felder bit-gepackt werden kÃ¶nnen, ohne Trainingsinformationen zu verlieren.
2. **Transport optimieren (Langfristig)**
   - Shared Memory oder hierarchische VecEnv, um groÃŸe Daten effizienter zu verteilen.
   - Asynchrone/gekapselte Map-Updates nur noch in DIagnostic-Modi.
3. **Validierung**
   - Stress-Test mit neuer Konfiguration (`send_map=False` / optional) Ã¼ber â‰¥5e5 Schritte.
   - Einzel-Env-LÃ¤ufe mit `send_map=True`, um Visualisierung/Debugging zu ermÃ¶glichen.

---

## 5. Remaining Work
- Implement `send_map` + Konfigurationshook.
- Entferne den temporÃ¤ren Nullarray-Hack nach der sauberen LÃ¶sung.
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


