# Refactoring Status (2025-10-06)

## Environments
- ✅ Legacy `red_gym_env_*` analysiert und in `src/environments/` konsolidiert.
- ✅ Beobachtungen/Reset-Verhalten entsprechen dem Original.
- 🔄 Map-Übertragung: Deadlock auf Windows → optionale/komprimierte Lösung geplant (siehe RESET_DEADLOCK_ANALYSIS.md).

## Trainers & Pipeline
- ✅ `BaseTrainer` + `DefaultTrainer` in `src/trainers/` implementiert; CLI delegiert an Trainer.
- ✅ Trainer-Registry (`TRAINER_REGISTRY`) vorhanden; Varianten v1–v4 aktuell auf DefaultTrainer gemappt.
- 🔄 Spezialtrainer für v3 (LSTM) und v4 (Lambda Discrepancy) vorbereiten (z. B. Policy/Callback Overrides).
- 🔄 Callback-/Logging-Modul vereinheitlichen (Stats, TensorBoard etc.).
- 🔄 Legacy-Code unter `src/pipeline_v2/` sukzessive entfernen, sobald neue Struktur stabil ist.

## Konfigurationen
- 🔄 Config-Hierarchie vereinheitlichen (`base.yaml` + Variant-Overrides).
- 🔄 Neue Flags (z. B. `send_map`) einführen, sobald Payload-Lösung feststeht.

## Offene Tasks
1. Spezialtrainer (LSTM/Lambda) registrieren und – bei Bedarf – Varianten-spezifische Hooks implementieren.
2. Kurze Validierungsläufe für v1–v4 mit neuem Trainer-Setup.
3. Callback/Logging-Aufräumarbeiten planen (Dokumentation ergänzen).
4. Map-Option finalisieren (Payload-Reduktion oder Shared-Memory-Lösung).

## Validation (2025-10-07)
- Quick smoke tests for v1–v4 (reduced timesteps) succeed with DefaultTrainer.
- LSTMEnv fix (`_init_state_bytes = None`) applied; v3/v4 now instantiate correctly.
