# Refactoring Status (2025-10-06)

## Environments
- ✅ Legacy `red_gym_env_*` analysiert und in `src/environments/` konsolidiert (`BaseRedGymEnv` + Varianten).
- ✅ Beobachtungs- und Reset-Verhalten entspricht dem Original (inkl. Debug-Hooks hinter `debug_reset_timing`).
- 🔄 Karte (map) aktuell wieder aktiv; Deadlock-Mitigations werden separat geplant (siehe `docs/RESET_DEADLOCK_ANALYSIS.md`).

## Trainer & Pipeline-Aufbau
- ✅ Neues Trainer-Framework (`src/trainers/`):
  - `BaseTrainer` kapselt Config-Handling, Run-Setup, VecEnv/Modell/Callback-Erstellung, Training (Resume/Neu).
  - `DefaultTrainer` deckt derzeit alle Varianten ab.
  - CLI (`pipeline_v2.train`) delegiert vollständig an Trainer.
- 🔄 Nächste Schritte:
  - Spezialtrainer für Varianten (z. B. LSTM/LD) registrieren.
  - Callback-/Logging-Modul vereinheitlichen (`src/callbacks/`).
  - Legacy-Code unter `src/pipeline_v2/` sukzessive auflösen, sobald neue Struktur stabil ist.

## Konfigurationen
- 🔄 Config-Hierarchie vereinheitlichen (`base.yaml` + Variant-Overrides).
- 🔄 Neue Flags (z. B. `send_map`) einführen, sobald Payload-Lösung entschieden ist.

## Offene Aufgaben
1. Map-Transport optional/komprimiert gestalten (Deadlock-Fix für Windows, optional auf Linux deaktivierbar).
2. Trainer-Registry erweitern (Variante-spezifische Hooks, Modell-/Policy-Auswahl).
3. Callback/Logging-Module refaktorieren und dokumentieren (JSON/CSV/Summary-Handling).
4. Spezifikation & README fortlaufend aktualisieren, sobald Teilaufgaben abgeschlossen sind.
