# Refactoring Status (2025-10-06)

## Erledigt
- Environments konsolidiert; Reset-Verhalten entspricht wieder dem Original.
- Trainer-Framework (BaseTrainer, LSTM-/Lambda-Trainer, Registry) produktiv; CLI nutzt neue Struktur.
- Config-Hierarchie (base.yaml + Variant-Overrides) eingeführt.
- Logging modularisiert (src/callbacks/): logging.format schaltet JSON <-> CSV, LD-Metriken unter logs/lambda_metrics.jsonl.

## Nächste Schritte
- Map/Payload-Strategie finalisieren (send_map-Flag, Kompression oder Shared Memory); siehe docs/RESET_DEADLOCK_ANALYSIS.md.
- Legacy-Code in src/pipeline_v2/ abbauen.
- Varianten-Validierung (v1–v4) dokumentieren und verbleibende Spezialtrainer-Hooks ergänzen, falls nötig.
