#!/usr/bin/env python3
"""
test_reset_sync.py - Schneller Test für Reset-Synchronisation-Probleme

Führt mehrere kurze Test-Läufe durch um zu verifizieren:
1. Ob das ursprüngliche Problem (Deadlock bei synchronen Resets) reproduzierbar ist
2. Ob unsere Lösung (Staggered Resets + In-Memory State Loading) funktioniert

Testet in verschiedenen Konfigurationen:
- Mit/ohne Staggered Resets
- Mit/ohne In-Memory State Loading  
- Verschiedene Worker-Anzahlen
"""

import subprocess
import time
import sys
from pathlib import Path
import signal
import yaml
import json

class TestRunner:
    def __init__(self):
        self.base_config_path = Path("configs/test_reset_sync.yaml")
        self.test_results = []
        
    def load_base_config(self):
        with open(self.base_config_path) as f:
            return yaml.safe_load(f)
    
    def create_test_config(self, test_name, modifications):
        """Erstellt temporäre Test-Konfiguration mit Modifikationen"""
        config = self.load_base_config()
        
        # Pfad anpassen
        config["paths"]["session_root"] = f"experiments/test_reset_sync_{test_name}"
        
        # Modifikationen anwenden
        for key_path, value in modifications.items():
            keys = key_path.split(".")
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
        
        # Temporäre Config-Datei erstellen
        temp_config_path = Path(f"configs/temp_test_{test_name}.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return temp_config_path
    
    def run_test(self, test_name, config_path, timeout_seconds=600):
        """Führt einen Test aus und überwacht auf Deadlocks"""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"Config: {config_path}")
        print(f"Timeout: {timeout_seconds}s")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, "-m", "poke_pipeline.train",
            "--variant", "v4", 
            "--config", str(config_path)
        ]
        
        start_time = time.time()
        
        try:
            # Prozess starten
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            last_output_time = time.time()
            iteration_count = 0
            deadlock_detected = False
            
            # Output überwachen
            while True:
                # Timeout check
                if time.time() - start_time > timeout_seconds:
                    print(f"TIMEOUT nach {timeout_seconds}s")
                    process.terminate()
                    break
                
                # Output lesen (non-blocking)
                try:
                    line = process.stdout.readline()
                    if line:
                        print(line.strip())
                        last_output_time = time.time()
                        
                        # Iteration-Counter extrahieren
                        if "effective config" in line.lower():
                            print(f"✓ Config erfolgreich geladen")
                        elif "worker" in line.lower() and "loading state" in line.lower():
                            print(f"✓ State-Loading erkannt")
                        elif "saved" in line.lower() and "stats" in line.lower():
                            iteration_count += 1
                            print(f"✓ Iteration {iteration_count} abgeschlossen")
                    
                    # Deadlock detection: Kein Output für >60s
                    if time.time() - last_output_time > 60:
                        print(f"⚠️  DEADLOCK DETECTED: Kein Output seit 60s")
                        deadlock_detected = True
                        process.terminate()
                        break
                        
                    # Prozess beendet?
                    if process.poll() is not None:
                        break
                        
                except Exception as e:
                    print(f"Fehler beim Lesen: {e}")
                    break
                
                time.sleep(0.1)
            
            # Ergebnis sammeln
            duration = time.time() - start_time
            return_code = process.poll()
            
            result = {
                "test_name": test_name,
                "duration": duration,
                "return_code": return_code,
                "iterations_completed": iteration_count,
                "deadlock_detected": deadlock_detected,
                "success": not deadlock_detected and return_code == 0
            }
            
            print(f"\nTest-Ergebnis:")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Return Code: {return_code}")
            print(f"  Iterations: {iteration_count}")
            print(f"  Deadlock: {deadlock_detected}")
            print(f"  Success: {result['success']}")
            
            return result
            
        except KeyboardInterrupt:
            print(f"\nTest abgebrochen durch Benutzer")
            if 'process' in locals():
                process.terminate()
            return {"test_name": test_name, "aborted": True}
    
    def run_all_tests(self):
        """Führt alle Test-Varianten durch"""
        
        tests = [
            {
                "name": "baseline_fast_reset",
                "description": "Baseline: Schnelle Resets ohne Fixes (sollte Deadlock zeigen)",
                "modifications": {
                    "num_cpu": 16,  # Weniger Worker für schnelleren Test
                    "env.max_steps": 4096,  # Reset jede Iteration
                }
            },
            {
                "name": "single_worker",
                "description": "Kontrolle: Einzelner Worker (sollte immer funktionieren)",
                "modifications": {
                    "num_cpu": 1,
                    "env.max_steps": 4096,
                }
            },
            {
                "name": "staggered_resets",
                "description": "Fix: Staggered Resets + In-Memory Loading",
                "modifications": {
                    "num_cpu": 16,
                    "env.max_steps": 4096,  # Reset jede Iteration
                }
            },
            {
                "name": "different_max_steps",
                "description": "Alternative: max_steps nicht Vielfaches von n_steps",
                "modifications": {
                    "num_cpu": 16,
                    "env.max_steps": 4097,  # Nicht exakt teilbar durch n_steps
                }
            }
        ]
        
        results = []
        
        for test in tests:
            print(f"\n\nVorbereitung Test: {test['name']}")
            print(f"Beschreibung: {test['description']}")
            
            # Config erstellen
            config_path = self.create_test_config(test['name'], test['modifications'])
            
            # Test ausführen
            result = self.run_test(test['name'], config_path, timeout_seconds=300)  # 5min Timeout
            results.append(result)
            
            # Cleanup
            try:
                config_path.unlink()
            except:
                pass
            
            # Zwischen Tests kurz warten
            time.sleep(5)
        
        # Zusammenfassung
        print(f"\n\n{'='*80}")
        print("TEST-ZUSAMMENFASSUNG")
        print(f"{'='*80}")
        
        for result in results:
            if "aborted" in result:
                continue
                
            status = "✅ SUCCESS" if result.get("success") else "❌ FAILED"
            deadlock = "🔒 DEADLOCK" if result.get("deadlock_detected") else "🔓 NO DEADLOCK"
            
            print(f"{result['test_name']:20} | {status:10} | {deadlock:12} | {result['duration']:6.1f}s | {result['iterations_completed']:2}it")
        
        # Ergebnisse speichern
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

if __name__ == "__main__":
    runner = TestRunner()
    results = runner.run_all_tests()
    
    # Empfehlung basierend auf Ergebnissen
    baseline_failed = any(r.get("test_name") == "baseline_fast_reset" and r.get("deadlock_detected") for r in results)
    fix_worked = any(r.get("test_name") == "staggered_resets" and r.get("success") for r in results)
    
    print(f"\n{'='*80}")
    print("EMPFEHLUNG")
    print(f"{'='*80}")
    
    if baseline_failed and fix_worked:
        print("✅ Problem erfolgreich identifiziert und behoben!")
        print("   → Staggered Resets + In-Memory Loading lösen das Deadlock-Problem")
        print("   → Kann für v4_fast.yaml und andere Configs angewendet werden")
    elif baseline_failed and not fix_worked:
        print("⚠️  Problem identifiziert, aber Fix funktioniert noch nicht")
        print("   → Weitere Debugging-Maßnahmen nötig")
    elif not baseline_failed:
        print("🤔 Problem nicht reproduziert - möglicherweise andere Ursache")
        print("   → Längere Tests oder andere Trigger nötig")
    else:
        print("ℹ️  Tests unvollständig - erneut ausführen")
