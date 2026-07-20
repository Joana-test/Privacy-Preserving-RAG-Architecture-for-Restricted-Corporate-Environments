# Sichere Informationsbereitstellung in RAG-Systemen für Unternehmensumgebungen

Begleitcode zur Masterarbeit **„Privacy-Preserving RAG Architecture for restricted Corporate Enviroments"**,
Ludwig-Maximilians-Universität München, 2026.

Die Arbeit untersucht, wie Retrieval-Augmented-Generation-Systeme (RAG) in
restriktiven Unternehmensumgebungen Information Leakage auch unter
fehlerhaften Sensitivity-Klassifikationen minimieren können. Dazu kombiniert
das implementierte System zwei Verteidigungslinien:

1. **Authorization-First Retrieval (AFR):** Deterministische Einschränkung
   des Suchraums auf den rollenspezifisch autorisierten Chunk-Pool vor der
   semantischen Suche.
2. **LLM-basierter Security Layer:** Nachgelagerte Few-Shot-Klassifikation
   der generierten Antwort (`SAFE`/`UNSAFE`) gegen die geltende
   Rollenrichtlinie.

Alle Komponenten sind vollständig lokal und CPU-basiert lauffähig
(llama-cpp-python mit GGUF-Modellen, FAISS, keine externen API-Aufrufe).



## Repository-Struktur

```
├── afr/                          # Kern des AFR-Frameworks
│   ├── tagging.py                # Sensitivity- und Domain-Klassifikation (regelbasiert, zur Indexierungszeit)
│   ├── policies.py               # Rollenrichtlinien (attributerweitertes RBAC)
│   ├── pep.py                    # Policy Enforcement Point: Zugriffsentscheidung je Chunk
│   ├── ingest.py                 # Dokumentenintegration, Chunking und FAISS-Indexierung
│   ├── llm_client.py             # Lokale GGUF-Inferenz über llama-cpp-python, JSON-Verarbeitung
│   └── rag.py                    # RAG-Pipeline (AFR und AFR + Security Layer), Prompt-Templates
├── evaluation/                   # Evaluationskorpus, Query-Set und Experimentskripte
│   ├── test_corpus.py            # 24 synthetische Chunks eines Unternehmenskorpus
│   ├── query_set.py              # 41 strukturierte Evaluationsanfragen
│   ├── eval_common.py            # Gemeinsame Funktionen: Fehlerinjektion, Query-Runner
│   ├── eval_afr_baseline.py      # Nur AFR, korrekte Labels (Epsilon = 0)
│   ├── eval_combined_baseline.py # Gepaarter Vergleich AFR vs. AFR+SL auf unkorrumpiertem Korpus
│   ├── eval_misclassification.py # Epsilon-Sweep: beide Varianten unter injizierten Labelfehlern
│   ├── model_comparison.py       # Re-Klassifikation gespeicherter Antworten mit anderen SL-Modellen
│   ├── analysis.py               # Erzeugt alle Tabellen und Grafiken der Evaluation
│   └── results/                  # Ergebnisdateien der Experimente (JSON)
│       ├── afr_baseline/         # Baseline-Durchlauf des AFR-Ansatzes
│       ├── misclassification/    # AFR-Pipeline und AFR+SL-Pipeline unter fehlerhaften Labels
│           ├── eps00
│           ├── eps10
│           ├── eps20
│           ├── eps30
│           ├── eps40
│           ├── eps50
│       ├── model_compare/        # Modellvergleich des Security Layers
│       │   ├── llamaQ4/          # Llama 3.1 8B, Q4_K_M (Basiskonfiguration)
│       │   ├── llamaQ8/          # Llama 3.1 8B, Q8_0
│       │   └── Qwen/             # Qwen2.5-14B, Q4_K_M
│       └── thesis_report/        # Generierter Report und Grafiken
└── README.md                     # Diese Datei

```


### Modelle

Die verwendeten GGUF-Modelle sind nicht Teil des Repositories und müssen
separat bezogen werden (z. B. über Hugging Face):

| Modell | Quantisierung | Verwendung |
|---|---|---|
| Llama 3.1 8B Instruct | Q4_K_M | Generator und Security Layer (Hauptkonfiguration) |
| Llama 3.1 8B Instruct | Q8_0 | Modellvergleich |
| Qwen2.5 14B Instruct | [Quantisierung] | Modellvergleich |


## Experimente reproduzieren

Die Evaluation umfasst 41 Queries × 6 Fehlklassifikationsraten
(ε = 0 bis 0,5) × 3 Seeds (42, 123, 777).

```bash
# Beispiel: Stresstest für eine einzelne Fehlklassifikationsrate
python -m evaluation.eval_misclassification --epsilons 0.2 --seeds 42

# Modellvergleich auf gespeicherten Antworten (Rerun-Regime)
python -m evaluation.model_comparison \
    --model-path ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
    --model-name "Llama-3.1-8B-Q8_0" \
    --output evaluation/results/model_compare/llamaQ8/Q8.json
```


Lange Läufe wurden auf CPU-Servern via `tmux` ausgeführt; ein vollständiger
Stresstest-Durchlauf benötigt auf reiner CPU-Hardware mehrere Stunden.

## Ergebnisse nachvollziehen

Die in Kapitel 5 berichteten Tabellenwerte (Structural/Answer Leak Rate,
FPR, Konfusionsmatrizen, Latenzen, Modellvergleich) werden aus den
Ergebnis-JSONs unter `evaluation/results/` regeneriert:

```bash
python -m evaluation.analysis
```

## Hinweis zum Evaluationskorpus

Der Evaluationskorpus (24 synthetische Chunks, `evaluation/test_corpus.py`)
und das Query-Set (`evaluation/query_set.py`) sind Namboothiri et al. (2026)
entnommen und im Rahmen dieser Arbeit angepasst. Die Anfragen Q41 und Q42
wurden ausgeschlossen, da Prompt Injection außerhalb des betrachteten
Bedrohungsmodells liegt; verwendet werden 41 Anfragen.


## Lizenz

`[MIT / Apache-2.0 / …]` – siehe `LICENSE`.

## Kontakt

J. Fermin – `[j.fermin@campus.lmu.de]`
Betreuung: `[CIS/PD Dr. Stefan Langer]`
