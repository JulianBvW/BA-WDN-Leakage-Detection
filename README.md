# BA-WDN-Leakage-Detection

Bachelorarbeit an der Universität Bielefeld über das Erkennen von Lecks in Wasserverteilungssystemen.

## Thema

In Zeiten drohender Wasserknappheit ist es wichtig zu Erkennen, ob und wo Annomalien in Form von Lecks in WDNs *(Water Distribution Networks)* auftreten.
Hier gehe ich über mehrere datengetriebene Ansätze um dieses Problem mittels maschinellem Lernen zu Lösen.

- [WNTR](https://wntr.readthedocs.io/en/latest/) als Basis zum Simulieren von WDNs
- N Sensoren darauf verteilt mit Druck-Werten `p(i)`
- Beispiel: Funktions-Ensamble (Gewichtung/ Majority-Voting):
  - Berechne `p_pred(i)` aus anderen Sensoren `p(j), j != i`
  - Annomalie falls `|p(i) - p_pred(i)| > threshold`

## Schwerpunkte

### Model
  - (Lineare) Regression
  - Zeitserien (z.B: Mittelwert über die letzten t Zeitpunkte)
  - RNN (Deep Learning)
  - Kernel-Methoden
  - Hyperparameter
  - Tageszeit nutzen?
### Schwellwert
  - Dynamisch? Fix? Wie optimiert?
  - Statistik: Hypothesentests, Signifikantsintervalle, etc.
### Explainable AI
  - Feature-importance
  - Abhängigkeit von Sensoren
### Datensätze
  - Spielzeugdatensatz (z.B. Net1)
  - [LeakDB](https://github.com/KIOS-Research/LeakDB)
### Evaluations-Metriken
  - Train-Test Split (z.B. Cross validation, ...) 
  - Annomaliedetektion => Zeitpunkte: Vgl. mit Annomalie-Zeitpunkten aus GroundTruth -- TP, FP, TN, FN (Confusion Matrix), Precision, Recall, ...
  - Detection Time (DT)
  - Unteschiedliche Arten von Annomalien (Größe, Zeitdauer, ....)

## Schriftliche Ausarbeitung

### Generelles

- Titel: **TODO**
- [Formelles](https://www.uni-bielefeld.de/fakultaeten/technische-fakultaet/organisation/formulare/) (Eigenständigkeitserklärung, **kein** Uni-Log, Studiengang, Name, Abschluss, Gutachternamen, ...)
- ca. **20-30 Seiten** (Code als Repository)
- Ausdruckt vor Corona, heute i.d.R. digital als .pdf
- Formatierung: Egal  <- Meisten Studis machen LaTeX, wenn Druck dann bitte auf Margins achten! Siehe [Smart Thesis Template](https://github.com/astoeckel/smart-thesis)
- Strukturierung:
  - Titelblatt
  - Eigenständigkeitserklärung (siehe Techfak)
  - Inhaltsverzeichnis (optional: Bildverzeichnis)
  - Inhalt
  - Literaturverzeichnis (Alles Behauptungen müssen durch Quellen belegt; Keine Vorgaben von Quellen)

### Inhalt

1. Einleitung
    - Thema/ Problem/ Aufgabenstellung beschreiben
    - Warum ist das wichtig? Was soll untersucht werden?
    - Ggf. was sind Hypothesen, ...
2. Grundlagen
    - Machine Learning Grundlagen (die für die BA relevant sind)
    - Grundlagen von Annomaliedetektion in WDN
3. Ausarbeitung (2 Kapitel)
    - Modelierung
    - Experimente
    - Diskussion der Ergebnisse
4. Zusammenfassung
    - Ggf. Future Work, ...

---

## Der Plan

Das Testen von (anfänglich) **3** erschiedenen Ansätzen zur Leck-Erkunnung:

1. Die *Baseline*
2. Regression
3. Graph-NN

### Die Baseline

Um eine **untere Schranke** zu definieren gegen welche die verbesserten Ideen zu vergleichen sind, soll ein **naiver** Ansatz mittels simpler **Klassifikation** erstellt werden.
Hierfür werden einfach die momentanen Druck-Werte jedes Sensors als Eingabe benutzt um jeweils einen Klassifikator zu trainieren, welcher für genau einen Sensor `0 (No Leak)` oder `1 (Leak)` bestimmt.

### Regression

Ein Besserer Ansatz wird der sein, mittels **Regression** die Werte jeweiliger Sensoren anhand aller anderen Sensoren zu **predicten**. Da der echte Wert jedes Sensors gegeben ist, lässt sich leicht die Differenz (der *Error*) berechnen. Dieser kann nun mit einem Schwellwert (evtl. dynamisch) verglichen werden und bei **zu großem Fehler** Alarm schlagen.

### Graph-NN

Sogenannte [Graph Neural Networks](https://en.wikipedia.org/wiki/Graph_neural_network) sind NNs welche **speziell für Graphen** einsetzbar sind. Eventuell lässt sich mit solchen Modellen die Leck-Erkennung noch weiter verbessern.

## Die Roadmap

- [X] Aufgabenstellung formulieren
- [X] Git initialisieren
- [ ] Weitere Literatur lesen
- [ ] Analysieren der Daten
- [ ] Testen verschiedener Leck-Arten (Bruch, Schleichend, ...)
- [ ] **Trainingsdaten** generieren (*train-test-split*)
- [ ] **Experiment: Baseline**
  - [ ] Modell aufstellen
  - [ ] Hyperparameter adjustieren
  - [ ] Gewichten
  - [ ] Metriken analysieren
- [ ] **Experiment: Regression**
  - [ ] Modell aufstellen
  - [ ] Hyperparameter adjustieren
  - [ ] Gewichten
  - [ ] Threshold adjustieren
  - [ ] Metriken analysieren
- [ ] **Experiment: Graph-NN**
  - [ ] Was ist das überhaupt?
  - [ ] Keine Ahnung?
- [ ] Weitere **Datensätze** testen
- [ ] **XAI**: Analysieren der Experimente
- [ ] Aufschreiben
  - [ ] Formelles
  - [ ] Bilder generieren
  - [ ] Kapitel schreiben
  - [ ] Prüfen
