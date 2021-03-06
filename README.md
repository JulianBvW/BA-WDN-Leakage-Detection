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

## Theorie

### Arten der Leck-Detektion

Die Detektion lässt sich in zwei Arten einstufen, aktive und passive Verfahren.

- **Aktives Verfahren (_Hardware-based_).** Hier geht es darum, mittels spezieller Technik wzB. Schallgeneratoren oder Kameras aktiv zu Anomalien zu suchen. Jeder proaktive Suchvorgang steht hier idR. für sich.
- **Passive Verfahren (_Model-based_).** Dies sind kontinuierliche Verfahren, welche durch im WDN angebrachte Sensoren durchgehend den aktuellen Status überwachen und bei Anomalien Alarm schlagen. Hierfür wird meistens eine Art virtuelles Netzwerk simuliert und mit dem Real-Zustand verglichen. Diese Simulation kann entweder aufgrund perfekt-kalibrierten, hydraulischen Modellen basieren (_Hydraulic-model-based_) oder mittels Langzeitdaten hervorgesagt werden (_Hydraulic-Measurement-based/ Data-driven_).

In dieser Arbeit geht es um die **datengetrieben Verfahren** (also passive, auf Messwerten basierenden Verfahren) und wie dieses Lernen eines simulierten Netzwerkes aussehen kann.

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
  - BattleDIM (L-Town Netzwerk)  <- Real world!
### Evaluations-Metriken
  - Train-Test Split (z.B. Cross validation, ...) 
  - Annomaliedetektion => Zeitpunkte: Vgl. mit Annomalie-Zeitpunkten aus GroundTruth -- TP, FP, TN, FN (Confusion Matrix), Precision, Recall, ...
  - Detection Time (DT)
  - Unteschiedliche Arten von Annomalien (Größe, Zeitdauer, ....)

## Schriftliche Ausarbeitung

### Generelles

- Titel: **TODO**
- Formelles (Eigenständigkeitserklärung, **kein** Uni-Log, Studiengang, Name, Abschluss, Gutachternamen, ...)
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
3. ~Recurrent- und Graph-NN~
4. ~Outlier Detection und TSF~

### Die Baseline

Um eine **untere Schranke** zu definieren gegen welche die verbesserten Ideen zu vergleichen sind, soll ein **naiver** Ansatz mittels simpler **Klassifikation** erstellt werden.
Hierfür werden einfach die momentanen Druck-Werte jedes Sensors als Eingabe benutzt um jeweils einen Klassifikator zu trainieren, welcher für genau einen Sensor `0 (No Leak)` oder `1 (Leak)` bestimmt.

### Regression

Ein Besserer Ansatz wird der sein, mittels **Regression** die Werte jeweiliger Sensoren anhand aller anderen Sensoren zu **predicten**. Da der echte Wert jedes Sensors gegeben ist, lässt sich leicht die Differenz (der *Error*) berechnen. Dieser kann nun mit einem Schwellwert (evtl. dynamisch) verglichen werden und bei **zu großem Fehler** Alarm schlagen.

### ~Recurrent- und Graph-NN~

~Neuere Arten von künstlichen, neuronalen Netzen könnten die Effizienz der Leck-Detektion noch weiter verbessern. Während sogenannte [Recurrent Neural Networks](https://de.wikipedia.org/wiki/Rekurrentes_neuronales_Netz) als Zusatz eine Art Gedächtnis besitzen, welches hilfreich beim Bearbeiten von **sequenziellen Daten** sind, gibt es auch sogenannte [Graph Neural Networks](https://en.wikipedia.org/wiki/Graph_neural_network), welche **speziell für Graphen**, wie einem WDN, einsetzbar sind.~

### ~Outlier Detection und Time Series Forecasting~

~Zwei Ansätze, die vlt. möglich sind.~

## Die Roadmap

- [X] Aufgabenstellung formulieren
- [X] Git initialisieren
- [X] Weitere Literatur lesen
- [X] Analysieren der Daten
  - [X] Zyklen erkennen (Tageszyklus, Jahreszeiten, ...)
    -> Nur Tageszyklen, Anpass-Zeit
- [X] Datenstrukturen aufsetzen
  - [X] Netzwerk
  - [X] Datensatz
  - [X] Datengenerator
    - [X] Einzelne Simulationen
    - [X] Gesamtes X, y
    - [X] Verschiedene Nodes mit Leck
  - [X] Modell (Klassifikation)
  - [X] Modell (Regression)
  - [X] Modell (Weiteres)
  - [X] Feature Extractor
    - [X] Past Days
    - [X] Time window mean
  - [X] Metrik
    - [X] Precision, Recall, Specificion, Accuracy
    - [X] Detection Time
      - [X] Mean
      - [X] Standartabweichung
      - [X] Median
  - [X] Helper
- [X] Testen verschiedener Leck-Arten (Bruch, Schleichend, ...)
- [X] **Trainingsdaten** richtig generieren
  - [X] *train-test-split*
  - [X] Cross Validation
  - [X] Grid Search
- [X] **Experiment: Baseline**
  - [X] Modelle testen
    - [X] SVM
    - [X] KNN
    - [X] NN
  - [X] Versch. Feature Extractor
  - [X] Hyperparameter adjustieren
  - [X] Metriken analysieren
- [X] **Experiment: Regression**
  - [X] Modell aufstellen
  - [X] Hyperparameter adjustieren
  - [X] Threshold adjustieren
    - [X] Magic number
    - [X] Time of day
  - [X] Metriken analysieren
- [X] ~**Experiment: Recurrent- und Graph-NN**~
  - [X] ~Was ist das überhaupt?~
  - [X] ~Keine Ahnung?~
- [X] ~**Experiment: Weitere Modelle**~
  - [X] ~Time Series Forecasting~
  - [X] ~Unsupervised Outlier Detection~
- [X] **Datensätze** testen
  - [X] Net1 Spielzeug
  - [X] LeakDB
    - [X] Szenario laden
    - [X] Datensatz laden
    - [X] Datensatz selektieren
    - [X] Modelle testen
    - [X] Metrik umschreiben (h -> ts)
  - [X] ~BattleDIM~
- [X] ~**XAI**: Analysieren der Experimente~
- [ ] Aufschreiben
  - [X] Formelles
  - [ ] Bilder generieren
  - [ ] Kapitel schreiben
  - [ ] Prüfen
