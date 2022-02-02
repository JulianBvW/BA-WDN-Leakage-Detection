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

### Fragen

1. Model
    - (Lineare) Regression
    - Zeitserien (z.B: Mittelwert über die letzten t Zeitpunkte)
    - RNN (Deep Learning)
    - Kernel-Methoden
    - Hyperparameter
2. Schwellwert
    - Dynamisch? Fix? Wie optimiert?
    - Statistik: Hypothesentests, Signifikantsintervalle, etc.
3. Explainable AI
    - Feature-importance
    - Abhängigkeit von Sensoren
4. Datensätze
    - Spielzeugdatensatz (z.B. Net1)
    - [LeakDB](https://github.com/KIOS-Research/LeakDB)
5. Evaluations-Metriken
	  - Train-Test Split (z.B. Cross validation, ...) 
	  - Annomaliedetektion => Zeitpunkte: Vgl. mit Annomalie-Zeitpunkten aus GroundTruth -- TP, FP, TN, FN (Confusion Matrix), Precision, Recall, ...
	  - Detection Time (DT)
	  - Unteschiedliche Arten von Annomalien (Größe, Zeitdauer, ....)
