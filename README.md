# hyperparameter optimization

Im Rahmen meiner Bachelorarbeit habe ich ein Kommandozeilentool implementiert mit dem man unter der Verwendung der Library LibSVM Datensätze für Klassifikation und Regression auswerten kann und deren Hyperparameter optimieren. Implementierte Algorithmen sind die Rastersuche, die zufällige Suche und der SMAC-Algorithmus.

### Was ist nötig zum Ausführen?:
 - die Library [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
 - die [SMAC Implementierung](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/) Version 2.10.03
 - Pfade anpassen in hyperparam_opti.py

Zum Aufrufen muss Folgendes in das Terminal eingegeben werden:
```sh
python hyperparam_opti.py [grid, random, smac] [cl, re] [training-data-file] [test-data-file] [#rounds] debug
```

Bei Fragen könnt ihr mir auch eine Email schreiben an [t.sonnekalb@campus.tu-berlin.de](mailto:t.sonnekalb@campus.tu-berlin.de).
