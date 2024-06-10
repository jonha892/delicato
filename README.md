# delicato

## Install dependencies

```bash
pip3 install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## Data stuff

Für den Fall das wir mehrere Datensätze zum trainieren benutzen überführen wir den bzw. die Datensätze in ein einheitliches Format.

### Datasets

#### 'nfi'
https://www.kaggle.com/datasets/divyanshrai/handwritten-signatures/data

Leider ist deren ids nicht ganz eindeutig. Z.B. ist die Signatur '003' im dataset 1 & 3 unterschiedlich.
Vielleicht ist das eine Ausnahme.
Ohne viel manuell zu beheben gehen wir dennoch davon aus, dass die IDs pro Datensatz einzigartig sind.

Dadurch haben wir leider keine kombinationen über die Datasets hinaus.

### Struktur

Die Ordnerstruktur ist wie folgt:

```
blob
|
|--{origin}_{id}
|         |
|         |--genuine
|         |     |
|         |     |--1.png
|         |     |--2.png
|         |--forged
|               |
|               |--1.png
|               |--2.png
|
|--{origin}_{id}
          |
          |--genuine
          |
          |--forged
```