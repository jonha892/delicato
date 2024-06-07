# delicato

## Install dependencies

```bash
pip3 install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## Data stuff

Für den Fall das wir mehrere Datensätze zum trainieren benutzen überführen wir den bzw. die Datensätze in ein einheitliches Format.

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