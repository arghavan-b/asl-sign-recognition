# Data

Place raw clips here, one folder per gloss (the folder name is the label):

```
data/raw/HELLO/clip001.mp4
data/raw/HELLO/clip002.mp4
data/raw/THANK-YOU/clip001.mp4
...
```

Then run `python -m src.extract` to populate `data/landmarks/` with cached
`.npy` arrays mirroring this structure.

Optional `data/splits/train.txt` and `data/splits/val.txt` let you pin a
**signer-disjoint** split (recommended). Each line is a path relative to
`data/landmarks/`, e.g. `HELLO/clip001.npy`.

Nothing in `raw/`, `landmarks/`, or `splits/` is committed (see `.gitignore`).
Keep any footage of real people encrypted and local; delete raw video after
extraction unless the subject opted in to longer retention.
