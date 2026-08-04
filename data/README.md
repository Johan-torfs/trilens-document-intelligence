# TriLens external dataset

External dataset images are stored locally under `data/external/` and are not committed to Git.

## Sources

### CORD

Receipt images fetched automatically from:

```text
naver-clova-ix/cord-v2
```

Output:

```text
data/external/cord/
```

### FUNSD

Scanned forms must be downloaded manually from:

<https://guillaumejaume.github.io/FUNSD/download>

Save the downloaded ZIP locally. It does not need to be copied into the repository.

Output:

```text
data/external/funsd/
```

### DocLayNet

Document pages fetched automatically from:

```text
docling-project/DocLayNet-v1.2
```

Output:

```text
data/external/doclaynet/
```

## Rebuild

From the project root:

```bash
python -m scripts.dataset.fetch_external_datasets \
  --funsd-archive ~/Downloads/dataset.zip
```

Expected result:

```text
CORD:       10 images
FUNSD:      10 images
DocLayNet:  30 images
Total:      50 images
```

The script removes and recreates the individual dataset directories, producing the same fixed sample on every run.

## Git policy

Commit:

```text
scripts/dataset/
data/README.md
```

Do not commit:

```text
data/external/
```

Third-party datasets retain their original licences and usage conditions.
