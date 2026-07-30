# Third-Party Notices

TriLens Document Intelligence uses third-party open-source libraries, pretrained models and public or synthetic datasets.

The MIT License in this repository applies only to original TriLens source code and documentation for which the project contributors hold the necessary rights.

It does not relicense third-party components.

## Pretrained models

The project can use pretrained models from the following model families:

- CLIP
- BLIP
- OpenFlamingo
- MPT components used by the configured OpenFlamingo checkpoint

Model implementations, configuration files, tokenizer files and downloaded weights remain subject to the licenses and usage conditions published by their respective authors and distributors.

Modelweights are downloaded at runtime or stored in a local model cache and are not distributed as part of this repository.

## Datasets

The project uses synthetic, public or derived document images for development and demonstration.

Each external dataset retains its original license and terms of use. Dataset files may not be redistributed through this repository unless their license explicitly permits redistribution.

The repository must not contain:

- real identity documents;
- customer documents;
- personal data;
- confidential business documents;
- copyrighted dataset files without redistribution permission.

## Dependencies

Python and JavaScript dependencies retain their own licenses.

Dependency names and versions are recorded in:

- `requirements.txt`
- `frontend/package-lock.json`

Users distributing the application are responsible for reviewing the licenses of the dependency versions they distribute.

## No endorsement

The inclusion or use of a third-party model, dataset or dependency does not imply endorsement by its authors or maintainers.
