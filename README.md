# Sample snapshot (generated, do not edit)

This branch holds the dataset samples shown on the site. It is written by
`scripts/sample.py` in the deploy workflow on `main`; do not open pull requests
against it.

- `manifest.json`: per dataset, the fingerprint of its config, the Hugging Face
  revision it was sampled from and when.
- `generated/<id>.json`: sampled rows.
- `samples/<id>/`: images and audio those rows reference.

To change a sample, edit `datasets/<id>.yaml` on `main` (or run the deploy
workflow with "force refetch"). Samples remain under each dataset's own license.
