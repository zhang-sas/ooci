# Source provenance and third-party assets

This directory is extracted from the local `DAGF-chuangxin2` project. It includes
the project's existing completed DAGF core plus its frozen-v1 AEOR and relation-only
RG extension. The completed core's original module documentation describes it as
an engineering interpretation of DAGF, not recovered author weights or an official
author implementation. Those source comments are retained.

Copied file locations and SHA-256 hashes are recorded in `source_provenance.json`.
The DAGF core was placed at `dagf_complete/`, a layout already supported by the
unchanged frozen baseline adapter. No model, data-loader or training-backend source
bytes were changed during this extraction.

The inspected source checkout did not contain a repository-wide license applicable
to this extracted set. This packaging step does not assign MIT, Apache, GPL or any
other new license to upstream code. A project-wide license can be added by the
repository owner after establishing the applicable permissions and attribution.

PyTorch, Transformers and the remaining dependencies retain their own licenses.
RoBERTa weights/tokenizer files, benchmark corpora, opinion lexicon data, private
annotations, checkpoints and run artifacts are not included in this release.
Retain the original asset notices and dataset citations when obtaining those files.

The opinion lexicon filenames expected by the existing code are `positive-words.txt`
and `negative-words.txt`. The original files contain their own Hu/Liu publication
and source notices; obtain them separately and preserve those notices.

No publication title, author list, DOI, GitHub upstream URL or benchmark result has
been invented for this release. Add the final paper citation when available.
