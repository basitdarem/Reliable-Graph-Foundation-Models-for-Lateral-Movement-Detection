# Third-party attribution

This project's fair evaluation protocol for the LANL dataset -- the fixed
train/test time cutoff, the "keep every remote LogOn, no NTLM-only filter"
rule, and the red-team label join key `(t, dst_user, src_comp, dst_comp)`
used in `lmsafe/data/lanl.py` and `lmsafe/data/schema.py` -- is **adapted
from** the evaluation protocol published alongside:

> Corentin Larroche, "Inductive Lateral Movement Detection in Enterprise
> Computer Networks," ESANN 2024 Proceedings, European Symposium on
> Artificial Neural Networks, Computational Intelligence and Machine
> Learning, i6doc.com, 2024.
> <https://www.esann.org/sites/default/files/proceedings/2024/ES2024-19.pdf>

and its accompanying open-source evaluation code, **LMDEval**:

> <https://github.com/cl-anssi/LMDEval/> (BSD-2-Clause license)

No source files from LMDEval are copied or redistributed in this
repository. `lmsafe/data/lanl.py` and `lmsafe/data/schema.py`
independently reimplement the specific label-join rule and filtering
logic described above, so that the numbers this project reports are
directly comparable to LMDEval's published fair-evaluation protocol
rather than to an easier, non-comparable split (see each module's
docstring and `tests/test_lanl_labels.py` for the regression tests that
pin this behaviour). This reimplementation is believed compatible with
LMDEval's BSD-2-Clause terms; this NOTICE exists to credit the original
work regardless.

BSD-2-Clause license text (as published by LMDEval, reproduced here for
attribution):

```
Copyright (c) Agence nationale de la securite des systemes d'information (ANSSI)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Datasets

This project's own code never bundles the full LANL or OpTC datasets (see
`.gitignore`); only a small, real excerpt of each is kept under
`tests/fixtures/` strictly for unit testing (a 8 KB slice of the public
`redteam.txt.gz` labels and a 241-line `ecar-bro/short` sample). The full
datasets must be obtained directly from their official sources:

- **LANL Comprehensive Multi-Source Cyber-Security Events dataset**:
  A. D. Kent, "Comprehensive, Multi-Source Cyber-Security Events," Los
  Alamos National Laboratory, 2015. <https://doi.org/10.17021/1179829>
- **DARPA OpTC (Operationally Transparent Cyber)**:
  C. Anjum et al., "An Analysis of Open-source Datasets for Network
  Security," EASE 2021. OpTC release notes and `ecar-bro`/`ecar`
  documentation at <https://github.com/FiveDirections/OpTC-data>.

## License of this project

Unless noted otherwise above, all original code in this repository is
released under the MIT License -- see `LICENSE`.
