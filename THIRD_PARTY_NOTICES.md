# Third-party notices

HeronLoom's own source is licensed under the AGPL-3.0 (see [LICENSE](LICENSE)).
This file covers code adapted from other projects, and the licences of
dependencies that carry obligations beyond attribution.

## TopiCLEAR

`adr_refiner.py`'s ADR refinement loop is adapted from TopiCLEAR
(https://github.com/aoi8716/TopiCLEAR), used and modified under the MIT
License reproduced below.

Fujita, A., Yamamoto, T., Nakayama, Y., Kobayashi, R. (2026). "TopiCLEAR:
Adaptive embedding clustering for interpretable topic discovery from
short texts." Expert Systems with Applications, 333, 133996.
https://doi.org/10.1016/j.eswa.2026.133996

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## BERTopic

`naming.py`'s c-TF-IDF weighting is adapted from BERTopic's
`ClassTfidfTransformer` (https://github.com/MaartenGr/BERTopic), used and
modified under the MIT License reproduced below.

Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a class-based
TF-IDF procedure." arXiv:2203.05794.

```
MIT License

Copyright (c) 2024, Maarten P. Grootendorst

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ForceAtlas2 (`fa2`) — GPL-3.0-only

`src/fa2_layout.py` imports `ForceAtlas2` from the [`fa2`](https://pypi.org/project/fa2/)
package (Bhargav Chippada), which provides the native three-dimensional layout
pass (`dim=3`). That package is licensed **GPL-3.0-only**.

HeronLoom itself is licensed under the **AGPL-3.0**. GPL-3.0 §13 and AGPL-3.0
§13 grant each other reciprocal permission to combine a GPL-3.0 work with an
AGPL-3.0 work into a single combined work, so `fa2` can be used within
HeronLoom without a licensing conflict — the combination as a whole remains
governed by HeronLoom's AGPL-3.0, including its network-interaction clause.

No part of `fa2` is copied into or redistributed with HeronLoom: it is a
declared dependency that `pip` fetches from PyPI onto the user's own machine,
and the project is distributed as source. Anyone redistributing a combined
binary or container image should still confirm their own distribution
complies with both licences' terms, since the obligations attach to the
combined work.

`fa2` is published as a source distribution only. It compiles a Cython
extension when a C compiler is present; without one it falls back to a
NumPy-vectorized or pure-Python backend, so the layout still runs, more
slowly.

## certifi — MPL-2.0

`certifi` is pulled in transitively by `requests`. It is licensed under the
Mozilla Public License 2.0, a file-level copyleft licence: its obligations
attach only to modifications of the MPL-licensed files themselves, not to
other files merely combined or linked with it. HeronLoom does not modify or
redistribute certifi's source, so no additional obligation arises from using
it alongside the AGPL-3.0-licensed HeronLoom source.

## Other dependencies

Every other declared dependency (see [`requirements.txt`](requirements.txt))
uses a permissive licence — MIT, BSD, or Apache-2.0 — which places no
copyleft obligation on HeronLoom and is freely combinable with the AGPL-3.0.
None of them is modified or redistributed as part of this repository.
