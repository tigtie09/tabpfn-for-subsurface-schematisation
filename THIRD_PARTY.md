# Third-party materials

The root MIT licence applies to the code and documentation written for this repository.
It does not replace the terms for external datasets, model weights, dependencies or model outputs.

## SchemaGAN code and data

- Original code: <https://github.com/fabcamo/schemaGAN>
- Data and released checkpoint: <https://doi.org/10.5281/zenodo.13143431>
- Paper: <https://doi.org/10.1016/j.compgeo.2025.107177>

The original code uses the MIT licence reproduced below.
The sampler and interpolation helpers retain the original scientific behaviour and attribution.
The Zenodo record identifies the dataset/model deposit with the MIT licence (`mit-license` in its API metadata, checked on 2026-11-03).
The two files in `examples/test/` are unchanged original dataset files; their hashes match the manifest.
The full dataset and SchemaGAN weights are downloaded from the original deposit rather than bundled here.

### SchemaGAN MIT licence

```text
MIT License

Copyright (c) 2025 F.A. Campos Montero

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

## TabPFN-3

- Model publisher: <https://huggingface.co/Prior-Labs/tabpfn_3>
- Upstream terms: <https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE>

TabPFN-3 weights use the publisher's TabPFN-3 Non-Commercial License v1.0, not this repository's MIT licence.
Those terms also address fine-tuned derivatives and model outputs, including predictions and scores; commercial or production use requires a separate licence from Prior Labs.
The TabPFN prediction in the README illustration (`docs/images/tabpfn-example.png`) is subject to those output terms.
No original or fine-tuned TabPFN weights are included in the tracked repository.
Download weights directly from the publisher and consult its full current licence for permitted uses.

## Dependencies

The pinned Python packages retain their own licences.
They are installed by uv rather than vendored into this repository.
