# DeepSeek V4.1 text-message adapter

Vesper's `messages.py` text encoder and response parser adapt the message
format, exact tool-instruction template, DSML tag spacing, numeric
reasoning-effort prefix, transitions and tool-result ordering from:

- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`
- Revision: `1bae0e1088993b9b82de08f59de5be5a34e42170`
- Source: `encoding/encoding.py`
- Raw source SHA-256: `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`
- Copyright (c) 2023 DeepSeek
- License: [DeepSeek-LICENSE](DeepSeek-LICENSE) (MIT, included in this package)

The optional CPU oracle imports the unchanged hash-checked upstream file from
ignored local artifacts and compares it with Vesper's separate implementation.
Upstream source files and model weights are not redistributed by this package.
Authored parser/scorer examples are test data, not generated model output or
a model-quality qualification. This addendum preserves the historical
attribution documents already bound to earlier acceptance evidence.
