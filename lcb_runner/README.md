# LiveCodeBench scoring helpers

The code-generation dataset types and scoring implementation in this directory
are adapted from [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench).
The upstream MIT license is preserved in [LICENSE](LICENSE).

Only the modules used by `eval.livecodebench_backend` are included. The loader
reads the release-specific JSONL files through Hugging Face; scoring uses the
public and private test cases. Prompt formatting and final fenced-code
extraction live in the evaluation backend. Other model runners and benchmark
scenarios are not part of this release.
