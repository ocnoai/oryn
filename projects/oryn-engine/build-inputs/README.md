# Native capability build inputs

`llama-arch-8642541b.cpp` is unmodified upstream source from
https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/src/llama-arch.cpp.
Its MIT license is preserved in `LLAMA_CPP_LICENSE`.

`llama-chat-policy.json` is Oryn-owned reviewed data. Its source revision and SHA-256 bind
that file; the build generator additionally binds the exact packaged native executable.
It does not run or load upstream source during model inspection or execution. The generator
emits available native facts and admitted/withheld task tuples into the common signed registry.
Runtime code consumes that registry and has no second admission table. New family admission
requires header, tokenizer, sealed framing, resource, inference and cancellation evidence.

IQ4_XS container validation uses GGML type 23 and a 256-value / 136-byte block,
verified at the same pinned source revision in
[`ggml.h`](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/ggml/include/ggml.h)
and [`ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/ggml/src/ggml-common.h).
The bounded parser validates complete blocks and tensor extents before native loading.
The signed policy can admit multiple sealed renderers for an architecture; the installed
header must select exactly one. Header template text is never executed.
