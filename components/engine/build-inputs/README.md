# Native capability build inputs

`llama-arch-b1b770d9.cpp` is unmodified upstream source from
https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/src/llama-arch.cpp.
Its MIT license is preserved in `LLAMA_CPP_LICENSE`.

`llama-chat-policy.json` is Oryn-owned reviewed data. Its source revision and SHA-256 bind
that file; the build generator additionally binds the exact packaged native executable.
It does not run or load upstream source during model inspection or execution. The generator
emits available native facts and admitted/withheld task tuples into the common signed registry.
Runtime code consumes that registry and has no second admission table. New family admission
requires header, tokenizer, sealed framing, resource, inference and cancellation evidence.

IQ4_XS container validation uses GGML type 23 and a 256-value / 136-byte block,
verified at the same pinned source revision in
[`ggml.h`](https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/ggml/include/ggml.h)
and [`ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/b1b770d90d94cad9e9ea5b993bdbc383a522fb28/ggml/src/ggml-common.h).
The bounded parser validates complete blocks and tensor extents before native loading.
The signed policy can admit multiple sealed renderers for an architecture; the installed
header must select exactly one. Header template text is never executed.
