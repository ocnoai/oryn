# Dependency licenses and notices

Oryn's [MIT License](LICENSE) applies to original Oryn code and documentation.
Third-party code, dependency archives and model artifacts retain their original
licenses. This index supplements the license and notice files shipped with those
components; it does not replace or relicense them.

## Bundled dependency artifacts

| Component | License and provenance |
| --- | --- |
| Compiled `@ocno/oryn-registry` library and baseline catalog | Original Oryn code/catalog work, MIT. The versioned package is bundled with Browser; its source tree and hosting are maintained separately. Preserve the package's MIT license. |
| Oryn Protocol, SDK and MFLUX provider | Original Oryn code, MIT. Protocol/SDK artifacts are rebuilt from included source by bootstrap; provider source is included under Engine. |
| Hugging Face Diffusers | Apache-2.0. Retain the `*.dist-info/licenses/LICENSE` inside each redistributed wheel. Reviewed source revision and wheel identity are recorded in [diffusers-source.json](components/browser/packaging/diffusers-source.json). |
| MFLUX | MIT, copyright Filip Strand. Preserve the license text in its wheel metadata and the embedded Diffusers reference license (`mflux/models/qwen21/reference/LICENSE.diffusers`). Reviewed source identity is recorded in [mflux-source.json](components/browser/packaging/mflux-source.json). |
| `oryn_depth_native` / Depth Anything 3 numerical sources | Apache-2.0. The wheel includes its upstream license. [depth-native-source.json](components/engine/vendor/depth-native-source.json) records the ByteDance-Seed/Depth-Anything-3 revision, file hashes and constrained integration changes. |

The selected artifact filenames and hashes are recorded in the public export's
verification metadata and consumer lockfiles. Preserve those identities together
when updating an artifact. Old locally cached versions are not part of the source
distribution unless explicitly selected.

## Source files with their own notices

Vendored implementations and assets under [Engine source](components/engine/src/oryn_engine/)
retain their adjacent `LICENSE`, `NOTICE` and attribution files. Native llama.cpp
build inputs retain [LLAMA_CPP_LICENSE](components/engine/build-inputs/LLAMA_CPP_LICENSE).
Do not replace those files with Oryn's MIT license.

Node and Python dependencies installed by package managers retain the license
terms shipped by their upstream projects. Application packaging must preserve
the notices for the dependencies it actually redistributes.

## Model artifacts

Model weights are downloaded separately and are not included in this source
export. Each model checkpoint has its own license and, where applicable, gated
access or acceptance requirements. Oryn's code license does not grant rights to
those model artifacts.
