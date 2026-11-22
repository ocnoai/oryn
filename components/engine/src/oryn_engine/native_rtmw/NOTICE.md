# RTMW native numerical graph

This Apache-2.0 numerical implementation is adapted from
[akore/rtmw-l-256x192](https://huggingface.co/akore/rtmw-l-256x192/tree/fd7df05e28ce758a98f90141e6f85097f662e48c),
a publisher port of [OpenMMLab MMPose](https://github.com/open-mmlab/mmpose).
The frozen model card declares Apache-2.0. Attribution is retained to the
publisher and OpenMMLab contributors. The standard license is included here.

Only the self-contained PyTorch numerical classes and graph constructor are
retained. Hugging Face configuration, model loading, training initialization,
output wrapper and coordinate-mode dispatch are excluded. The owned binding
validates the configuration and loads all original Safetensors tensors strictly.
The head runs once; its original decoder supplies coordinates and confidence.
No repository code is downloaded or executed by the production runtime.
