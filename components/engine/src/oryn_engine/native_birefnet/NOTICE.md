BiRefNet numerical implementation: Copyright (c) 2024 ZhengPeng, MIT.
Swin Transformer backbone: Copyright (c) 2021 Microsoft, MIT; written by
Ze Liu, Yutong Lin and Yixuan Wei. The accompanying MIT license terms apply.

Derived from ZhengPeng7/BiRefNet at e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
and verified against ZhengPeng7/BiRefNet_lite at aa62cd87eafb9cc43056d08ef3615a14628b831d.
The two original numerical implementations differ only in the selected Swin
backbone (apart from comments). Selected numerical classes/functions are retained;
engine-owned construction fixes inference settings and removes pretrained loaders,
repository dispatch, training configuration and unused alternative graphs.
All instantiated supervision/gradient heads and complete backbone weights remain
mandatory. Exact extraction locations and modifications are in provenance.json.
The upstream repository LICENSE is pinned at c4d649666cc171486329413dc2b61f0137158b6a.
