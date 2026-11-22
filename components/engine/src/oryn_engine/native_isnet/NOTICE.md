IS-Net numerical implementation from the DIS project by Xuebin Qin and contributors.
Source: https://github.com/xuebinqin/DIS
Pinned revision: b6764e20381f6f42a70f83fa3324181529ed1403
License: Apache-2.0 (full upstream LICENSE.md retained as LICENSE).

The engine retains the complete ISNetDIS graph and all six side outputs.
Unused training losses and the separate ground-truth encoder are excluded;
the deprecated F.upsample name is replaced by equivalent F.interpolate.
No repository model code, pretrained loader or checkpoint conversion is included.
Original RMBG-1.4 numeric classes and forward graph match this implementation;
model weights retain their separately declared publisher license.
