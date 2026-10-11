# mamba 2 windows (cuda12.8) whl, including causal-conv1d, mamba-ssm installation environment instructions

Compilation environment:

```
VS2019 (Theoretical support for VS2022)
win10 x64
anaconda3+python3.10
torch==2.8.0+cu128
cuda12.8.0+cudnn9.8.0
triton-windows==3.3.0a0.post17
causal_conv1d==1.4.0
mamba==2.2.2
RTX5060 Laptop Graphics Card
```

Compiled complete environment module version name:

```
annotated-doc==0.0.5
anyio==4.15.1
causal_conv1d @ file:///C:/Users/75035/Downloads/causal-conv1d-1.4.0
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.5.0
colorama==0.4.6
einops==0.8.2
exceptiongroup==1.3.1
filelock==3.32.3
fsspec==2026.7.0
h11==0.16.0
hf-xet==1.6.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
idna==3.19
Jinja2==3.1.6
mamba_ssm @ file:///C:/Users/75035/Downloads/mamba-2.2.2
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
networkx==3.4.2
ninja==1.13.2
numpy==2.2.6
packaging==26.3
pillow==12.3.0
Pygments==2.21.0
PyYAML==6.0.3
regex==2026.9.3
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
shellingham==1.5.4
sympy==1.14.0
tokenizers==0.15.2
torch==2.8.0+cu128
torchvision==0.23.0+cu128
tqdm==4.70.0
transformers==4.37.1
triton-windows==3.3.0a0.post17
typer==0.27.2
typing_extensions==4.16.0
urllib3==2.7.0
```

Test passed code:

```
import torch
from mamba_ssm import Mamba
from mamba_ssm import Mamba2
 
batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print('Mamba:', x.shape)
 
batch, length, dim = 2, 64, 256
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
print('Mamba2:', x.shape)
```

Post-compiled file provided:

```
causal-conv1d-1.4.0-src-cuda12.8.zip
causal_conv1d-1.4.0-cp310-cp310-win_amd64.whl
mamba-2.2.2-src-cuda12.8.zip
mamba_ssm-2.2.2-cp310-cp310-win_amd64.whl
requirements.txt
triton_windows-3.3.0a0.post17-cp310-cp310-win_amd64.whl
Compilation Notes.txt
```

All compiled whl file download addresses:


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
