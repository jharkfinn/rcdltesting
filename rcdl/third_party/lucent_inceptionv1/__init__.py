"""InceptionV1 (GoogLeNet, Google's inception5h graph) in PyTorch, vendored from
Lucent 0.1.8 (github.com/greentfrapp/lucent, Apache License 2.0), whose model files are themselves from ProGamerGov's
neural-dream (MIT License; see the headers of InceptionV1.py and helper_layers.py). Changes: one relative import, and the
two local response normalizations. Google's graph has LRN depth_radius 5 (window 11), alpha 1e-4, beta 0.5, bias 2;
upstream Lucent uses window 9, bias 1 and passes alpha unscaled although torch's local_response_norm divides alpha by the
window. With upstream's values every activation after the first LRN differs from Lucid/TensorFlow by about 2x its scale
(rcdl/inception_provenance.compare_forward); with these, all layers match TensorFlow to float32 rounding.

Weights: inception5h.pth (ProGamerGov/pytorch-old-tensorflow-models), equal bit for bit to Google's
download.tensorflow.org/models/inception5h.zip (rcdl/inception_provenance.py). Input: RGB, NCHW, values x * 255 - 117
for x in [0, 1] (Lucid's InceptionV1 image_value_range (-117, 138)).
"""

from .InceptionV1 import InceptionV1
