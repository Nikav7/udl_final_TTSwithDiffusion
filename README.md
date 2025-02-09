# UDL Final Project a.y. 2024/2025
# TTS with Diffussion
<p align="center">
    <img src="resources/reverse-diffusion.gif" alt="drawing" width="500"/>
</p>


# Grad-TTS

Official Repository: https://github.com/huawei-noah/Speech-Backbones/tree/main

**Authors**: Veronica Valente and Linas Raicinskis 

## Installation

Firstly, install all Python package requirements:

```bash
pip install -r requirements.txt
```

Secondly, build `monotonic_align` code (Cython):

```bash
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

**Note**: code is tested on Python==3.7.4

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict and a custom Italian dictionary, CMU repository: [link](https://github.com/cmusphinx/cmudict) Italian dict: [link](https://github.com/cmusphinx/cmudict](https://github.com/Kyubyong/pron_dictionaries )
