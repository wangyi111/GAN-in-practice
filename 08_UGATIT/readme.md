# U-GAT-IT

This is a forward of the [official implementation](https://github.com/taki0112/UGATIT) of the paper [**Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation (ICLR 2020)**](https://arxiv.org/abs/1907.10830) which has a great result on unpaired image to image translation (selfie2anime, especially).

<div align="center">
  <img src="./assets/teaser.png">
</div>

## Dataset
* [selfie2anime dataset](https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view?usp=sharing)

## Implementation
* [tensorflow](./tensorflow/)
* [pytorch](./pytorch/)

## Web page
* [Selfie2Anime](https://selfie2anime.com) by [Nathan Glover](https://github.com/t04glovern)
* [Selfie2Waifu](https://waifu.lofiu.com) by [creke](https://github.com/creke)

## Telegram Bot
* [Selfie2AnimeBot](https://t.me/selfie2animebot) by [Alex Spirin](https://github.com/sxela)

## Architecture
<div align="center">
  <img src = './assets/generator_fix.png' width = '785px' height = '500px'>
</div>

---

<div align="center">
  <img src = './assets/discriminator_fix.png' width = '785px' height = '450px'>
</div>

## Results
### Ablation study
<div align="center">
  <img src = './assets/ablation.png' width = '438px' height = '346px'>
</div>

### User study
<div align="center">
  <img src = './assets/user_study.png' width = '738px' height = '187px'>
</div>

### Kernel Inception Distance (KID)
<div align="center">
  <img src = './assets/kid_fix2.png' width = '750px' height = '400px'>
</div>

## Author
[Junho Kim](http://bit.ly/jhkim_ai), Minjae Kim, Hyeonwoo Kang, Kwanghee Lee