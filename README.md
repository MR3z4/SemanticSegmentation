[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Python][python-shield]][project-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/MR3z4/SemanticSegmentation">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Semantic Segmentation</h3>

  <p align="center">
    This is a repository to train semantic segmetation models. It will be improved over time.
    <br />
    <a href="https://github.com/MR3z4/SemanticSegmentation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/MR3z4/SemanticSegmentation">View Demo</a>
    ·
    <a href="https://github.com/MR3z4/SemanticSegmentation/issues">Report Bug</a>
    ·
    <a href="https://github.com/MR3z4/SemanticSegmentation/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-using">Built Using</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#requierments">Requierments</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project started as my master thesis. I will keep improving it as long as i can.


### Built Using

* Python 3.7
* PyTorch 1.2.0
* torchvision 0.4.0



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Requierments

First install the requierments as followed.
  ```sh
  pip install -r requirments.txt
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MR3z4/SemanticSegmentation.git
   ```
2. Run the training code
   ```sh
   python main.py
   ```



<!-- USAGE EXAMPLES -->
## Usage

It will be completed over time.


<!-- ROADMAP -->
## Roadmap
- [x] Multi GPU support
- [x] Add RMI loss
- [x] Add Mixup option for training
- [x] Add Mixup Without Hesitation for training with mixup
- [x] Add AdaBelief optimizer option for training
- [x] Add CE2P Network(with normal BatchNorm) for training.
- [x] Add InPlace Active BatchNorm for CE2P.
- [x] Add SCHP completely.
- [ ] Add MixMatch option for training
- [ ] Add FixMatch option for training
- [ ] Distibuted support

See the [open issues](https://github.com/MR3z4/SemanticSegmentation/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

M.Mohammadzade - [@realMReza](https://twitter.com/realMReza) - mohammadzade.m.r@gmail.com

Project Link: [https://github.com/MR3z4/SemanticSegmentation](https://github.com/MR3z4/SemanticSegmentation)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* Peike Li, Yunqiu Xu, Yunchao Wei, Yi Yang. "Self-Correction for Human Parsing" IEEE Transactions on Pattern Analysis and Machine Intelligence 2020, [arXiv:1910.09777](https://arxiv.org/abs/1910.09777), [Project Code](https://github.com/PeikeLi/Self-Correction-Human-Parsing)
* Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz. "mixup: Beyond Empirical Risk Minimization." arXiv preprint arXiv:1710.09412. [arXiv:1710.09412]( https://arxiv.org/abs/1710.09412 )
* Hao Yu, Huanyu Wang, Jianxin Wu. "Mixup Without Hesitation" arXiv preprint arXiv:2101.04342. [arXiv:2101.04342]( https://arxiv.org/abs/2101.04342 )
* Shuai Zhao, Yang Wang, Zheng Yang, Deng Cai. "Region Mutual Information Loss for Semantic Segmentation", NeurIPS 2019, [arXiv:1910.12037](https://arxiv.org/abs/1910.12037), [Project Code](https://github.com/ZJULearning/RMI)
* Juntang Zhuang, Tommy Tang, Yifan Ding , Sekhar Tatikonda, Nicha Dvornek, Xenophon Papademetris, James S. Duncan. "AdaBelief Optimizer: fast as Adam, generalizes as good
as SGD, and sufficiently stable to train GANs." [arXiv:2010.07468](https://arxiv.org/abs/2010.07468), [Project Code](https://github.com/juntang-zhuang/Adabelief-Optimizer)





<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/MR3z4/SemanticSegmentation.svg
[contributors-url]: https://github.com/MR3z4/SemanticSegmentation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MR3z4/SemanticSegmentation.svg
[forks-url]: https://github.com/MR3z4/SemanticSegmentation/network/members
[stars-shield]: https://img.shields.io/github/stars/MR3z4/SemanticSegmentation.svg
[stars-url]: https://github.com/MR3z4/SemanticSegmentation/stargazers
[issues-shield]: https://img.shields.io/github/issues/MR3z4/SemanticSegmentation.svg
[issues-url]: https://github.com/MR3z4/SemanticSegmentation/issues
[license-shield]: https://img.shields.io/github/license/MR3z4/SemanticSegmentation.svg
[license-url]: https://github.com/MR3z4/SemanticSegmentation/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mohammadreza-mohammadzade-545653104
[python-shield]: https://img.shields.io/badge/python-3.7-green.svg
[project-url]: https://github.com/MR3z4/SemanticSegmentation
