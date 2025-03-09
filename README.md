# 🎙️ VoxKor-VITS

VoxKor-VITS는 VITS(Conditional Variational Autoencoder with Adversarial Learning) 기반의 한국어 음성 합성 모델입니다. 해당 레포에서는 학습 코드를 공유하고 있습니다.<br>

## ✨ Features
- VITS 아키텍처 기반의 고품질 한국어 TTS
- 다중 화자 음성 합성 지원
- 실시간 추론이 가능한 빠른 속도
- 간편한 학습 및 커스텀 화자 추가

## ⚙️ Installation
[**Please installation build-tools**](https://visualstudio.microsoft.com/visual-cpp-build-tools/)<br>
[**Please installation pytorch**](https://pytorch.org/)
```bash
git clone https://github.com/byeolki/VoxKor-VITS.git
cd VoxKor-VITS
pip install -r requirements.txt
```

## 🚀 Usage
자세한 사용법은 `tutorial.ipynb`를 참고해주세요.

## 🙏 Credits & References
이 프로젝트는 다음 오픈소스 프로젝트들을 참고 및 활용했습니다:

- [JK-VITS](https://github.com/kdrkdrkdr/JK-VITS) - 한국어 TTS 모델 (MIT License)

## ⚖️ License
이 프로젝트의 코드는 MIT License에 따라 배포됩니다:

MIT License

Copyright (c) 2024 Byeolki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

단, 이 프로젝트에 포함된 외부 라이브러리들은 각각의 라이센스를 따릅니다:
- JK-VITS 관련 코드: MIT License
