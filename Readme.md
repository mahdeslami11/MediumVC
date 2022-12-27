# MediumVC
MediumVC is an utterance-level method towards any-to-any VC. Before that, we propose [SingleVC](https://github.com/BrightGu/SingleVC) to perform A2O tasks(X<sub>i</sub> → Ŷ<sub>i</sub>) , X<sub>i</sub> means utterance i spoken by X). The Ŷ<sub>i</sub> are considered as SSIF. To build SingleVC, we employ a novel data augment strategy: pitch-shifted and duration-remained(PSDR) to produce paired asymmetrical training data. Then, based on pre-trained SingleVC, MediumVC performs an asymmetrical reconstruction task(Ŷ<sub>i</sub> → X̂<sub>i</sub>). Due to the asymmetrical reconstruction mode, MediumVC achieves more efficient feature decoupling and fusion. Experiments demonstrate MediumVC performs strong robustness for unseen speakers across multiple public datasets.
Here is the official implementation of the paper, [MediumVC](http://arxiv.org/abs/2110.02500).


The following are the overall model architecture.

![Model architecture](Demo/image/mediumvc.png)

For the audio samples, please refer to our [demo page](https://brightgu.github.io/MediumVC/). The more converted speeches can be found in "Demo/ConvertedSpeeches/".

### Envs
You can install the dependencies with
```bash
pip install -r requirements.txt
```

### Speaker Encoder
[Dvector](https://github.com/yistLin/dvector)  is a robust  speaker verification (SV) system pre-trained on [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)  using GE2E loss, and it  produces 256-dim speaker embedding. In our evaluation on multiple datasets(VCTK with 30000 pairs, Librispeech with 30000 pairs and VCC2020 with 10000 pairs), the equal error rates(EERs)and thresholds(THRs) are recorded in Table. Then Dvector with THRs is also employed to calculate SV accuracy(ACC) of pairs produced by MediumVC and other contrast methods for objective evaluation. The more details can access [paper](http://arxiv.org/abs/2110.02500).

| Dataset | VCTK | LibriSpeech | VCC2020 |
| :------:| :------: | :------: |:------: |
| EER(%)/THR | 7.71/0.462 | 7.95/0.337 |1.06/0.432 |

### Vocoder
The [HiFi-GAN](https://github.com/jik876/hifi-gan) vocoder is employed to convert log mel-spectrograms to waveforms. The model is trained on universal datasets with 13.93M parameters. Through our evaluation, it can synthesize 22.05 kHz high-fidelity speeches over 4.0 MOS, even in cross-language or noisy environments.

### Infer
You can download the [pretrained model](https://drive.google.com/file/d/1mMSLYdHZZ9PtJo6kceMO2483TxKXgLa_/view?usp=sharing), and then edit "Any2Any/infer/infer_config.yaml".Test Samples could be organized  as "wav22050/$figure$/*.wav". 
```bash
python Any2Any/infer/infer.py
```
### Train from scratch

####  Preprocessing
The corpus should be organized as "VCTK22050/$figure$/*.wav", and then edit the config file "Any2Any/pre_feature/preprocess_config.yaml".The output "spk_emb_mel_label.pkl" will be used for training.
```bash
python Any2Any/pre_feature/figure_spkemb_mel.py
```
#### Training
Please edit the paths of pretrained  hifi-model,wav2mel,dvector,SingleVC in config file "Any2Any/config.yaml" at first.
```bash
python Any2Any/solver.py
```

Seyed mahdi godrazi

1- 

pre-training model trained on VCTK. 
demonstrate that, for seen or unseen speakers, MediumVC
performs better both in naturalness and similarity. The advance in naturalness indicates, compared with Wav2Vec
2.0 based embeddings in FragmentVC and deep content-related
features in AutoVC and AdaIN-VC, SSIF maintains more robust speaker-independent features. Meanwhile, removing the 
influence of multi-speakers and building a specific-speaker
periodic pattern further promote the advance in similarity.
Additionally, AdaIn-VC with the least parameters achieves the best performances on ACC.
We consider the main problem is the employing of discrete speaker embeddings produced by extra pre-trained SV systems in the other four
methods(except FragmentVC). Compared to discrete speaker embeddings, that from the Autoencoder-based model(AdaInVC)
seems to be smoother and more adaptable to unseen speakers.

2- 

In this paper, we propose SingleVC to perform A2O VC,and based on it, we propose MediumVC to perform A2A
VC. The key to well-performance is that we build asymmetric reconstruction tasks for self-supervised learning. For Sin-
gleVC, we employ PSDR to edit source pitches, promoting the SingleVC to learn robust content information by rebuild-
ing source speech. For MediumVC, employing SSIF processed by SingleVC promotes MediumVC to rely more on
speaker embeddings to enhance target similarity. It is asymmetric tasks that drive models to learn more robust features
purposefully.


3-
"
ModuleNotFoundError                       Traceback (most recent call last)

<ipython-input-1-cccb1023891d> in <module>
      9 import yaml
     10 from torch.utils.data import DataLoader
---> 11 from Any2Any import util
     12 from Any2Any.meldataset import Test_MelDataset, get_infer_dataset_filelist,mel_denormalize
     13 from Any2Any.model.any2any import MagicModel

ModuleNotFoundError: No module named 'Any2Any'


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
# import sys
# sys.path.append("")
import torch
from torch.backends import cudnn
import numpy as np
import yaml
from torch.utils.data import DataLoader
from Any2Any import util
from Any2Any.meldataset import Test_MelDataset, get_infer_dataset_filelist,mel_denormalize
from Any2Any.model.any2any import MagicModel
	
from hifivoice.inference_e2e import  hifi_infer


class Solver():
	def __init__(self, config):
		super(Solver, self).__init__()
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.make_records()
		self.Generator = MagicModel().to(self.device)
		if self.config['pre_train_singlevc']:
			singlevc_checkpoint = torch.load(self.config['singlevc_model_path'], map_location='cpu')
			self.Generator.any2one.load_state_dict(singlevc_checkpoint['Generator'])
			self.Generator.any2one.eval()
			self.Generator.any2one.remove_weight_norm()
		self.resume_model(self.config['resume_path'])
		self.logging.info('config = %s', self.config)
		print('param Generator size = %fM ' % (util.count_parameters_in_M(self.Generator)))
	"

