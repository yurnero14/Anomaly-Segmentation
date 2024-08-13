## Prepare directory of Training [Cityscapes](https://www.cityscapes-dataset.com) and Validation [All validation dataset](https://drive.google.com/drive/folders/1asRIA8azwl3ZHznza3VKki308VeRZvr3?usp=drive_link) in a ./datasets dir with following structure

```plaintext
datasets
├── Train_Dataset
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── leftImg8bit
│       ├── test
│       ├── train
│       └── val
└── Validation_Dataset
    ├── FS_LostFound_full
    │   ├── images
    │   └── labels_masks
    ├── fs_static
    │   ├── images
    │   └── labels_masks
    ├── RoadAnomaly
    │   ├── images
    │   └── labels_masks
    ├── RoadAnomaly21
    │   ├── images
    │   └── labels_masks
    └── RoadObsticle21
        ├── images
        └── labels_masks
```

## RoadAnomaly:

```plaintext
@misc{lis2019detectingunexpectedimageresynthesis,
      title={Detecting the Unexpected via Image Resynthesis}, 
      author={Krzysztof Lis and Krishna Nakka and Pascal Fua and Mathieu Salzmann},
      year={2019},
      eprint={1904.07595},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1904.07595}, 
}
```

## fs_static & FS_LostFound_full

```plaintext
@article{DBLP:journals/corr/abs-1904-03215,
  author       = {Hermann Blum and
                  Paul{-}Edouard Sarlin and
                  Juan I. Nieto and
                  Roland Siegwart and
                  Cesar Cadena},
  title        = {The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation},
  journal      = {CoRR},
  volume       = {abs/1904.03215},
  year         = {2019},
  url          = {http://arxiv.org/abs/1904.03215},
  eprinttype    = {arXiv},
  eprint       = {1904.03215},
  timestamp    = {Tue, 13 Apr 2021 10:56:09 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1904-03215.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## RoadAnomaly21 & RoadObstacle21

```plaintext
@misc{segmentmeifyoucan2021,
	  title={SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation}, 
	  author={Robin Chan and Krzysztof Lis and Svenja Uhlemeyer and Hermann Blum and Sina Honari and Roland Siegwart and Pascal Fua and Mathieu Salzmann and Matthias Rottmann},
	  year={2021},
	  eprint={2104.14812},
	  archivePrefix={arXiv},
	  primaryClass={cs.CV}
}
```
