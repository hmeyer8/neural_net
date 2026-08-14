# Computer Vision

The eight-week track described in [../ROADMAP.md](../ROADMAP.md). Tensors through a deployed
inference service.

## Layout

```
cv/
├── week01_foundations/     shape drills, dataloaders, splits, dumb baseline
├── week02_cnns/            manual convolution, CIFAR-10, debugging drills
├── week03_transfer/        pretrained backbones, custom dataset, freeze vs fine-tune
├── week04_evaluation/      metrics, thresholds, failure analysis, reproducibility audit
├── week05_detection/       IoU, NMS, detector fine-tuning
└── capstone/               Equipment Listing Image QA — the shipped project
    ├── src/
    │   ├── config.py
    │   ├── dataset.py
    │   ├── model.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── inference.py
    │   └── api.py
    ├── tests/
    ├── docs/
    ├── examples/
    ├── Dockerfile
    └── README.md
```

Directories appear as I get to them rather than all at once.

## Capstone: Equipment Listing Image QA

Given the images attached to an equipment listing, return structured, actionable output rather
than a bare class label:

```json
{
  "asset_class": "heavy_truck",
  "view": "front_left",
  "usable_image": true,
  "confidence": 0.94,
  "manual_review": false
}
```

The `manual_review` flag is the part I care about most. A model that knows when to abstain is
worth more to a business than a model with two more points of accuracy, because it lets you set
the precision you actually need and route the rest to a human.

Data is open-licensed only, with the license recorded in the dataset card. No employer data
goes in this repo.

## Weekly gates

A week isn't done when the days are done. It's done when I can pass its gate cold. The gates are
in the roadmap.