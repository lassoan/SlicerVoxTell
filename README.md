# SlicerVoxTell

3D Slicer extension for [VoxTell](https://github.com/MIC-DKFZ/VoxTell) — a free-text promptable universal 3D medical image segmentation model.

VoxTell accepts free-form natural language descriptions (e.g., "liver", "right kidney", "brain tumor") and generates volumetric 3D segmentation masks for CT, MRI, and PET images.

## Features

- 🗣️ **Text-based prompting**: Segment anatomical structures and pathologies using natural language
- 🧠 **Multi-modality support**: CT, MRI, and PET volumetric data
- 🔌 **Seamless 3D Slicer integration**: Select volumes from the scene, view segmentation results as segments
- ⚙️ **Automatic dependency installation**: Install `voxtell` and download model weights from within 3D Slicer

## Installation

1. Open 3D Slicer
2. Go to **Edit > Application Settings > Modules** and add the path to this repository, or install via the Extension Manager once the extension is published
3. Restart 3D Slicer

## Usage

1. Open the **VoxTell** module (under **Segmentation**)
2. **Install dependencies**: Click **Install dependencies** to install the `voxtell` Python package
3. **Get the model**: Either:
   - Click **Download model** to automatically download the model weights from Hugging Face (~4 GB), or
   - Specify the path to an existing model directory
4. **Select input**: Choose a volume from the scene
5. **Enter prompts**: Type free-text descriptions, one per line (e.g., `liver`, `right kidney`, `spleen`)
6. **Run segmentation**: Click **Run segmentation**

The results appear as segments in a new segmentation node named `<VolumeName>_VoxTell`.

## Important Notes

- ⚠️ **Image Orientation**: Images must be in RAS orientation for correct anatomical localization. VoxTell uses `NibabelIOWithReorient` for reorientation.
- **GPU recommended**: A GPU with ≥8 GB VRAM is strongly recommended for reasonable inference speed.
- **Research use only**: VoxTell is a research tool and should not be used for clinical decision-making without expert review.

## Citation

If you use this extension in your research, please cite:

```bibtex
@misc{rokuss2025voxtell,
      title={VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation},
      author={Maximilian Rokuss and Moritz Langenberg and Yannick Kirchhoff and Fabian Isensee and Benjamin Hamm and Constantin Ulrich and Sebastian Regnery and Lukas Bauer and Efthimios Katsigiannopulos and Tobias Norajitra and Klaus Maier-Hein},
      year={2025},
      eprint={2511.11450},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.11450},
}
```
