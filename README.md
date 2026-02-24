# SlicerVoxTell

SlicerVoxTell is a 3D Slicer extension that segments structures in CT, MRI, and PET volumes from plain-language text prompts (for example, "liver" or "right kidney").
It is intended for research workflows where users need fast, prompt-based 3D segmentation directly in Slicer.

![SlicerVoxTell screenshot](https://raw.githubusercontent.com/lassoan/SlicerVoxTell/main/screenshot01.jpg)

Developers of this extension are not affiliated with developers of the underlying [VoxTell model](https://github.com/MIC-DKFZ/VoxTell) developed at DKFZ by Maximilian Rokuss et al.

## Modules

- **VoxTell**: Runs VoxTell free-text 3D segmentation on the selected input volume and creates/updates a Slicer segmentation node with one segment per prompt.


## Installation

GPU with at least 8 GB VRAM is strongly recommended for practically usable inference speed (typically less than 1 minute).

1. Open 3D Slicer.
2. Install `VoxTell` extension from Extension Manager.
3. Restart 3D Slicer.

## Tutorial


1. Open **Sample Data** and load **CTACardio**.
2. Open **VoxTell** (category: **Segmentation**).
3. In **Setup**, click **Install dependencies and model**
4. In **Inputs**, select the input volume (CTACardio should be selected by default).
5. Enter prompts, one per line (for example: `liver`, `ribs`, `vertebrae`, `aorta`).
6. Choose device (**GPU (CUDA)** recommended when available) and click **Run segmentation**.
7. If a GPU is available, segmentation result should be available in about 30 seconds.

## Publication
The extension uses VoxTell, which is described in this paper (arXiv): https://arxiv.org/abs/2511.11450

If you use this extension in your research, please cite:

```bibtex
@misc{rokuss2025voxtell,
      title={VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation},
      author={Maximilian Rokuss and Moritz Langenberg and Yannick Kirchhoff and Fabian Isensee and Benjamin Hamm and Constantin Ulrich and Sebastian Regnery and Lukas Bauer and Efthimios Katsigiannopulos and Tobias Norajitra and Klaus Maier-Hein},
      year={2025},
      eprint={2511.11450},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.11450}
}
```

## Safety and Privacy

- The extension does **not** send data to any external services.
- Model files are downloaded from the official VoxTell Hugging Face repository (`mrokuss/VoxTell`).

## License

This extension is distributed under the **MIT License** (see [LICENSE.txt](LICENSE.txt)).

## Support and Maintenance

- Maintainers monitor repository issues and pull requests.
- Maintainers also respond to @mentions on the Slicer Forum: https://discourse.slicer.org
