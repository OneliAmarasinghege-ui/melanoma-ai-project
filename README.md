# Melanoma Image AI — Group 12 project repository

Attention U-Net lesion segmentation, dual-view EfficientNetB0 classification,
evaluation and comparisons, and the Skin Review Gradio application.

## Quick links

- [Open the patient-friendly app notebook in Colab](https://colab.research.google.com/github/OneliAmarasinghege-ui/melanoma-ai-project/blob/main/notebooks/software/Melanoma_Patient_Friendly_App.ipynb)
- [Browse the selected Attention U-Net notebooks](notebooks/attention_unet)
- [Browse the team pipeline notebooks](notebooks/team_pipelines)
- [Browse the app source](software)

## Start here

### Open the patient-friendly software

1. Extract this ZIP on your computer.
2. In Google Colab, choose **File → Upload notebook** and open
   `notebooks/software/Melanoma_Patient_Friendly_App.ipynb`.
3. Run the cells from the top. Allow Google Drive access when requested.
4. Set `RUN_DIR` to your existing completed model run if automatic discovery
   cannot find it. Enter your team contact details in the configuration cell.
5. Run the last cell and open the displayed Gradio link. The runtime must stay
   connected while the demonstration is used.

This notebook contains the app source. You do not need to upload a `.py` file
through Colab's **Upload notebook** window.

### Recreate training, evaluation and report outputs

| Purpose | Notebook under `notebooks/attention_unet/` |
| --- | --- |
| Train the final-candidate configuration | `Melanoma_Attention_UNet_Final_Candidate.ipynb` |
| Evaluate saved models, calculate intervals, make comparisons and export reports | `Melanoma_Attention_UNet_Combined_Reporting_CIs_and_Comparisons(1).ipynb` |
| Earlier reporting and app workflow | `Melanoma_Attention_UNet_Reporting_and_App_1377.ipynb` |
| Supplemental confidence-interval analysis | `Melanoma_Results_Confidence_Intervals_and_Comparisons.ipynb` |

Use a Colab GPU for training. Read and run each notebook from the top; its setup
cells define its dependencies, paths and mode. The combined notebook defaults to
`evaluate_saved`: it expects existing saved weights. Switch to `train_new` only
when you intend to create a new run. Training is not required to open the app if
your completed run is already in Drive. A new training run is a new experiment
and is not guaranteed to reproduce the exact previously reported scores.

## Repository contents

- `notebooks/attention_unet/`: selected training, reporting and analysis workflows.
- `notebooks/software/`: the patient-friendly Skin Review launcher.
- `notebooks/comparisons/`: image-stage comparisons across pipelines.
- `notebooks/team_pipelines/`: the available ResUNet, U-Net++ and TransUNet notebooks.
- `software/`: original Skin Review Python source, CSS, requirements and two ISIC examples.
- `archive/notebooks/`: all distinct recovered notebook files, with embedded outputs cleared for GitHub.
- `archive/scripts/`: recovered standalone Python scripts.
- `archive/figure_bundles/`: earlier figure ZIP files.
- `docs/`: software Results document and the seven supplied interface screenshots.
- `NOTEBOOK_INDEX.csv`: original filenames, archive locations and duplicate mapping.
- `FILE_MANIFEST.json`: hashes and sizes of the packaged files.

All notebook copies in this GitHub repository have embedded outputs cleared;
cell source is unchanged. The distinct historical versions remain in the archive.
Original notebooks with saved outputs are retained in the full project ZIP
provided separately. Exact duplicate uploads share one archive file and are
mapped in the index. Earlier experiments do not override the selected workflow.

## Saved model files needed to run inference

Your trained weights and complete saved run remain in Google Drive. They were
not among the recovered source files and are not included in this ZIP. The app
expects a run containing at least:

```
your_run/
  models/attention_unet_best.keras
  models/selected_final_classifier.keras
  tables/selected_configuration.json
```

Keep any additional configuration, manifest, tables and figures from the same
run together. The dataset itself is not bundled; the training/evaluation
notebooks contain their original download and path setup code.

## Run the Python application locally

Follow `software/README.md` for the original application documentation. From this
repository directory, install the app's requirements in an isolated environment:

```bash
python -m pip install -r software/requirements.txt
python software/app.py --run-dir /absolute/path/to/your_run
```

Other pipelines retain their own notebook dependency cells. The application
requirements are not a shared environment specification for every historical
experiment.

## Checks on this package

Notebook JSON and cell structure were checked, the working notebook code was
checked for syntax after Colab magic transformation, and the extracted app Python
files were compiled without running them. See `PACKAGE_CHECKS.json` for the
outcome. Training, model inference and live UI tests were not rerun when this
archive was assembled; the trained weights were unavailable here.

This is a student research prototype. Its outputs are research predictions and
do not establish a clinical diagnosis. The example files are public ISIC images.
The app's original README contains source and interpretation details.

## Repository

This repository is published at https://github.com/OneliAmarasinghege-ui/melanoma-ai-project.

Checkpoint files, runtime downloads and temporary reports are excluded by the
supplied `.gitignore`. Real patient information should not be added to this
research repository.
