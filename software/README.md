# Skin Review - redesigned Attention U-Net research app

Use **Melanoma_Patient_Friendly_App.ipynb** in Google Colab for the easiest setup.
The notebook contains all application code and two public ISIC example images.
It loads the models already saved in your Google Drive. It does not train a model.

## Colab

1. In Colab, choose **File > Upload notebook** and select the `.ipynb` file.
2. Run the install cell. In the setup cell, add your real team name, email and
   optional HTTPS website. You can leave these blank while testing.
3. Run the load cell and allow the usual Google Drive mount. It looks for the
   completed reporting run first; set `RUN_DIR` yourself if your folder moved.
4. Run the launch cell. Use the displayed app link while the runtime stays connected.

Do not upload the Python files as a notebook. They are provided in the source ZIP
for your team to inspect or run locally. No separate Python upload is needed in Colab.

## What changed

- Calm, responsive result page with a plain-language explanation and practical next steps.
- Expandable result interpretation and technical score/cutoff, with no cancer-risk percentage.
- Five real pipeline images and a sixth classification explanation, each with a short caption.
- Optional notes and an image-linked PDF report that the user downloads and shares themselves.
- Melanoma education, Australian health/support contacts, and configurable project contact details.
- Existing saved evaluation tables, test losses and training curves remain available in a secondary section.
- Selecting another image clears the old result, notes and report. Unreadable images, missing
  heatmaps and unreliable lesion crops receive explicit messages.

The existing `attention_pipeline_core.py` is copied without changes. The original
image file is passed directly to inference. The original ROI encoding, image sizes,
saved inference averaging and model cutoffs are retained. This changes the interface,
not the trained model's accuracy, sensitivity, specificity or Dice score.

## Required saved run

```
your_run/
  models/attention_unet_best.keras
  models/selected_final_classifier.keras
  tables/selected_configuration.json
```

Optional: `tables/segmentation_configuration.json`, `deployment_manifest.json`,
`tables/complete_classification_test_metrics.csv`,
`tables/complete_segmentation_test_metrics.csv`, and `figures/*.png`.
If evaluation files are absent, the app says so. It does not manufacture results.
Checkpoint hashes are checked when the saved provenance/manifest provides them.

Preferred existing reporting run:
`/content/drive/MyDrive/Melanoma_Attention_UNet_Reporting_and_App/20261012T051542_296259Z`

Fallback frozen run:
`/content/drive/MyDrive/Melanoma_ISIC2016_Final_Candidate/20261007T163637_386595Z`

## Local use

With Python 3.11 or 3.12, install the requirements in a virtual environment, then run:

```
python -m pip install -r requirements.txt
python app.py --run-dir /absolute/path/to/your_run
```

Optional environment variables: `PROJECT_NAME`, `TEAM_NAME`, `TEAM_EMAIL`,
`PROJECT_WEBSITE`. Local use does not create a public link unless `--share` is added.
For a Colab demonstration the notebook creates a temporary public Gradio link.
Only de-identified research images should be used in a public demo. Reports and
uploads may remain in temporary runtime storage until cleanup. This is not a
patient-record service, an authenticated clinical portal or a clinical deployment.

## Interpretation

This is a student research prototype, not a diagnostic service. The model can miss
melanoma and flag non-cancerous lesions. Its score is not calibrated cancer risk.
A new, changing or otherwise concerning spot needs clinical assessment regardless
of the model output. Grad-CAM is an explanation, not a cancer map. No patient-specific
Dice/IoU is shown without a ground-truth mask. A biopsy may be needed for diagnosis.

The project test set was viewed during development; saved evaluation results are
exploratory, not independent clinical validation. The software redesign does not
make the model clinically validated.

## Contact and education sources

Patient information is paraphrased from Cancer Council and Healthdirect Australia;
links are in the app, report and `settings.py`. Source links checked 16 October 2026.
Healthdirect nurse advice: 1800 022 222. Cancer Council / CancerConnect: 13 11 20.
Replace the project contact setup fields with your team's actual details.

The example files are public ISIC 2016 research images, ISIC_0000003 and
ISIC_0000013, from the ISIC Archive. They are not images of a user of this app.
Official challenge source: https://challenge.isic-archive.com/landing/2016/

## Checks completed for this update

The package is checked for Python/notebook syntax, result handling, image validation,
report contents, clearing and session separation, and desktop/mobile layout.
The inference adapter is checked with synthetic model fixtures. Any preview or
sample PDF uses a clearly labelled replay of a saved public ISIC example.
Your actual frozen model weights remain in Google Drive and must be loaded in Colab
for the final check on that runtime. Preview outputs are not a new evaluation run.
