"""Editable project details. Never put patient details or passwords here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    project_name: str = "Skin Review"
    team_name: str = "Melanoma Image AI project team"
    team_email: str = ""
    project_website: str = ""
    location: str = "Australia"
    demo_notice: str = ""


SOURCES = [
    ("Melanoma information", "https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma"),
    ("Signs and changes to look for", "https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma/signs-and-risk-factors"),
    ("How melanoma is diagnosed", "https://connect.cancer.org.au/cancer-information/types-of-cancer/melanoma/diagnosis"),
    ("Find a GP or health service", "https://www.healthdirect.gov.au/australian-health-services"),
    ("Healthdirect advice", "https://www.healthdirect.gov.au/contact-us"),
]

STAGES = [
    ("original", "Your original image", "The dermoscopic image supplied to the software. Both the full image and a lesion crop are used for classification."),
    ("mask", "Estimated lesion boundary", "Attention U-Net estimates which pixels belong to the lesion. White marks the predicted lesion, not cancer cells."),
    ("mask_overlay", "Outline on your image", "The predicted boundary is placed over the original image so its position can be checked. It can include or miss parts of the lesion."),
    ("roi", "Closer view of the lesion", "A padded crop focuses on the estimated lesion and surrounding skin. This ROI is used alongside the full image."),
    ("gradcam", "What influenced the model", "Grad-CAM highlights regions influencing the predicted class in the ROI branch. It is not a cancer map or proof of a correct prediction."),
]
