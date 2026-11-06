"""Local entry point. Colab users should run the supplied notebook instead."""
from pathlib import Path
import argparse
import os

from attention_pipeline_core import MelanomaPipeline
from settings import AppSettings
from skin_review_ui import launch_app


def main():
    parser = argparse.ArgumentParser(description="Skin Review research app")
    parser.add_argument("--run-dir", default=os.environ.get("MELANOMA_RUN_DIR", ""))
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link; research images only")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not args.run_dir:
        parser.error("Supply --run-dir pointing to the completed reporting/model run.")
    settings = AppSettings(project_name=os.environ.get("PROJECT_NAME", "Skin Review"),
                           team_name=os.environ.get("TEAM_NAME", "Melanoma Image AI project team"),
                           team_email=os.environ.get("TEAM_EMAIL", ""),
                           project_website=os.environ.get("PROJECT_WEBSITE", ""))
    samples = sorted(Path(__file__).with_name("examples").glob("*.jpg"))
    launch_app(MelanomaPipeline.load(args.run_dir), settings, samples, share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
