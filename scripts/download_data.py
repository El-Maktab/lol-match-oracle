import subprocess
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "paololol/league-of-legends-ranked-matches",
            "-p",
            str(output_dir),
            "--unzip",
            "--force",
        ],
        check=True,
    )

    print(f"Files downloaded to {output_dir}")


if __name__ == "__main__":
    main()
