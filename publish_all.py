import pathlib
import subprocess

all_paths = pathlib.Path(".").glob("**/pyproject.toml")


for path in all_paths:
    build_command = [
        "poetry",
        "build",
        "--no-interaction",
        "-C",
        str(path.parent.absolute()),
    ]
    publish_command = [
        "poetry",
        "publish",
        "--no-interaction",
        "--skip-existing",
        "--repository",
        "mlcommons",
        "-C",
        str(path.parent.absolute()),
    ]

    subprocess.run(build_command, check=True)
    subprocess.run(publish_command, check=True)
