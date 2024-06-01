import sys
import subprocess
import os
import logging

from config import ROOT_DIR


root_layers = os.path.join(ROOT_DIR, "layers")
# TODO: probably easier to look for all files in ../layers that have changed and simply commit those
website_files = ["OSMnxedges_1.js"]

def update_website():
    """
    Create a subprocess that pushes changes to the website related files to the GitHub repository.
    The command chain always uses the currently checked out branch as the source and the website branch
    as the destination.
    We amend the newest commit with the previous one to reduce clutter in the commit history.
    """

    files = ""
    for file in website_files:
        files += os.path.join(root_layers, file) + " "

    # check if we are on the 'website' branch
    check_branch = subprocess.Popen(
        "echo \"$(git rev-parse --abbrev-ref HEAD)\"",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    try:
        out, error = check_branch.communicate()
        print(out)
        if out.strip() != "website":
            raise Exception("We are not on the correct branch to publish the newest changes of the website. Please switch to the branch 'website'.")
    except Exception as e:
        logging.exception(e)

    # execute the amendment workflow
    # used 'git commit --amend -C HEAD' before to simply reuse the previous commit message
    # git commit -m \"Update website\"; git reset --soft HEAD~1; between git add and git commit--amend
    pr = subprocess.Popen(
        f"eval $(ssh-agent); git add {files[:-1]}; git commit --amend -m \"Update website $(date '+%Y-%m-%d %H:%M:%S %Z')\"; git push -f origin @:website",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    try:
        out, error = pr.communicate()
#        print(out)
    except Exception as e:
        logging.exception(e)

    logging.info("Website update was pushed successfully.")


def main():
    update_website()


if __name__ == '__main__':
    main()
