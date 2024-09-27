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

    # files is used to build the git add command which needs to look like this: git add file1 file2 file3
    files = ""
    for file in website_files:
        files += os.path.join(root_layers, file) + " "

    # execute the amendment workflow
    # create a subprocess that starts an ssh agent that contains the credential for the repository we want to push the update to
    # it also amends this comment to the most recent one and force pushes that to the 'website' branch
    # this way we update the website branch without cluttering up the commit messages (-f is necessary because we are rewriting the commit history)
    # if we dont want to force push, we cannot use the amend option
    # pushing to the 'website' branch also updates the Github Pages website
    pr = subprocess.Popen(
        f"eval $(ssh-agent); git add {files[:-1]}; git commit --amend -m \"Update website $(date '+%Y-%m-%d %H:%M:%S %Z')\"; git push -f origin @:website",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )

    # flag that helps communicating to the log file whether the website update was successfully pushed
    try:
        out, error = pr.communicate()
        success = True
    except Exception as e:
        logging.exception(e)
        success = False

    if success:
        logging.info("Website update was pushed successfully.")


def main():
    update_website()


if __name__ == '__main__':
    main()
