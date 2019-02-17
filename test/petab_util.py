import git


repo_base = "doc/example/tmp/benchmark-models/"
try:
    git.Git().clone("git://github.com/LoosC/Benchmark-Models.git",
                    repo_base, depth=1)
except Exception:
    git.Git(repo_base).pull()

# model folder base
folder_base = repo_base + "hackathon_contributions_new_data_format/"
