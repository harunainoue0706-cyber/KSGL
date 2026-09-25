# Anonymization checklist

Before creating or updating the public review repository:

- do not add author names, affiliations, personal e-mail addresses, profile links, or acknowledgements;
- do not add `CITATION.cff` until de-anonymization;
- do not commit `.git/` history copied from a non-anonymous repository;
- do not commit local absolute paths, shell histories, scheduler logs, checkpoints, or run locks;
- keep external experiment logging disabled unless using an anonymous account;
- review `git status` before every push.

A simple content scan can be run with:

```bash
grep -RniE '(@[^ ]+\.[A-Za-z]+|/Users/|/home/|/root/)' . \
  --exclude-dir=.git \
  --exclude='*.npy' \
  --exclude='*.npz'
```
