# Instructions for acquiring datasets:

For ease of use please download the following datasets using these commands in **this** directory:

Precomputed LAION's got talent enhanced:
```bash
hf download Vano04/laions-got-talent-enhanced-precomputed-en --repo-type dataset --local-dir LAION
```
MELD:
```bash
hf download Vano04/MELD-Preprocessed --repo-type dataset --local-dir MELD
```
and run `sh unpack.sh`

# Reconstruct
To reconstruct the datasets as I have made them you will need to follow the steps below:

## MELD
The easier of the two, first download MELD from https://affective-meld.github.io/ or:
```bash
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
```
Then unarchive it:
```bash
tar -xvzf MELD.Raw.tar.gz
```
Proceed to unarchive test, train, and dev splits:
```bash
tar -xvzf train.tar.gz
tar -xvzf dev.tar.gz
tar -xvzf test.tar.gz
```

The files in their raw form are .mp4 video files and csv lists, to process these into 16kHz audio files we need to run a script. I have included the script in this datasets directory as `process_meld.py`, it must be moved into the downloaded extracted MELD.Raw and run with `uv run process_meld.py`.

## LAIONs Got Talent Enhanced
Preprocessing and computing this dataset is quite difficult. First the dataset needs to be downloaded in its raw form, this dataset in compressed form is ~200gb in size.

You will need to download this with the script `download_laion.py` provided in this directory by running it with `uv run download_laion.py`.

Now manually you must delete all files with ADDITIONAL_STUFF so `rm ADDITIONAL_STUFF*`. Subsequently if you will not be utilizing more than one language you may also delete anything with the prefixes `de_`, `es_`, and `fr_`for german, spanish, and french splits. Now you must unpack all tar files, this can be done by running `uv run process_LAION_tar_files.py` after placing that script from this directory inside the download directory for LAION.

Now for the precomputing step, this would have taken days of running with my own naive implementation so I optimized the script using Codex from OpenAI.

Moving `embed_ddp.py` from this directory to the LAION download directory and running it with:
```bash
uv run datasets/embed_ddp.py \
    --laion-root . \
    --output-dir ./embeddings \
    --batch-size 8 \
    --skip-existing
```
will give you a directory called embeddings with parquet files.

These can be quickly preshuffled and combined with `reshard.py` by running `uv run python reshard_dataset.py --input-dir embeddings --output-dir data_sharded` you will get the same parquet files I have in my dataset. Just copy over the `__init__.py` and `dataset.py` and you have reconstructed the dataset.
