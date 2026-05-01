# LBR: Towards Mitigating Length Bias in Large Language Models for Recommendation

## Environment

1. Download the [genre](https://github.com/facebookresearch/GENRE).

2. Change the `genre` source path in `pyproject.toml`.

    ```toml
    [tool.uv.sources]
    ...
    genre = { path = "../../software/GENRE" }  <-- Change here
    ```

3. Build environment.

    (Please install the morden python package manager `uv` first.)

    ```shell
    uv sync
    ```

4. Download the Llama-3.2-3B, replace the soft link or change the "base_model" param in scripts.

## Usage

Coming soon ...