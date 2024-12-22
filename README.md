# EEM: Explain Electricity Markets

## The `.env` file

Put the following keys/tokens in there:

```text
ENTSOE_API_KEY=asd123415-asd324-asd1-4111-1135f3fgfg
HUGGINGFACE_ACCESS_TOKEN=hf_asdfg23452sdfubg8792g34t7
```

## Updating packages

Currently the proposed way is:

```shell
uv lock --upgrade
uv sync
```
