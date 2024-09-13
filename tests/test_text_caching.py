# Initialize Redis client
import boto3
import redis
import numpy as np
from io import BytesIO
import torch
from transformers import default_data_collator
import lz4.frame
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional

class Tokenizer:
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        self.model_name = checkpoint_dir.stem
        self.use_bos = self.check_if_bos_token_used(checkpoint_dir)
        self.bos_id = None
        self.eos_id = None

        # some checkpoints have both files, `.json` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer

            self.processor = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"

            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                eos_token = config.get("eos_token")
                if bos_token is not None and isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if eos_token is not None and isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")

        elif (vocabulary_path := checkpoint_dir / "tokenizer.model").is_file():
            from sentencepiece import SentencePieceProcessor

            self.processor = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.processor.bos_id()
            self.eos_id = self.processor.eos_id()
        else:
            raise NotImplementedError

        # NOTE: A temporary fix until it's resolved on Tokenizers side.
        # LlaMA tokenizer strips leading spaces if to decode a single token at a time.
        # https://github.com/huggingface/transformers/issues/31643
        self.apply_decoding_fix = None
        if (config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            with open(config_path, encoding="utf-8") as fp:
                self.apply_decoding_fix = "LlamaTokenizer" in json.load(fp)["tokenizer_class"]

    @property
    def vocab_size(self) -> int:
        if self.backend == "huggingface":
            return self.processor.get_vocab_size(with_added_tokens=False)
        if self.backend == "sentencepiece":
            return self.processor.vocab_size()
        raise RuntimeError

    def token_to_id(self, token: str) -> int:
        if self.backend == "huggingface":
            id_ = self.processor.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.processor.piece_to_id(token)
        else:
            raise RuntimeError
        if id_ is None:
            raise ValueError(f"token {token!r} not found in the collection.")
        return id_

    def check_if_bos_token_used(self, checkpoint_dir: Path) -> bool:
        if not (tokenizer_config_path := checkpoint_dir / "tokenizer_config.json").is_file():
            return False
        with open(tokenizer_config_path, encoding="utf-8") as fp:
            config = json.load(fp)
        # for LlaMA-3 tokenizer there is no `add_bos_token` at all and `tokenizer_class` is only
        # `PreTrainedTokenizerFast`
        if checkpoint_dir.stem.startswith("Meta-Llama-3"):
            return True
        if "add_bos_token" in config:
            return config["add_bos_token"]
        # if `add_bos_token` isn't in the config file, but LLaMA tokenizer is used - return True.
        # ex: https://huggingface.co/stabilityai/StableBeluga2/blob/main/tokenizer_config.json#L2
        return config.get("tokenizer_class") == "LlamaTokenizer"

    def encode(
        self,
        string: str,
        device: Optional[torch.device] = None,
        bos: Optional[bool] = None,
        eos: bool = False,
        max_length: int = -1,
    ) -> torch.Tensor:
        if self.backend == "huggingface":
            tokens = self.processor.encode(string).ids
        elif self.backend == "sentencepiece":
            tokens = self.processor.encode(string)
        else:
            raise RuntimeError(f"`{self.backend}` is not supported.")
        if tokens is None:
            raise ValueError("`self.processor` returned tokens of None value.")

        if bos or (bos is None and self.use_bos):
            if self.bos_id is None:
                raise NotImplementedError("This tokenizer does not have a defined bos token.")
            if not tokens or tokens[0] != self.bos_id:
                tokens = [self.bos_id] + tokens
        # if the processor misbehaves and adds `bos` token no matter what
        elif tokens and tokens[0] == self.bos_id:
            tokens = tokens[1:]

        if eos and (not tokens or tokens[-1] != self.eos_id):
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]
        return torch.tensor(tokens, dtype=torch.int, device=device)

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        if len(tokens) == 1 and self.apply_decoding_fix:
            dummy_token_id = 33  # \x1e
            dummy_token = self.processor.decode([dummy_token_id])
            return self.processor.decode([dummy_token_id] + tokens)[len(dummy_token) :]
        return self.processor.decode(tokens)


def main():
    cache_client = redis.StrictRedis(host="localhost", port=6379)
    tokenizer = Tokenizer('tests\pythia-14m-tokenizer')
    max_length = 512
    
    bucket_name = 'owt-5mb-text-chunks'
    data_path = 'train/chunk-0-5.bin'
    s3 = boto3.client('s3')

    # Fetch the binary file from S3
    response = s3.get_object(Bucket=bucket_name, Key=data_path)
    content = response['Body'].read()
    # Store the content in Redis
    cache_client.set(1, content)
    # Retrieve the content from Redis
    binary_data = cache_client.get(1)
    # Create a BytesIO stream from the binary data
    binary_stream = BytesIO(binary_data)
    index = 0

    tokenized_docs = []

    while True:
        offset = (1 + (index - 0) if index >= 0 else index + 1) * 4
        # Read the entire content of the binary file
        binary_stream.seek(offset)
        pair = binary_stream.read(8)
        begin, end = np.frombuffer(pair, np.uint32)
        if begin.item() == len(binary_stream.getvalue()):
            break
        binary_stream.seek(begin)
        raw_item_data = binary_stream.read(end - begin)

        shift_idx = 4
        sizes = np.frombuffer(raw_item_data[:shift_idx], np.uint32)
        data = ""
        for size, data_format in zip(sizes, 'str'):
            # size = size.item()
            data_bytes = raw_item_data[shift_idx : shift_idx + size]
            data += data_bytes.decode('utf-8')
            shift_idx += size
        index += 1
        tokenized_docs.append(tokenizer.encode(data, eos=True))
    
    tokenized_samples = gen_samples(tokenized_docs, max_length)

    # print(f"Tokenized data: {tokenized_data}")
    tokens_as_bytes = _torch_tenors_to_bytes(tokenized_samples)
    cache_client.set(2, tokens_as_bytes)

    # Retrieve the tokenized data from Redis
    byte_data = cache_client.get(2)
    tokenized_samples = _bytes_to_torch_batch(byte_data)
    pass


def gen_samples(tokenized_data, block_size):
        # Create a list to hold samples of size self.context_size
        samples = []
        for item in tokenized_data:
            chunk_size = block_size + 1  # Define the chunk size
            # Split ids into chunks of size block_size+1
            for i in range(0, item.size(0), chunk_size):
                # Extract a chunk from the ids
                chunk = item[i:i + chunk_size]
                # Pad the last chunk if it is smaller than block_size+1
                if chunk.size(0) < chunk_size:
                    padding_length = chunk_size - chunk.size(0)
                    padding = torch.full((padding_length,), fill_value=0, dtype=torch.long)
                    chunk = torch.cat((chunk, padding))

                input_ids = chunk[0:block_size].contiguous().long()
                targets = chunk[1:block_size + 1].contiguous().long()
                samples.append((input_ids, targets))
        return samples

def truncate_or_pad_sequence(sequence, max_length):
    # Truncate sequence if longer than max_length
    if sequence.size(1) > max_length:
        return sequence[:, :max_length]
    # Pad sequence if shorter than max_length
    elif sequence.size(1) < max_length:
        pad_length = max_length - sequence.size(1)
        return torch.cat([sequence, torch.zeros(sequence.size(0), pad_length, dtype=sequence.dtype)], dim=1)
    return sequence

def _torch_tenors_to_bytes(tokenized_data: torch.Tensor) -> str:
        with BytesIO() as buffer:
            torch.save(tokenized_data, buffer)
            compressed_byte_data = lz4.frame.compress(buffer.getvalue())
        return compressed_byte_data

def _bytes_to_torch_batch(bytes_minibatch) -> tuple:
        compressed_batch = lz4.frame.decompress(bytes_minibatch)
        with BytesIO(compressed_batch) as buffer:
            data_samples = torch.load(buffer)
        return data_samples


if __name__ == '__main__':
    main()