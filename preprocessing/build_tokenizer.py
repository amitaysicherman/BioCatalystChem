import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
import argparse

PAD = "[PAD]"
EOS = "[EOS]"
UNK = "[UNK]"
SPACIAL_TOKENS = {PAD: 0, EOS: 1, UNK: 2}

def encode_eos_pad(tokenizer, text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    tokens = tokens + [tokenizer.eos_token_id]
    if len(tokens) > max_length:
        return None, None
    n_tokens = len(tokens)
    padding_length = max_length - len(tokens)
    if padding_length > 0:
        tokens = tokens + [tokenizer.pad_token_id] * padding_length
    mask = [1] * n_tokens + [0] * padding_length
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)
    return tokens, mask


def get_first_ec_token_index(tokenizer: PreTrainedTokenizerFast, ec_split):
    for index in range(len(tokenizer.get_vocab())):
        token = tokenizer.decode(index)
        if ec_split and token.startswith("[v"):
            return index
        if not ec_split and token.startswith("[ec:"):
            return index


def get_ec_order(tokenizer: PreTrainedTokenizerFast, ec_split=0):
    assert ec_split == 0
    ec_order = []
    for index in range(len(tokenizer.get_vocab())):
        token = tokenizer.decode(index)
        if token.startswith("[ec:"):
            ec_order.append(unwrap_ec(token))
    return ec_order


def get_tokenizer_file_path():
    output_path = "./datasets/tokenizer"
    # if not ec_split:
    #     output_path += "_no_ec"
    # print(f"Tokenizer file path: {output_path}")
    return output_path


def wrap_ec(ec):
    return f"[ec:{ec}]"


def unwrap_ec(ec):
    return ec.split(":")[1][:-1]


def ec_tokens_to_seq(ec_tokens_str):
    ec_tokens = ec_tokens_str.split()
    ec = []
    for token in ec_tokens:
        assert token[0] == "[" and token[-1] == "]"
        token = token[1:-1]
        assert token[0] in ["v", "u", "t", "q"]
        token = token[1:]
        ec.append(token)
    return wrap_ec(".".join(ec))


def redo_ec_split(text,return_smiles_num=False):
    if "|" not in text:
        return text
    text_no_ec = text.split("|")[0].strip()
    ec = text.split("|")[1].strip()
    ec = ec_tokens_to_seq(ec)
    if return_smiles_num:
        ec=ec.replace("[","").replace("]","").replace("ec:","")
        return text_no_ec, ec
    return f"{text_no_ec} | {ec}"


def read_files(file_paths, ec_split):
    contents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
        if not ec_split and "|" in lines[0]:
            lines = [redo_ec_split(line) for line in lines]
        contents.extend(lines)
    return contents


def build_vocab(texts, ec_split):
    all_word = set()
    for text in texts:
        words = text.split()
        all_word.update(words)
    words_dict = {**SPACIAL_TOKENS}
    no_ec_words = set()
    ec_words = set()
    for word in all_word:
        if ec_split and word.startswith("[") and word[1] in ["v", "u", "t", "q"]:
            ec_words.add(word)
            continue
        if not ec_split and word.startswith("[ec:"):
            ec_words.add(word)
            continue
        no_ec_words.add(word)
    for word in no_ec_words:
        words_dict[word] = len(words_dict)
    for word in ec_words:
        words_dict[word] = len(words_dict)
    return words_dict


def create_word_tokenizer(file_paths, ec_split):
    texts = read_files(file_paths, ec_split)
    vocab = build_vocab(texts, ec_split)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_process = None
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD,
    )

    return fast_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ec_split", type=int, default=0)
    args = parser.parse_args()

    file_paths = []

    datasets = ['ecreact/level4', 'uspto']
    for dataset in datasets:
        for split in ['train', 'valid', 'test']:
            for side in ['src', 'tgt']:
                file_paths.append(f"datasets/{dataset}/{side}-{split}.txt")

    word_tokenizer = create_word_tokenizer(file_paths, args.ec_split)
    print(word_tokenizer)

    if args.ec_split:
        test_text = "O = S ( = O ) ( [O-] ) S | [v1] [u8] [t5] [q2]"
    else:
        test_text = "O = S ( = O ) ( [O-] ) S | [ec:1.8.5.2]"
    print(f"OR: {test_text}")
    encoded = word_tokenizer.encode(test_text, add_special_tokens=False)
    print(f"EN: {encoded}")
    decoded = word_tokenizer.decode(encoded, clean_up_tokenization_spaces=False, skip_special_tokens=True)
    print(f"DE: {decoded}")
    assert decoded == test_text
    output_path = get_tokenizer_file_path(args.ec_split)
    word_tokenizer.save_pretrained(output_path)


