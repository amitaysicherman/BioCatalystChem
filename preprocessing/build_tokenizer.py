from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
import argparse


def ec_tokens_to_seq(ec_tokens_str):
    ec_tokens = ec_tokens_str.split()
    ec = []
    for token in ec_tokens:
        assert token[0] == "[" and token[-1] == "]"
        token = token[1:-1]
        assert token[0] in ["v", "u", "t", "q"]
        token = token[1:]
        ec.append(token)
    return "[" + ".".join(ec) + "]"


def read_files(file_paths, ec_split):
    contents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().splitlines()
        if not ec_split and "|" in lines[0]:
            lines_no_ec = [line.split("|")[0] for line in lines]
            ec = [line.split("|")[1] for line in lines]
            ec = [ec_tokens_to_seq(ec_tokens) for ec_tokens in ec]
            lines = [f"{line} | {ec_}" for line, ec_ in zip(lines_no_ec, ec)]
        contents.extend(lines)
    return contents


def build_vocab(texts):
    all_word = set()
    for text in texts:
        words = text.split()
        all_word.update(words)
    return {word: i for i, word in enumerate(all_word)}


def create_word_tokenizer(file_paths, ec_split):
    texts = read_files(file_paths, ec_split)
    vocab = build_vocab(texts)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_process = None
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    return fast_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--ec_split", type=int, default=0)
args = parser.parse_args()

file_paths = []
for dataset in ['ecreact', 'uspto']:
    for split in ['train', 'valid', 'test']:
        for side in ['src', 'tgt']:
            file_paths.append(f"datasets/{dataset}/{side}-{split}.txt")

word_tokenizer = create_word_tokenizer(file_paths, args.ec_split)
print(word_tokenizer)
print(word_tokenizer.get_vocab())

test_text = "C N C 1 C c 2 c [nH] c 3 c c c c ( c 2 3 ) C 1 C = C ( C ) C = O . N C ( = O ) C 1 = C N ( [C@@H] 2 O [C@H] ( C O P ( = O ) ( O ) O P ( = O ) ( O ) O C [C@H] 3 O [C@@H] ( n 4 c n c 5 c ( N ) n c n c 5 4 ) [C@H] ( O ) [C@@H] 3 O ) [C@@H] ( O ) [C@H] 2 O ) C = C C 1 . [H+] | [v1] [u5] [t1]"
print(f"OR: {test_text}")
encoded = word_tokenizer.encode(test_text, add_special_tokens=False)
print(f"EN: {encoded}")

decoded = word_tokenizer.decode(encoded, clean_up_tokenization_spaces=False, skip_special_tokens=True)
print(f"DE: {decoded}")
assert decoded == test_text
output_path = "./datasets/tokenizer"
if not args.ec_split:
    output_path += "_no_ec"
word_tokenizer.save_pretrained(output_path)
