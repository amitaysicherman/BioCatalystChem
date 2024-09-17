from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
def read_files(file_paths):
    contents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            contents.append(file.read())
    return contents


def build_vocab(texts):
    all_word = set()
    for text in texts:
        words = text.split()
        all_word.update(words)
    return {word: i for i, word in enumerate(all_word)}


def create_word_tokenizer(file_paths):
    texts = read_files(file_paths)
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


file_paths = []
for dataset in ['ecreact','uspto']:
    for split in ['train', 'valid', 'test']:
        for side in ['src', 'tgt']:
            file_paths.append(f"datasets/{dataset}/{side}-{split}.txt")

word_tokenizer = create_word_tokenizer(file_paths)
print(word_tokenizer)
print(word_tokenizer.get_vocab())

test_text = "C N C 1 C c 2 c [nH] c 3 c c c c ( c 2 3 ) C 1 C = C ( C ) C = O . N C ( = O ) C 1 = C N ( [C@@H] 2 O [C@H] ( C O P ( = O ) ( O ) O P ( = O ) ( O ) O C [C@H] 3 O [C@@H] ( n 4 c n c 5 c ( N ) n c n c 5 4 ) [C@H] ( O ) [C@@H] 3 O ) [C@@H] ( O ) [C@H] 2 O ) C = C C 1 . [H+] | [v1] [u5] [t1]"
print(f"OR: {test_text}")
encoded = word_tokenizer.encode(test_text, add_special_tokens=False)
print(f"EN: {encoded}")

decoded = word_tokenizer.decode(encoded, clean_up_tokenization_spaces=False, skip_special_tokens=True)
print(f"DE: {decoded}")
assert decoded == test_text
word_tokenizer.save_pretrained("./datasets/tokenizer")