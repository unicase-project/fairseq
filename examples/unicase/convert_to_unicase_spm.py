import fire

import sentencepiece_model_pb2 as model
import sentencepiece as spm
import json


def title_case(word):
    """
    Change first letter to upper, rest leave lower-case
    :param word: lower-case word
    :return: modified word
    """
    first_letter_idx = None
    for idx, ch in enumerate(word):
        if is_letter_case(ch):
            first_letter_idx = idx
            break

    new_word = word[:first_letter_idx] + word[first_letter_idx].upper() + word[first_letter_idx+1:]

    return new_word


def is_letter_case(character):
    """
    Check whether character is a letter case
    :param character: character to check
    :return: boolean
    """
    return character.lower() != character.upper()


def is_valid_token(word):
    """
    Valid tokens contain 2 or more letters
    Example:
    Valid tokens: "(this", "_john", "ing:", "it"
    Invalid tokens: ""a", 123,
    :param word:
    :return:
    """

    if word[0] == "<" and word[-1] == ">":
        return False
    if word != word.lower():
        print(f"found not lowercase word: {word}")
        return False
    cased_ct = sum([1 for ch in word if is_letter_case(ch)])
    return cased_ct >= 2


def is_letter_token(word):
    """
    Valid tokens contain 2 or more letters
    Example:
    Valid tokens: "(this", "_john", "ing:", "it"
    Invalid tokens: ""a", 123,
    :param word:
    :return:
    """

    if word[0] == "<" and word[-1] == ">":
        return False
    if word != word.lower():
        print(f"found not lowercase word: {word}")
        return False
    cased_ct = sum([1 for ch in word if is_letter_case(ch)])
    return cased_ct == 1


def convert_spm_to_unicase(spm_lower_model, out_prefix):
    m = model.ModelProto()
    m.ParseFromString(open(spm_lower_model, 'rb').read())

    vocab = []
    vocab_size = len(m.pieces)

    for p in m.pieces:
        vocab.append((p.piece, p.score, p.type))

    cased_tokens = []
    uncased_tokens = []

    added_uncased_tokens = {}

    for ttext, tscore, ttype in vocab:
        if ttext != ttext.lower():
            print(f"found not lowercase word: {ttext}")
            continue
        elif is_valid_token(ttext):
            cased_tokens.append((ttext, tscore, ttype))
            cased_tokens.append((title_case(ttext), tscore, ttype))
            cased_tokens.append((ttext.upper(), tscore, ttype))
        else:
            uncased_tokens.append((ttext, tscore, ttype))
            added_uncased_tokens[ttext] = 1
            # for single letter tokens also add an uppercased token
            if is_letter_token(ttext):
                if ttext.upper() in added_uncased_tokens:
                    print(f"token is already added: {ttext.upper()}")
                    continue
                uncased_tokens.append((ttext.upper(), tscore, ttype))
                added_uncased_tokens[ttext.upper()] = 1

    # remove all items - TODO: must be more clever way to do it
    for i in range(vocab_size):
        m.pieces.remove(m.pieces[0])

    # add all items
    for ttext, tscore, ttype in uncased_tokens + cased_tokens:
        m.pieces.add(piece=ttext, score=tscore, type=ttype)

    config = {}
    config["original_lowercase_vocab_size"] = vocab_size
    config["vocab_size"] = len(m.pieces)
    config["non_word_tokens"] = len(uncased_tokens)
    config["word_tokens"] = len(cased_tokens)
    config["num_of_shapes"] = 3

    assert config["word_tokens"] % config["num_of_shapes"] == 0

    with open(out_prefix + ".model", "wb") as f:
        f.write(m.SerializeToString())

    with open(out_prefix + ".config", "w") as f:
        json.dump(config, f)

    # Test encoding
    text = "richer personas deepcopy copycat copywriter witcher i'm international speedup coresponding"
    orig = spm.SentencePieceProcessor()
    orig.load(filename=spm_lower_model)
    new = spm.SentencePieceProcessor()
    new.load(filename=out_prefix + ".model")

    print(f"ORG: {orig.encode_as_pieces(text)}")
    print(f"ORG: {orig.encode_as_pieces(text.upper())}")

    print(f"NEW: {new.encode_as_pieces(text)}")
    print(f"NEW: {new.encode_as_pieces(text.upper())}")

    print(f"ORG: {orig.encode_as_ids(text)}")
    print(f"ORG: {orig.encode_as_ids(text.upper())}")
    print(f"NEW: {new.encode_as_ids(text)}")
    print(f"NEW: {new.encode_as_ids(text.upper())}")


if __name__ == "__main__":
    fire.Fire(convert_spm_to_unicase)